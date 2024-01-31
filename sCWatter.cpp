#include <iostream>
#include <tira/optics/planewave.h>
#include "CoupledWaveStructure.h"
#include "FourierWave.h"
#include <complex>
#include <string>
#include <math.h>
#include <fstream>
#include <boost/program_options.hpp>
#include <random>
#include <iomanip>
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include <extern/libnpy/npy.hpp>
#include <chrono> 
#include <ctime>


#include "tira/optics/planewave.h"
#include "tira/field.h"
#include "third_Lapack.h"

// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif

std::vector<double> in_dir;
double in_lambda;

bool LOG = false;
bool Pz = true;				// Calculate Pz by default
std::string logprefix = "";

volume < std::complex<double> >* Volume;

std::vector<double> in_n;
std::vector<double> in_kappa;
std::vector<double> in_ex;
std::vector<double> in_ey;
std::vector<double> in_ez;
double in_z;
std::vector<double> in_size;
std::vector<size_t> num_pixels;
std::vector<double> in_normal;
std::string in_outfile;
std::string in_sample;
std::vector<int> in_coeff;
double in_n_sample;
double in_kappa_sample;
std::vector<double> in_center;
std::vector<double> in_rec_bar;
std::vector<double> in_circle;

unsigned int L;		// L=2
unsigned int ZL;	// Number of layers
int M[2];	// Set the number of the Fourier Coefficients
Eigen::MatrixXcd A;
Eigen::VectorXcd b;
Eigen::VectorXcd ni;
double* z;
std::complex<double> k;
std::vector<std::complex<double>> E0;
Eigen::VectorXcd EF;
int MF;
Eigen::RowVectorXcd Sx;				// Fourier coefficients of x component of direction
Eigen::RowVectorXcd Sy;				// Fourier coefficients of y component of direction
std::vector<Eigen::RowVectorXcd> Sz(2);		// 2D vector for the Fourier coefficients of z component for the upper and lower regions
Eigen::VectorXcd Ex, Ey, Ez;
int ei = 0;				// The current row for the matrix
int l;			// The current layer l.
std::ofstream logfile;

std::vector<Eigen::VectorXcd> eigenvalues;			// eigen values for current layer
std::vector<Eigen::MatrixXcd> eigenvectors;			// eigen vectors for current layer
std::vector<Eigen::VectorXcd> Beta;			// eigen vectors for current layer
std::vector<Eigen::MatrixXcd> GD;					// Dimension: (layers, coeffs)
std::vector<Eigen::MatrixXcd> GC;					// Dimension: (layers, coeffs)
Eigen::MatrixXcd f1;
Eigen::MatrixXcd f2;
Eigen::MatrixXcd f3;
Eigen::MatrixXcd tmp;					// Temposarily store some Eigen::MatrixXcd
Eigen::MatrixXcd tmp_2;					// Temposarily store additional Eigen::MatrixXcd
Eigen::MatrixXcd Gd_static;					
Eigen::MatrixXcd Gc_static;					
std::chrono::duration<double> elapsed_seconds;
int counts = 0;
bool psf = false;
bool EXTERNAL = false;
/// Convert a complex vector to a string for display
template <typename T>
std::string vec2str(glm::vec<3, std::complex<T> > v, int spacing = 20) {
	std::stringstream ss;
	if (v[0].imag() == 0.0 && v[1].imag() == 0.0 && v[2].imag() == 0.0) {				// if the vector is real
		ss << std::setw(spacing) << std::left << v[0].real() << std::setw(spacing) << std::left << v[1].real() << std::setw(spacing) << std::left << v[2].real();
	}
	else {
		ss << std::setw(spacing) << std::left << v[0] << std::setw(spacing) << std::left << v[1] << std::setw(spacing) << std::left << v[2];
	}
	return ss.str();
}

/// Convert a real vector to a string for display
std::string vec2str(glm::vec<3, double> v, int spacing = 20) {
	std::stringstream ss;
	ss << std::setw(spacing) << std::left << v[0] << std::setw(spacing) << std::left << v[1] << std::setw(spacing) << std::left << v[2];
	return ss.str();
}

/// Return a value in the A matrix
std::complex<double>& Mat(int row, int col) {
	return A(row, col);
}

// Enumerators used to access elements of the A matrix
enum Coord { X, Y, Z };
enum Dir { Transmitted, Reflected };

// Methods used to access elements of the matrix A based on the layer number, direction, and field coordinate
std::complex<double>& Mat(int row, int layer, Dir d, Coord c, int m, int M) {
	return A(row, (layer * 6 + d * 3 + c - 3) * M + m);
}
size_t idx(int layer, Dir d, Coord c, int m, int M) {
	return (layer * 6 + d * 3 + c - 3) * M + m;
}

/// Output the coupled wave matrix as a string
std::string mat2str(int width = 10, int precision = 2) {
	std::stringstream ss;
	ss << A;
	return ss.str();
}

// Output the b matrix (containing the unknowns) to a string
std::string b2str(int precision = 2) {
	std::stringstream ss;
	ss << b;
	return ss.str();
}

// Initialize the A matrix and b vector of unknowns
void InitMatrices() {
	A = Eigen::MatrixXcd::Zero(6 * MF * (L - 1), 6 * MF * (L - 1));									// allocate space for the matrix
	b = Eigen::VectorXcd::Zero(6 * MF * (L - 1));												// zero out the matrix
}

// set all of the layer refractive indices and boundary positions based on user input
void InitLayer_n() {
	ni.resize(L);
	ni[0] = std::complex<double>(in_n[0], 0);
	for (size_t l = 1; l < L; l++)
		ni[l] = std::complex<double>(in_n[l], in_kappa[l - 1]);			// store the complex refractive index for each layer
}

// set all of the layer refractive indices and boundary positions based on user input
void InitLayer_z() {
	z = new double[ZL];														// allocate space to store z coordinates for each interface
	z[0] = in_z - in_size[2] / 2.0;
	for (int i = 1; i < ZL; i++) {
		z[i] = z[i - 1] + in_size[2] / (num_pixels[0]);
	}
}

// The struct is to integrate eigenvalues and their indices
struct EiV {
	size_t idx;
	std::complex<double> value;
};

// Sort by eigenvalues' imaginery parts. The real parts are the tie-breaker.
bool sorter(EiV const& lhs, EiV const& rhs) {
	if (abs(lhs.value.imag() - rhs.value.imag()) > std::numeric_limits<double>::epsilon())
		return lhs.value.imag() < rhs.value.imag();
	else if (abs(lhs.value.real() - rhs.value.real()) > std::numeric_limits<double>::epsilon())
		return lhs.value.real() < rhs.value.real();
	else
		return lhs.value.imag() < rhs.value.imag();
}
/// <summary>
/// Temporarily depreacated.
/// <summary>
/// <param name="eigenvalues_unordered"></param>
/// <param name="eigenvectors_unordered"></param>
void Eigen_Sort(Eigen::VectorXcd eigenvalues_unordered, Eigen::MatrixXcd eigenvectors_unordered) {
	unsigned int len = eigenvalues_unordered.size();
	// Sort the unordered eigenvalues and track the indices
	std::vector<EiV> eiV(len);
	for (size_t i = 0; i < len; i++) {
		eiV[i].idx = i;
		eiV[i].value = eigenvalues_unordered(i);
	}
	std::sort(eiV.begin(), eiV.end(), &sorter);

	Eigen::VectorXcd evl;
	Eigen::MatrixXcd evt;
	evl.resize(len);
	evt.resize(len, len);

	//if (logfile) {
	//	logfile << "eigenvalues_unordered: " << std::endl;
	//	logfile << eigenvalues_unordered << std::endl << std::endl;
	//	logfile << "eigenvectors_unordered: " << std::endl;
	//	logfile << eigenvectors_unordered << std::endl << std::endl;
	//}
	//std::cout << "eigenvalues_unordered: " << eigenvalues_unordered << std:: endl;
	for (size_t i = 0; i < len / 2; i++) {
		evl[2 * i] = eigenvalues_unordered[eiV[len - 1 - i].idx];
		evt.col(2 * i) = eigenvectors_unordered.col(eiV[len - 1 - i].idx);
		evl[2 * i + 1] = eigenvalues_unordered[eiV[i].idx];
		evt.col(2 * i + 1) = eigenvectors_unordered.col(eiV[i].idx);
		//std::cout << "eigenvalues_unordered: " << eigenvalues_unordered << std::endl;
		//std::cout << "eigenvalues: " << evl << std::endl;
	}
	//if (logfile) {
	//	logfile << "evl: " << std::endl;
	//	logfile << evl << std::endl << std::endl;
	//	logfile << "evt: " << std::endl;
	//	logfile << evt << std::endl << std::endl;
	//}
	//std::cout << "eigenvalues_ordered: " << evl << std::endl;

	eigenvalues.push_back(evl);				// For computing the inner structure of the sample
	eigenvectors.push_back(evt);
}

// Do eigen decomposition for Phi. 
// Sort the eigenvectors and eigenvalues by pairs. 
// Build matrices Gd and Gc.
void EigenDecompositionD() {
	//std::vector<Eigen::VectorXcd> eigenvalues_unordered;
	//std::vector<Eigen::MatrixXcd> eigenvectors_unordered;
	bool EIGEN = false;
	bool MKL_lapack = true;

	for (size_t i = 0; i < num_pixels[0]; i++) {
		if (EIGEN) {
			std::chrono::time_point<std::chrono::system_clock> s = std::chrono::system_clock::now();
			Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(Volume->_Phi[0]);
			Volume->_Phi.erase(Volume->_Phi.begin());
			Eigen_Sort(es.eigenvalues(), es.eigenvectors());
			std::chrono::time_point<std::chrono::system_clock> e = std::chrono::system_clock::now();
			elapsed_seconds = e - s;
			if (LOG)
				logfile << "				Time for EIGEN eigendecomposition (layer " << i << "):" << elapsed_seconds.count() << "s" << std::endl;

		}
		if (MKL_lapack) {
			std::chrono::time_point<std::chrono::system_clock> s = std::chrono::system_clock::now();		// set a timer
			std::complex<double>* A = new std::complex<double>[4 * MF * 4 * MF];							// allocate space for the array that will be sent to MKL
			Eigen::MatrixXcd::Map(A, Volume->_Phi[0].rows(), Volume->_Phi[0].cols()) = Volume->_Phi[0];										// copy values from D(l) to the array
			Volume->_Phi.erase(Volume->_Phi.begin());
			// RUIJIAO: you should be able to de-allocate D here (since you have it in A and don't need it later)

			std::complex<double>* evl = new std::complex<double>[4 * MF];									// allocate space for the eigenvalues
			std::complex<double>* evt = new std::complex<double>[4 * MF * 4 * MF];							// allocate space for the eigenvectors
			MKL_eigensolve(A, evl, evt, 4 * MF);															// perform the eigendecomposition
			delete[] A;																						// delete the matrix
			std::chrono::time_point<std::chrono::system_clock> e = std::chrono::system_clock::now();
			elapsed_seconds = e - s;
			if (LOG)
				logfile << "				Time for Intel MKL eigendecomposition (layer " << i << "):" << elapsed_seconds.count() << "s" << std::endl;

			//const std::vector<long unsigned> shape{ (unsigned long)4 * MF, (unsigned long)4 * MF };
			//const bool fortran_order{ false };

			// sort eigenvalues and eigenvectors based on the imaginary component of the eigenvalue
			Eigen_Sort(Eigen::Map<Eigen::VectorXcd>(evl, 4 * MF), Eigen::Map < Eigen::MatrixXcd, Eigen::ColMajor >(evt, 4 * MF, 4 * MF));

			delete[] evl;
			delete[] evt;
		}

	}
	if (LOG)
		logfile << "				Starting calculation for connection matrix Q..." << std::endl;
	Eigen::MatrixXcd Gc;					// Upward
	Eigen::MatrixXcd Gd;					// Downward
	Gd.resize(4 * MF, 4 * MF);																		// allocate space for \check{Q}
	Gc.resize(4 * MF, 4 * MF);																		// allocate space for \hat{Q}
	std::complex<double> Di;
	std::complex<double> Ci;
	Beta.resize(num_pixels[0]);
	std::complex<double> Di_exp;
	std::complex<double> Ci_exp;

	// process the property matrix for each layer
	for (size_t i = 0; i < num_pixels[0]; i++) {													// for each layer
		if (LOG)
			logfile << "				     Calculating Q for layer " << i << std::endl;

		for (size_t j = 0; j < eigenvalues[i].size(); j++) {								// for each eigenvalue
			if (num_pixels[0] == 1) {
				Ci = std::exp(std::complex<double>(0, 1) * k * eigenvalues[i](j) * (std::complex<double>)(z[ZL - 1] - z[0]));
				Di = std::exp(std::complex<double>(0, 1) * k * eigenvalues[i](j) * (std::complex<double>)(z[0] - z[ZL - 1]));
			}
			else {
				Ci_exp = std::complex<double>(0, 1) * k * eigenvalues[i](j) * ((std::complex<double>) (z[i + 1] - z[i]));
				Di_exp = std::complex<double>(0, 1) * k * eigenvalues[i](j) * ((std::complex<double>) (z[i] - z[i + 1]));
				Ci = std::exp(Ci_exp);
				Di = std::exp(Di_exp);
			}
			if (j % 2 != 0) {
				Gd.col(j) = eigenvectors[i].col(j) * Di;
				Gc.col(j) = eigenvectors[i].col(j);
				if (Di_exp.real() > 0)
					std::cout << "Debug the eigen sorter again! Di" << std::endl;
			}

			else {
				Gd.col(j) = eigenvectors[i].col(j);
				Gc.col(j) = eigenvectors[i].col(j) * Ci;
				if (Ci_exp.real() > 0)
					std::cout << "Debug the eigen sorter again! Ci" << std::endl;
			}
		}
		if (!EXTERNAL) {
			GD.push_back(Gd);
			GC.push_back(Gc);
		}
		if (i == 0) {
			Gd_static = Gd;
			Gc_static = Gc;
		}
		else {
			tmp = MKL_inverse(Gd);
			tmp_2 = MKL_multiply(Gc, tmp, 1);
			Gc = MKL_multiply(tmp_2, Gc_static, 1);
			Gc_static = Gc;
		}
	}
}

void MatTransfer() {
	f1.resize(4 * MF, 3 * MF);
	f2.resize(4 * MF, 3 * MF);
	f3.resize(4 * MF, 3 * MF);
	f1.setZero();
	f2.setZero();
	f3.setZero();

	// Focus on z=0
	Eigen::RowVectorXcd phase = (std::complex<double>(0, 1) * k * (std::complex<double>)(z[0]) * Eigen::Map<Eigen::RowVectorXcd>(Sz[0].data(), Sz[0].size())).array().exp();
	Eigen::MatrixXcd Phase = phase.replicate(MF, 1);		// Phase is the duplicated (by row) matrix from phase.

	Eigen::MatrixXcd SZ0 = Sz[0].replicate(MF, 1);		// neg_SZ0 is the duplicated (by row) matrix from neg_Sz0.
	Eigen::MatrixXcd SZ1 = Sz[1].replicate(MF, 1);		// neg_SZ0 is the duplicated (by row) matrix from neg_Sz0.
	Eigen::MatrixXcd SX = Sx.replicate(MF, 1);		// neg_SX is the duplicated (by row) matrix from phase.
	Eigen::MatrixXcd SY = Sy.replicate(MF, 1);		// neg_SX is the duplicated (by row) matrix from phase.
	Eigen::MatrixXcd identity = Eigen::MatrixXcd::Identity(MF, MF);

	// first constraint (Equation 8)
	f1.block(0, 0, MF, MF) = identity.array() * Phase.array();
	f1.block(MF, MF, MF, MF) = identity.array() * Phase.array();
	f1.block(2 * MF, MF, MF, MF) = (std::complex<double>(-1, 0)) * identity.array() * Phase.array() * SZ0.array();
	f1.block(2 * MF, 2 * MF, MF, MF) = identity.array() * Phase.array() * SY.array();
	f1.block(3 * MF, 0, MF, MF) = identity.array() * Phase.array() * SZ0.array();
	f1.block(3 * MF, 2 * MF, MF, MF) = (std::complex<double>(-1, 0)) * identity.array() * Phase.array() * SX.array();
	phase.resize(0);
	Phase.resize(0, 0);

	// second constraint (Equation 9)
	f2.block(0, 0, MF, MF) = identity.array();
	f2.block(MF, MF, MF, MF) = identity.array();
	f2.block(2 * MF, MF, MF, MF) = SZ0.array() * identity.array();
	f2.block(2 * MF, 2 * MF, MF, MF) = SY.array() * identity.array();
	f2.block(3 * MF, 0, MF, MF) = std::complex<double>(-1, 0) * SZ0.array() * identity.array();
	f2.block(3 * MF, 2 * MF, MF, MF) = std::complex<double>(-1, 0) * SX.array() * identity.array();
	SZ0.resize(0, 0);

	// third constraint (Equation 10)	
	f3.block(0, 0, MF, MF) = -identity.array();
	f3.block(MF, MF, MF, MF) = -identity.array();
	f3.block(2 * MF, MF, MF, MF) = SZ1.array() * identity.array();
	f3.block(2 * MF, 2 * MF, MF, MF) = -SY.array() * identity.array();
	f3.block(3 * MF, 0, MF, MF) = -SZ1.array() * identity.array();
	f3.block(3 * MF, 2 * MF, MF, MF) = SX.array() * identity.array();
	SZ1.resize(0, 0);
	SX.resize(0, 0);
	SY.resize(0, 0);
	identity.resize(0, 0);
	//if (logfile) {
	//	logfile << "f1: " << std::endl;
	//	logfile << f1 << std::endl;
	//	logfile << "f2: " << std::endl;
	//	logfile << f2 << std::endl;
	//	logfile << "f3: " << std::endl;
	//	logfile << f3 << std::endl;
	//}
}

//// Set the equations that force the divergence of the electric field to be zero (Gauss' equation)
void SetGaussianConstraints() {

	// set reflected constraints
	for (size_t m = 0; m < MF; m++) {
		Mat(ei, 0, Reflected, X, m, MF) = Sx(m);
		Mat(ei, 0, Reflected, Y, m, MF) = Sy(m);
		Mat(ei, 0, Reflected, Z, m, MF) = -Sz[0](m);
		ei += 1;
	}
	// set transmitted constraints
	for (size_t m = 0; m < MF; m++) {
		Mat(ei, 1, Transmitted, X, m, MF) = Sx(m);
		Mat(ei, 1, Transmitted, Y, m, MF) = Sy(m);
		Mat(ei, 1, Transmitted, Z, m, MF) = Sz[1](m);
		ei += 1;
	}
}

// Force the field within each layer to be equal at the layer boundary
void SetBoundaryConditions() {
	std::complex<double> i(0.0, 1.0);
	if (LOG)
		logfile << "		Starting eigendecomposition of D (all layers)..." << std::endl;
	std::chrono::time_point<std::chrono::system_clock> eigen1 = std::chrono::system_clock::now();
	EigenDecompositionD();		// Compute GD and GC
	std::chrono::time_point<std::chrono::system_clock> eigen2 = std::chrono::system_clock::now();
	elapsed_seconds = eigen2 - eigen1;
	if (LOG)
		logfile << "			Time for EigenDecompositionD() (all layers): " << elapsed_seconds.count() << "s" << std::endl;

	MatTransfer();				// Achieve the connection between the variable vector and the field vector
	std::chrono::time_point<std::chrono::system_clock> matTransfer = std::chrono::system_clock::now();
	elapsed_seconds = matTransfer - eigen2;
	if (LOG)
		logfile << "			Time to transform R to P: " << elapsed_seconds.count() << "s" << std::endl;

	//const std::vector<long unsigned> shape{ (unsigned long)4 * MF, (unsigned long)4 * MF };
	//const bool fortran_order{ false };
	//Eigen::MatrixXcd A_block2 = MKL_multiply(tmp, f3, 1);
	//npy::SaveArrayAsNumpy("Gc_static.npy", fortran_order, shape.size(), shape.data(), &Gc_static(0, 0));

	Eigen::MatrixXcd Gc_inv = MKL_inverse(Gc_static);
	Gc_static.resize(0, 0);
	std::chrono::time_point<std::chrono::system_clock> inv = std::chrono::system_clock::now();
	elapsed_seconds = inv - matTransfer;
	if (LOG)
		logfile << "			Time to calculate one inversion: " << elapsed_seconds.count() << "s" << std::endl;

	A.block(2 * MF, 0, 4 * MF, 3 * MF) = f2;
	tmp = MKL_multiply(Gd_static, Gc_inv, 1);
	Gc_inv.resize(0, 0);
	A.block(2 * MF, 3 * MF, 4 * MF, 3 * MF) = MKL_multiply(tmp, f3, 1);
	f3.resize(0, 0);
	std::chrono::time_point<std::chrono::system_clock> mul = std::chrono::system_clock::now();
	elapsed_seconds = mul - inv;
	if (LOG)
		logfile << "			Time to calculate one multiplication: " << elapsed_seconds.count() / 2 << "s" << std::endl;

	b.segment(2 * MF, 4 * MF) = std::complex<double>(-1, 0) * f1 * Eigen::Map<Eigen::VectorXcd>(EF.data(), 3 * MF);

	if (logfile) {
		// RUIJIAO: save to NPY files using logprefix_????.npy
		/*logfile << "LHS matrix in the linear system:" << std::endl;
		logfile << A << std::endl << std::endl;
		logfile << "RHS vector in the linear system:" << std::endl;
		logfile << b << std::endl << std::endl;
		*/
	}
}

// Converts a b vector to a list of corresponding plane waves
std::vector<tira::planewave<double>> mat2waves(tira::planewave<double> i, Eigen::VectorXcd x, size_t p) {
	std::vector<tira::planewave<double>> P;

	P.push_back(i);											// push the incident plane wave into the P array

	tira::planewave<double> r(Sx(p) * k,
		Sy(p) * k,
		-Sz[0](p) * k * in_n[0],
		x[idx(0, Reflected, X, p, MF)],
		x[idx(0, Reflected, Y, p, MF)],
		x[idx(0, Reflected, Z, p, MF)]
	);
	tira::planewave<double> t(Sx(p) * k,
		Sy(p) * k,
		Sz[1](p) * k * in_n[1],
		x[idx(1, Transmitted, X, p, MF)],
		x[idx(1, Transmitted, Y, p, MF)],
		x[idx(1, Transmitted, Z, p, MF)]
	);
	//std::cout << "r: " << std::endl << r.str() << std::endl;
	//std::cout << "t: " << std::endl << t.str()<< std:: endl;
	P.push_back(r);
	P.push_back(t);
	return P;
}

/// Removes waves in the input set that have a k-vector pointed along the negative z axis
std::vector< tira::planewave<double> > RemoveInvalidWaves(std::vector<tira::planewave<double>> W) {
	std::vector<tira::planewave<double>> new_W;
	for (size_t i = 0; i < W.size(); i++) {
		if (W[i].getKreal()[2] > 0)
			new_W.push_back(W[i]);
	}

	return new_W;
}

int main(int argc, char** argv) {
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

	// Set up all of the input options provided to the user
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("sample", boost::program_options::value<std::string>(&in_sample), "input sample as an .npy file")
		("lambda", boost::program_options::value<double>(&in_lambda)->default_value(1.0), "incident field vacuum wavelength")
		("direction", boost::program_options::value<std::vector<double> >(&in_dir)->multitoken()->default_value(std::vector<double>{0, 0, 1}, "0, 0, 1"), "incoming field direction")
		("ex", boost::program_options::value<std::vector<double> >(&in_ex)->multitoken()->default_value(std::vector<double>{0, 0}, "0, 0"), "x component of the electrical field")
		("ey", boost::program_options::value<std::vector<double> >(&in_ey)->multitoken()->default_value(std::vector<double>{1, 0}, "1, 0"), "y component of the electrical field")
		("ez", boost::program_options::value<std::vector<double> >(&in_ez)->multitoken()->default_value(std::vector<double>{0, 0}, "0 0"), "z component of the electrical field")
		("n", boost::program_options::value<std::vector<double>>(&in_n)->multitoken()->default_value(std::vector<double>{1.0, 1.0}, "1, 1"), "real refractive index (optical path length) of the upper and lower layers")
		("kappa", boost::program_options::value<std::vector<double> >(&in_kappa)->multitoken()->default_value(std::vector<double>{0}, "0.00"), "absorbance of the lower layer (upper layer is always 0.0)")
		// The center of the sample along x/y is always 0/0.
		("size", boost::program_options::value<std::vector<double>>(&in_size)->multitoken()->default_value(std::vector<double>{10, 10, 2}, "10, 10, 4"), "The real size of the single-layer sample")
		("z", boost::program_options::value<double >(&in_z)->multitoken()->default_value(0, "0.0"), "the center of the sample along z-axis")
		("output", boost::program_options::value<std::string>(&in_outfile)->default_value("c.cw"), "output filename for the coupled wave structure")
		("coef", boost::program_options::value<std::vector<int> >(&in_coeff)->multitoken(), "number of Fourier coefficients (can be specified in 2 dimensions)")
		("psf", "generate the point spread function(PSF) for the optic system")
		("external", "save waves for visualization of external field only")
		("log", "produce a log file")
		("npz", "calculate Px and Py only since Pz is computationally intensive and almost zero when incident wave is y-polarized.")
		;
	// I have to do some strange stuff in here to allow negative values in the command line. I just wouldn't change any of it if possible.
	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).style(
		boost::program_options::command_line_style::unix_style ^ boost::program_options::command_line_style::allow_short
	).run(), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {									// output all of the command line options
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("log"))									// if a log is requested, begin output
		LOG = true;

	if (vm.count("npz"))									// if a log is requested, begin output
		Pz = false;

	if (LOG) {									// if a log is requested, begin output
		std::stringstream ss;
		ss << std::time(0) << "_scattervolume.log";
		logfile.open(ss.str());
	}

	if (LOG) {
		for (int c = 0; c < argc; c++)
			logfile << argv[c] << " ";
		logfile << std::endl;
	}

	if (LOG)
		logfile << "Initialization starts..." << std::endl;


	if (vm.count("psf")) {
		psf = true;
	}

	if (vm.count("external")) {
		EXTERNAL = true;
	}

	// Calculate the number of layers based on input parameters (take the maximum of all layer-specific command-line options)
	L = in_n.size();
	Eigen::Vector3d dir(in_dir[0], in_dir[1], in_dir[2]);
	dir.normalize();																							// set the normalized direction of the incoming source field

	glm::tvec3<std::complex<double>> e = glm::tvec3<std::complex<double>>(std::complex<double>(in_ex[0], in_ex[1]),
		std::complex<double>(in_ey[0], in_ey[1]),
		std::complex<double>(in_ez[0], in_ez[1]));				// set the input electrical field
	glm::tvec3<double> dirvec(dir(0), dir(1), dir(2));
	orthogonalize(e, dirvec);
	dir = dir * in_n[0];

	// wavenumber
	k = (std::complex<double>)(2 * PI / in_lambda);

	// Define sample volume, reformat, and reorgnize.
	InitLayer_n();
	Volume = new volume< std::complex<double> >(in_sample, ni, in_center, in_size, k.real(), std::complex<double>(in_n_sample, in_kappa_sample));
	ni.resize(0);
	num_pixels = Volume->reformat();
	ZL = num_pixels[0] + 1;

	// store all of the layer positions and refractive indices
	InitLayer_z();

	// Get the number of the Fourier Coefficients
	if (in_coeff.size() == 0) {
		M[0] = num_pixels[2];
		M[1] = num_pixels[1];
	}
	else if (in_coeff.size() == 1) {
		M[0] = std::sqrt(in_coeff[0]);
		M[1] = std::sqrt(in_coeff[0]);
	}
	else if (in_coeff.size() == 2) {
		M[0] = in_coeff[0];
		M[1] = in_coeff[1];
	}

	// Give warning if the decomposed wave goes opposite.
	if (pow((double(M[0] / 2) / in_size[0] * in_lambda + dir[0]), 2) + (pow((double(M[1] / 2) / in_size[1] * in_lambda + dir[1]), 2)) >= pow(in_n[0], 2)) {
		std::cout << "Cutting off invalid waves with imagnery sz..." << std::endl;
		std::cout << "[WARNING] " << "Propagation directions for decomposed waves are not all downward. We suggest to increase in_size or decrease the wavelength to tolerate higher Fourier coefficients. Constraints: (float(M[0]/2)/size[2])^2 + (float(M[1]/2)/size[1])^2 < (n/lambda)^2" << std::endl;
		if (M[0] != 1 && M[1] != 1) {
			if ((double(M[0] / 2) / in_size[0] * in_lambda + dir[0]) > (std::complex<double>(0.71, 0) * in_n[0]).real()) {
				M[0] = int(0.707 * in_n[0] * 2 * in_size[0] / in_lambda) - 1;
				std::cout << "M[0] is corrected as " << M[0] << std::endl;
			}
			if ((double(M[1] / 2) / in_size[1] * in_lambda + dir[1]) > (std::complex<double>(0.71, 0) * in_n[0]).real()) {
				M[1] = int(0.707 * in_n[1] * 2 * in_size[1] / in_lambda) - 1;
				std::cout << "M[1] is corrected as " << M[1] << std::endl;
			}
		}
		if (M[0] == 1) {
			if ((double(M[1] / 2) / in_size[1] * in_lambda + abs(dir[1])) >= in_n[0]) {
				M[1] = int(in_n[1] * 2 * in_size[1] / in_lambda - pow(10, -10));
				if (M[1] % 2 == 0)
					M[1] -= 1;
				std::cout << "For the 1-d volume, M[1] is corrected as " << M[1] << std::endl;
			}
		}
		if (M[1] == 1) {
			if ((double(M[0] / 2) / in_size[1] * in_lambda + abs(dir[0])) >= in_n[0]) {
				M[0] = int(in_n[0] * 2 * in_size[0] / in_lambda - pow(10, -10));
				if (M[0] % 2 == 0)
					M[0] -= 1;
				std::cout << "For the 1-d volume, M[0] is corrected as " << M[0] << std::endl;
			}
		}
	}
	if (M[0] > num_pixels[2]) {
		M[0] = num_pixels[2];
		std::cout << "M[0] is corrected as " << num_pixels[2] << ", since input is larger than its pixel numbers" << std::endl;
	}
	if (M[1] > num_pixels[1]) {
		std::cout << "M[1] is corrected as " << num_pixels[1] << ", since input is larger than its pixel numbers" << std::endl;
		M[1] = num_pixels[1];
	}
	MF = M[0] * M[1];

	std::chrono::time_point<std::chrono::system_clock> D_before = std::chrono::system_clock::now();
	Volume->CalculateD(M, dir);	// Calculate the property matrix for the sample
	std::chrono::time_point<std::chrono::system_clock> D_after = std::chrono::system_clock::now();
	elapsed_seconds = D_after - D_before;
	if (LOG)
		logfile << "		Time to create property matrix D (" << 4 * MF << "x" << 4 * MF << "): " << elapsed_seconds.count() << " s" << std::endl;

	// Fourier transform for the incident waves
	E0.push_back(e[0]);
	E0.push_back(e[1]);
	E0.push_back(e[2]);

	std::vector<Eigen::MatrixXcd> Ef(3);

	Ef[0] = fftw_fft2(E0[0] * Eigen::MatrixXcd::Ones(num_pixels[1], num_pixels[2]), M[1], M[0]);	// M[0]=3 is column. M[1]=1 is row. 
	Ef[1] = fftw_fft2(E0[1] * Eigen::MatrixXcd::Ones(num_pixels[1], num_pixels[2]), M[1], M[0]);
	Ef[2] = fftw_fft2(E0[2] * Eigen::MatrixXcd::Ones(num_pixels[1], num_pixels[2]), M[1], M[0]);
	if (psf == true) {
		Ef[0] = E0[0] * Eigen::MatrixXcd::Ones(M[0], M[1]);	// M[0]=3 is column. M[1]=1 is row. 
		Ef[1] = E0[1] * Eigen::MatrixXcd::Ones(M[0], M[1]);
		Ef[2] = E0[2] * Eigen::MatrixXcd::Ones(M[0], M[1]);
	}

	EF.resize(3 * MF);
	EF.segment(0, MF) = Eigen::Map<Eigen::VectorXcd>(Ef[0].data(), MF);
	EF.segment(MF, MF) = Eigen::Map<Eigen::VectorXcd>(Ef[1].data(), MF);
	EF.segment(2 * MF, MF) = Eigen::Map<Eigen::VectorXcd>(Ef[2].data(), MF);

	// Sync the Fourier transform of direction propagation with Volume
	Sx = Eigen::Map<Eigen::RowVectorXcd>(Volume->_meshS0.data(), MF);
	Sy = Eigen::Map<Eigen::RowVectorXcd>(Volume->_meshS1.data(), MF);
	Sz[0] = Eigen::Map<Eigen::RowVectorXcd>(Volume->_Sz[0].data(), MF);
	Sz[1] = Eigen::Map<Eigen::RowVectorXcd>(Volume->_Sz[1].data(), MF);


	if (LOG)
		logfile << "Input processing and FFT finished." << std::endl;
	std::chrono::time_point<std::chrono::system_clock> initialized = std::chrono::system_clock::now();
	elapsed_seconds = initialized - start;
	if (LOG)
		logfile << "Time to process input and perform FFT: " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	if (LOG)
		logfile << "Start creating linear system..." << std::endl;
	// Build linear system
	InitMatrices();
	std::chrono::time_point<std::chrono::system_clock> initDone = std::chrono::system_clock::now();
	elapsed_seconds = initDone - initialized;
	if (LOG)
		logfile << "		Time to allocate memory: " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	SetGaussianConstraints();
	std::chrono::time_point<std::chrono::system_clock> gauss = std::chrono::system_clock::now();
	elapsed_seconds = gauss - initDone;
	if (LOG)
		logfile << "		Time to calculate Gaussian constraints: " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	SetBoundaryConditions();
	std::chrono::time_point<std::chrono::system_clock> boundary = std::chrono::system_clock::now();
	elapsed_seconds = boundary - gauss;
	if (LOG)
		logfile << "		Time to set boundary conditions: " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	if (LOG)
		logfile << "Linear system complete." << std::endl;
	std::chrono::time_point<std::chrono::system_clock> built = std::chrono::system_clock::now();
	elapsed_seconds = built - initialized;
	if (LOG)
		logfile << "Time to complete linear system: " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	// MKL solution
	if (LOG)
		logfile << "Solving linear system..." << std::endl;

	MKL_linearsolve(A, b);
	Eigen::VectorXcd x = b;
	//std::cout << "x: " << x << std::endl;
	if (LOG)
		logfile << "Linear system solved." << std::endl;
	A.resize(0, 0);
	b.resize(0);
	std::chrono::time_point<std::chrono::system_clock> solved = std::chrono::system_clock::now();
	elapsed_seconds = solved - built;
	if (LOG)
		logfile << "Time to solve linear system: " << elapsed_seconds.count() << "s" << std::endl << std::endl;

	std::vector<Eigen::MatrixXcd> Q_check(ZL - 1);		// Q_check and Q_hat are for the inside sample only.
	std::vector<Eigen::MatrixXcd> Q_hat(ZL - 1);
	if (!EXTERNAL) {
		if (LOG)
			logfile << "Start calculating Beta for internal field..." << std::endl;
		// The data structure that all data goes to
		size_t MF4 = MF * 4;								// MF4 is the length of beta/gamma/gg

		// Solve for beta
		std::chrono::time_point<std::chrono::system_clock> beta_before = std::chrono::system_clock::now();
		Eigen::MatrixXcd EF_mat;
		Eigen::MatrixXcd Pr_0;
		Eigen::MatrixXcd beta;
		EF_mat = Eigen::Map< Eigen::MatrixXcd>(EF.data(), 3 * MF, 1);
		Pr_0 = Eigen::Map< Eigen::MatrixXcd>(x.data(), 3 * MF, 1);
		for (size_t i = 0; i < num_pixels[0]; i++) {
			if (i == 0) {
				tmp = MKL_inverse(GD[i]);
				tmp_2 = MKL_multiply(tmp, f1, 1);
				beta = MKL_multiply(tmp_2, EF_mat, 1);
				tmp_2 = MKL_multiply(tmp, f2, 1);
				beta += MKL_multiply(tmp_2, Pr_0, 1);
				EF_mat.resize(0, 0);
				Pr_0.resize(0, 0);
			}
			else {
				tmp = MKL_inverse(GD[i]);
				tmp_2 = MKL_multiply(tmp, GC[i - 1], 1);
				tmp = MKL_multiply(tmp_2, beta, 1);
				beta = tmp;
			}
			Beta[i] = beta;
		}
		beta.resize(0, 0);
		f1.resize(0, 0);
		f2.resize(0, 0);
		tmp.resize(0, 0);
		tmp_2.resize(0, 0);
		std::vector<Eigen::MatrixXcd>().swap(GD);
		std::vector<Eigen::MatrixXcd>().swap(GC);
		std::chrono::time_point<std::chrono::system_clock> beta_end = std::chrono::system_clock::now();
		elapsed_seconds = beta_end - beta_before;
		if (LOG)
			logfile << "Beta solved:" << elapsed_seconds.count() << "s" << std::endl;
		if (LOG)
			logfile << "Time to solve Beta:" << elapsed_seconds.count() << "s" << std::endl << std::endl;

		// ------------------------------------Calculating Px and Py----------------------------------------
		if (LOG)
			logfile << "Start to calculate internal field Px and Py..." << std::endl;
		std::chrono::time_point<std::chrono::system_clock> internal_before = std::chrono::system_clock::now();
		Eigen::MatrixXcd Q1, Q2, I1, I2, J1, J2;
		Q1.resize(4 * MF, 4 * MF);
		Q2.resize(4 * MF, 4 * MF);
		I1.resize(MF, 4 * MF);
		I2.resize(MF, 4 * MF);
		J1.resize(MF, 4 * MF);
		J2.resize(MF, 4 * MF);
		Eigen::VectorXd p_series;
		Eigen::VectorXd q_series;
		p_series.setLinSpaced(M[0], -double(M[0] / 2), double((M[0] - 1) / 2));                         // M=3: p_series=[-1, 0, 1]. M=2: p_series=[-1, 0]
		q_series.setLinSpaced(M[1], -double(M[1] / 2), double((M[1] - 1) / 2));
		Eigen::VectorXd WQ = 2.0 * q_series * M_PI / in_size[1];
		Eigen::VectorXd UP = 2.0 * p_series * M_PI / in_size[0];

		// Calculate beta according to the GD, GC, and Pt/Pr
		Eigen::MatrixXcd Q_even_cols;
		Eigen::MatrixXcd Q_odd_cols;
		Eigen::MatrixXcd beta_even;
		Eigen::MatrixXcd beta_odd;
		Eigen::MatrixXcd beta_even_t;
		Eigen::MatrixXcd beta_odd_t;
		for (int i = 0; i < ZL - 1; i++) {
			Eigen::MatrixXcd beta = Beta[i].asDiagonal();
			//std::cout << "beta: " << beta << std::endl;
			std::chrono::time_point<std::chrono::system_clock> mid_2 = std::chrono::system_clock::now();
			Q_even_cols = Eigen::MatrixXcd::Map(eigenvectors[i].data(), 8 * MF, 2 * MF).topRows(4 * MF);
			Q_odd_cols = Eigen::MatrixXcd::Map(eigenvectors[i].data(), 8 * MF, 2 * MF).bottomRows(4 * MF);
			beta_even = Eigen::MatrixXcd::Map(beta.data(), 8 * MF, 2 * MF).topRows(4 * MF);
			beta_odd = Eigen::MatrixXcd::Map(beta.data(), 8 * MF, 2 * MF).bottomRows(4 * MF);

			beta_even_t = beta_even.transpose();
			beta_odd_t = beta_odd.transpose();

			beta_even = Eigen::MatrixXcd::Map(beta_even_t.data(), 4 * MF, 2 * MF).topRows(2 * MF);
			beta_odd = Eigen::MatrixXcd::Map(beta_odd_t.data(), 4 * MF, 2 * MF).bottomRows(2 * MF);

			Q1 = MKL_multiply(Q_even_cols, beta_even, 1);		// Q1: 4M*1
			Q2 = MKL_multiply(Q_odd_cols, beta_odd, 1);			// Q2: 4M*1
			Q_even_cols.resize(0, 0);
			Q_odd_cols.resize(0, 0);
			beta_even_t.resize(0, 0);
			beta_odd_t.resize(0, 0);
			beta_even.resize(0, 0);
			beta_odd.resize(0, 0);

			Q_check[i].resize(3 * MF, 2 * MF);
			Q_hat[i].resize(3 * MF, 2 * MF);
			Q_check[i].setZero();
			Q_hat[i].setZero();
			Q_check[i].block(0, 0, MF, 2 * MF) = Q1.block(0, 0, MF, 2 * MF);
			Q_hat[i].block(0, 0, MF, 2 * MF) = Q2.block(0, 0, MF, 2 * MF);
			Q_check[i].block(MF, 0, MF, 2 * MF) = Q1.block(MF, 0, MF, 2 * MF);
			Q_hat[i].block(MF, 0, MF, 2 * MF) = Q2.block(MF, 0, MF, 2 * MF);

			if (Pz) {
				if (LOG)
					logfile << "		Start to calculate Pz from magnetic field..." << std::endl;
				std::chrono::time_point<std::chrono::system_clock> Pz_start = std::chrono::system_clock::now();
				I1 = Q1.block(2 * MF, 0, MF, 2 * MF); // even; downward Hx
				I2 = Q2.block(2 * MF, 0, MF, 2 * MF); // odd; upward Hx
				J1 = Q1.block(3 * MF, 0, MF, 2 * MF); // even; downward Hy
				J2 = Q2.block(3 * MF, 0, MF, 2 * MF); // odd; upward Hy
				for (int qi = 0; qi < M[1]; qi++) {
					for (int pi = 0; pi < M[0]; pi++) {
						for (int qj = 0; qj < M[1]; qj++) {
							int indR = int(qi - qj + M[1]) % M[1];
							std::complex<double> wq = std::complex<double>(WQ[qj]) + dir[1] * k;
							for (int pj = 0; pj < M[0]; pj++) {
								int indC = int(pi - pj + M[0]) % M[0];
								std::complex<double> up = std::complex<double>(UP[pj]) + dir[0] * k;
								Eigen::VectorXcd ef2 = Volume->NIf[i]((indR + M[1] / 2) % M[1] * M[0] + (indC + M[0] / 2) % M[0])
									* (up * J1.row(qj * M[0] + pj) - wq * I1.row(qj * M[0] + pj));
								Q_check[i].row(2 * MF + qi * M[0] + pi) += std::complex<double>(-1, 0) / k * ef2;
								ef2 = Volume->NIf[i]((indR + M[1] / 2) % M[1] * M[0] + (indC + M[0] / 2) % M[0])
									* (up * J2.row(qj * M[0] + pj) - wq * I2.row(qj * M[0] + pj));
								Q_hat[i].row(2 * MF + qi * M[0] + pi) += std::complex<double>(-1, 0) / k * ef2;
							}
						}
					}
				}
				if (LOG)
					logfile << "		Pz calculated." << std::endl;
				std::chrono::time_point<std::chrono::system_clock> Pz_end = std::chrono::system_clock::now();
				elapsed_seconds = Pz_end - Pz_start;
				if (LOG)
					logfile << "		Time to calculate a single Pz:" << elapsed_seconds.count() << "s" << std::endl;
			}
		}
		Q1.resize(0, 0);
		Q2.resize(0, 0);
		I1.resize(0, 0);
		I2.resize(0, 0);
		J1.resize(0, 0);
		J2.resize(0, 0);
		if (LOG)
			logfile << "Internal field calculated." << std::endl;
		std::chrono::time_point<std::chrono::system_clock> internal_end = std::chrono::system_clock::now();
		elapsed_seconds = internal_end - internal_before;
		if (LOG)
			logfile << "Time to solve the inside Field on boundaries:" << elapsed_seconds.count() << "s" << std::endl << std::endl;
	}

	if (LOG)
		logfile << "Start to build class CoupledWaveStructure:" << elapsed_seconds.count() << "s" << std::endl;
	std::chrono::time_point<std::chrono::system_clock> cw_before = std::chrono::system_clock::now();
	CoupledWaveStructure<double> cw;
	cw.Layers.resize(ZL);							// ZL is the number of boundaries. Pr for the upper boundary only; Pt for the lower boundary only. 
	tira::planewave<double> zeros(0, 0, k, 0, 0, 0);
	for (int kk = 0; kk < M[1]; kk++) {
		for (int j = 0; j < M[0]; j++) {
			std::complex<double> sy = (kk - M[1] / 2) / in_size[1] * in_lambda + dir[1];
			std::complex<double> sx = (j - M[0] / 2) / in_size[0] * in_lambda + dir[0];
			counts += 1;
			int p = kk * M[0] + j;
			tira::planewave<double> i(Sx(p) * k, Sy(p) * k, Sz[0](p) * k, EF(p), EF(MF + p), EF(2 * MF + p));
			cw.Pi.push_back(i);
			std::vector<tira::planewave<double>> P = mat2waves(i, x, p);							// P[0]=Pi; P[1]=Pr; P[2]=Pt.
			// generate plane waves from the solution vector
			tira::planewave<double> r, t;
			for (size_t l = 0; l < ZL; l++) {														// for each layer
				if (l == 0) {
					cw.Layers[l].z = z[l];
					r = P[1].wind(0.0, 0.0, -z[l]);
					//r = P[1 + l * 2 + 0];
					cw.Layers[l].Pr.push_back(r);
					if (!EXTERNAL) {
						for (int jj = 0; jj < 2 * MF; jj++) {
							tira::planewave<double> t(Sx(p) * k, Sy(p) * k, eigenvalues[l](2 * jj) * k,
								Q_check[l](p, jj), Q_check[l](MF + p, jj), Q_check[l](2 * MF + p, jj)
							);
							cw.Layers[l].Pt.push_back(t.wind(0.0, 0.0, -z[l]));
						}
					}


				}
				else if (l == ZL - 1) {
					cw.Layers[l].z = z[l];
					//t = P[1 + (l - 1) * 2 + 1];
					t = P[2].wind(0.0, 0.0, -z[l]);
					cw.Layers[l].Pt.push_back(t);
					if (!EXTERNAL) {
						for (int jj = 0; jj < 2 * MF; jj++) {
							tira::planewave<double> r(Sx(p) * k, Sy(p) * k, eigenvalues[l - 1](2 * jj + 1) * k,
								Q_hat[l - 1](p, jj), Q_hat[l - 1](MF + p, jj), Q_hat[l - 1](2 * MF + p, jj)
							);
							cw.Layers[l].Pr.push_back(r.wind(0.0, 0.0, -z[l]));
						}
					}
				}
				else {
					if (!EXTERNAL) {
						for (int jj = 0; jj < 2 * MF; jj++) {
							tira::planewave<double> r(Sx(p) * k, Sy(p) * k, eigenvalues[l - 1](2 * jj + 1) * k,
								Q_hat[l - 1](p, jj), Q_hat[l - 1](MF + p, jj), Q_hat[l - 1](2 * MF + p, jj)
							);
							cw.Layers[l].Pr.push_back(r.wind(0.0, 0.0, -z[l]));
						}

						for (int jj = 0; jj < 2 * MF; jj++) {
							tira::planewave<double> t(Sx(p) * k, Sy(p) * k, eigenvalues[l](2 * jj) * k,
								Q_check[l](p, jj), Q_check[l](MF + p, jj), Q_check[l](2 * MF + p, jj)
							);
							cw.Layers[l].Pt.push_back(t.wind(0.0, 0.0, -z[l]));
						}
						cw.Layers[l].z = z[l];
					}
				}

			}
		}
	}
	if (LOG)
		logfile << "CoupledWaveStructure built." << elapsed_seconds.count() << "s" << std::endl;
	std::chrono::time_point<std::chrono::system_clock> cw_end = std::chrono::system_clock::now();
	elapsed_seconds = cw_end - cw_before;
	if (LOG)
		logfile << "Time to build CoupledWaveStructure:" << elapsed_seconds.count() << "s" << std::endl << std::endl;

	if (counts != M[0] * M[1]) {
		std::cout << "[WARNING] Not all decomposed Fourier waves are valid. We suggest to increase in_size or decrease the wavelength to tolerate higher Fourier coefficients. Constraints: (float(M[0]/2)/size[2])^2 + (float(M[1]/2)/size[1])^2 < (n/lambda)^2. Scatterviewsample is auto disabled. Please use Scatterview instead." << std::endl;
	}
	std::cout << "Field saved in " << in_outfile << "." << std::endl;
	std::cout << "Number of pixels (x, y): [" << num_pixels[2] << "," << num_pixels[1] << "]" << std::endl;
	std::cout << "Number of sublayers: " << num_pixels[0] << std::endl;
	std::cout << "Number of Fourier coefficients (Mx, My): [" << M[0] << "," << M[1] << "]" << std::endl;
	std::cout << "Effective number of Fourier coefficients: [" << counts << "]" << std::endl;

	if (LOG)
		logfile << "Start to save field... " << std::endl;
	std::chrono::time_point<std::chrono::system_clock> save_before = std::chrono::system_clock::now();
	if (in_outfile != "") {
		cw.save(in_outfile);
	}
	if (LOG)
		logfile << "field saved. " << std::endl;
	std::chrono::time_point<std::chrono::system_clock> save_end = std::chrono::system_clock::now();
	elapsed_seconds = save_end - save_before;
	if (LOG)
		logfile << "Time to save the field " << elapsed_seconds.count() << "s" << std::endl << std::endl;
	elapsed_seconds = save_end - start;
	if (LOG)
		logfile << "Total time:" << elapsed_seconds.count() << "s" << std::endl;
}