#include <stdint.h>
#include <iostream>
#include <tira/optics/planewave.h>
#include "CoupledWaveStructure.h"
#include "FourierWave.h"

#include <extern/libnpy/npy.hpp>
#include <complex>
#include <math.h>
#include <fstream>
#include <boost/program_options.hpp>
#include <random>
#include <iomanip>
#include "glm/gtc/quaternion.hpp"
#include "Eigen/Dense"

std::vector<double> in_dir;
double in_lambda;

std::vector<double> in_n;
std::vector<double> in_kappa;
std::vector<double> in_ex;
std::vector<double> in_ey;
std::vector<double> in_ez;
std::vector<double> in_z;
std::vector<double> in_normal;
double in_na;
std::string in_outfile;
std::string in_sample;
double in_alpha;
double in_beta;
std::vector<unsigned int> in_samples;
std::string in_mode;
std::vector<bool> in_wavemask;

unsigned int L;
Eigen::MatrixXcd A;
Eigen::VectorXcd b;
std::vector<std::complex<double>> ri;
std::vector< std::complex<double>> sz;
double* z;
double k;				// wavenumber in the incident layer
double k_vac;			// free space (vacuum) wavenumber

std::ofstream logfile;

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

std::string vec2str(glm::vec<3, double> v, int spacing = 20) {
	std::stringstream ss;
	ss << std::setw(spacing) << std::left << v[0] << std::setw(spacing) << std::left << v[1] << std::setw(spacing) << std::left << v[2];
	return ss.str();
}

std::complex<double>& Mat(int row, int col) {
	//return A[row * 6 * L + col];
	return A(row, col);
}

enum Coord { X, Y, Z };
enum Dir { Transmitted, Reflected };

std::complex<double>& Mat(int row, int layer, Dir d, Coord c) {
	//return A[row * 6 * L + layer * 6 + d * 3 + c];
	return A(row, layer * 6 + d * 3 + c);
}

size_t idx(int layer, Dir d, Coord c) {
	return layer * 6 + d * 3 + c;
}



/// <summary>
/// Output the coupled wave matrix as a string
/// </summary>
/// <param name="width"> Spacing between horizontal elements </param>
/// <param name="precision"> Number of decimal points to display </param>
/// <returns></returns>
std::string mat2str(int width = 10, int precision = 2) {
	std::stringstream ss;
	/*for (int row = 0; row < 6 * L; row++) {
		for (int col = 0; col < 6 * L; col++) {
			ss << std::setw(width) << std::setprecision(precision) << Mat(row, col);
		}
		ss << std::endl;
	}*/
	ss << A;
	return ss.str();
}

std::string b2str(int precision = 2) {
	std::stringstream ss;
	/*for (int row = 0; row < 6 * L; row++) {
		ss << std::setprecision(precision) << b[row] << std::endl;
	}*/
	ss << b;
	return ss.str();
}

void InitMatrices() {
	//A = new std::complex<double>[6 * L * 6 * L];														// allocate space for the matrix
	A = Eigen::MatrixXcd::Zero(6 * L, 6 * L);
	//memset(A, 0, 6 * L * 6 * L * sizeof(std::complex<double>));										// zero out the matrix
	b = Eigen::VectorXcd::Zero(6 * L);
	//memset(b, 0, 6 * L * sizeof(std::complex<double>));
}

void InitLayerProperties() {

	ri.push_back(in_n[0]);
	for (size_t l = 1; l < L; l++)
		ri.push_back(std::complex<double>(in_n[l], in_kappa[l - 1]));						// store the complex refractive index for each layer
	z = new double[L - 1];													// allocate space to store z coordinates for each interface
	for (size_t l = 0; l < L - 1; l++)
		z[l] = in_z[l];
}

/// <summary>
/// Calculate the z component of the direction vector for each layer. This is how the refractive index is stored for each layer.
/// </summary>
/// <param name="p"></param>
void InitSz(tira::planewave<double> p, double incident_refractive_index) {
	glm::vec<3, double> s = p.getDirection() * incident_refractive_index;
	sz.resize(L);										// allocate space to store the sz coordinate for each layer

	if (logfile) {
		logfile << "sx = " << s[0] << ", sy = " << s[1] << std::endl;
		logfile << "SZ---------------" << std::endl;
	}

	for (size_t l = 0; l < L; l++) {
		sz[l] = std::sqrt(ri[l] * ri[l] - s[0] * s[0] - s[1] * s[1]);
		if (logfile) logfile << "sz(" << l << ") = " << sz[l] << "   (RI = " << ri[l] << ")" << std::endl;
	}
	if (logfile) logfile << "---------------" << std::endl << std::endl;
}

/// <summary>
/// Set the boundary conditions given an E0 vector
/// </summary>
/// <param name="E"></param>
void SetBoundaryConditions(tira::planewave<double> p) {
	Mat(0, 0, Transmitted, X) = 1;
	Mat(1, 0, Transmitted, Y) = 1;
	Mat(2, 0, Transmitted, Z) = 1;

	glm::vec<3, std::complex<double> > E = p.getE0();
	b[0] = E[0];
	b[1] = E[1];
	b[2] = E[2];

	Mat(3, L - 1, Reflected, X) = 1;
	Mat(4, L - 1, Reflected, Y) = 1;
	Mat(5, L - 1, Reflected, Z) = 1;
}

void SetGaussianConstraints(tira::planewave<double> p) {
	glm::vec<3, double> s = p.getDirection();
	size_t start_row = 6;
	// set reflected constraints
	for (size_t l = 0; l < L - 1; l++) {
		Mat(start_row + l, l, Reflected, X) = s[0];
		Mat(start_row + l, l, Reflected, Y) = s[1];
		Mat(start_row + l, l, Reflected, Z) = -sz[l];
	}

	start_row += L - 2;
	// set transmitted constraints
	for (size_t l = 1; l < L; l++) {
		Mat(start_row + l, l, Transmitted, X) = s[0];
		Mat(start_row + l, l, Transmitted, Y) = s[1];
		Mat(start_row + l, l, Transmitted, Z) = sz[l];
	}
}

void SetBoundaryConstraints(tira::planewave<double> p) {
	glm::vec<3, double> s = p.getDirection();
	double zn, zp;
	size_t start_row = 6 + 2 * (L - 1);
	std::complex<double> i(0.0, 1.0);

	for (size_t l = 0; l < L - 1; l++) {

		if (l == 0) zn = z[l];
		else zn = z[l] - z[l - 1];

		if (l == L - 2) zp = 0;
		else zp = z[l] - z[l + 1];

		// first constraint (Equation 8)
		Mat(start_row + l * 4 + 0, l, Transmitted, X) = std::exp(i * k * sz[l] * zn);
		Mat(start_row + l * 4 + 0, l, Reflected, X) = 1.0;
		Mat(start_row + l * 4 + 0, l + 1, Transmitted, X) = -1.0;
		Mat(start_row + l * 4 + 0, l + 1, Reflected, X) = -std::exp(-i * k * sz[l + 1] * zp);

		// second constraint (Equation 9)
		Mat(start_row + l * 4 + 1, l, Transmitted, Y) = std::exp(i * k * sz[l] * zn);
		Mat(start_row + l * 4 + 1, l, Reflected, Y) = 1.0;
		Mat(start_row + l * 4 + 1, l + 1, Transmitted, Y) = -1.0;
		Mat(start_row + l * 4 + 1, l + 1, Reflected, Y) = -std::exp(-i * k * sz[l + 1] * zp);

		// third constraint (Equation 10)
		Mat(start_row + l * 4 + 2, l, Transmitted, Z) = s[1] * std::exp(i * k * sz[l] * zn);
		Mat(start_row + l * 4 + 2, l, Transmitted, Y) = -sz[l] * std::exp(i * k * sz[l] * zn);
		Mat(start_row + l * 4 + 2, l, Reflected, Z) = s[1];
		Mat(start_row + l * 4 + 2, l, Reflected, Y) = sz[l];
		Mat(start_row + l * 4 + 2, l + 1, Transmitted, Z) = -s[1];
		Mat(start_row + l * 4 + 2, l + 1, Transmitted, Y) = sz[l + 1];
		Mat(start_row + l * 4 + 2, l + 1, Reflected, Z) = -s[1] * std::exp(-i * k * sz[l + 1] * zp);
		Mat(start_row + l * 4 + 2, l + 1, Reflected, Y) = -sz[l + 1] * std::exp(-i * k * sz[l + 1] * zp);

		// third constraint (Equation 11)
		Mat(start_row + l * 4 + 3, l, Transmitted, X) = sz[l] * std::exp(i * k * sz[l] * zn);
		Mat(start_row + l * 4 + 3, l, Transmitted, Z) = -s[0] * std::exp(i * k * sz[l] * zn);
		Mat(start_row + l * 4 + 3, l, Reflected, X) = -sz[l];
		Mat(start_row + l * 4 + 3, l, Reflected, Z) = -s[0];
		Mat(start_row + l * 4 + 3, l + 1, Transmitted, X) = -sz[l + 1];
		Mat(start_row + l * 4 + 3, l + 1, Transmitted, Z) = s[0];
		Mat(start_row + l * 4 + 3, l + 1, Reflected, X) = sz[l + 1] * std::exp(-i * k * sz[l + 1] * zp);
		Mat(start_row + l * 4 + 3, l + 1, Reflected, Z) = s[0] * std::exp(-i * k * sz[l + 1] * zp);
	}
}

/// <summary>
/// Convert the solution matrix to a set of waves
/// </summary>
/// <param name="i"></param>
/// <param name="x"></param>
/// <returns></returns>
std::vector<tira::planewave<double>> mat2waves(tira::planewave<double> i, Eigen::VectorXcd x) {
	std::vector<tira::planewave<double>> P;

	P.push_back(i);											// push the incident plane wave into the P array
	for (size_t l = 0; l < L - 1; l++) {						// for each layer
		glm::vec<3, double> s = i.getDirection() * in_n[0];
		tira::planewave<double> r(s[0] * k_vac,
			s[1] * k_vac,
			-sz[l] * k_vac,
			x[idx(l, Reflected, X)],
			x[idx(l, Reflected, Y)],
			x[idx(l, Reflected, Z)]);

		tira::planewave<double> t(s[0] * k_vac,
			s[1] * k_vac,
			sz[l + 1] * k_vac,
			x[idx(l + 1, Transmitted, X)],
			x[idx(l + 1, Transmitted, Y)],
			x[idx(l + 1, Transmitted, Z)]);

		P.push_back(r);
		P.push_back(t);

	}

	return P;
}

/// Removes waves with a k-vector pointed along the negative z axis
std::vector< tira::planewave<double> > RemoveInvalidWaves(std::vector<tira::planewave<double>> W) {
	std::vector<tira::planewave<double>> new_W;
	for (size_t i = 0; i < W.size(); i++) {
		if (W[i].getKreal()[2] > 0)
			new_W.push_back(W[i]);
	}

	return new_W;
}



int main(int argc, char** argv) {

	// Declare the supported options.
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("lambda", boost::program_options::value<double>(&in_lambda)->default_value(5), "incident field vacuum wavelength")
		("direction", boost::program_options::value<std::vector<double> >(&in_dir)->multitoken()->default_value(std::vector<double>{1, 0, 1}, "0, 0, 1"), "incoming field direction")
		("ex", boost::program_options::value<std::vector<double> >(&in_ex)->multitoken()->default_value(std::vector<double>{1, 0}, "0, 0"), "x component of the electrical field")
		("ey", boost::program_options::value<std::vector<double> >(&in_ey)->multitoken()->default_value(std::vector<double>{1, 0}, "1, 0"), "y component of the electrical field")
		("ez", boost::program_options::value<std::vector<double> >(&in_ez)->multitoken()->default_value(std::vector<double>{0, 0}, "0 0"), "z component of the electrical field")
		("n", boost::program_options::value<std::vector<double>>(&in_n)->multitoken()->default_value(std::vector<double>{1, 1.4, 1.4, 1.0}, "1, 1.4, 1.4, 1.0"), "real refractive index (optical path length) of all L layers")
		("kappa", boost::program_options::value<std::vector<double> >(&in_kappa)->multitoken()->default_value(std::vector<double>{0.05}, "0.05, 0.00, 0.00"), "absorbance of layers 2+ (layer 1 is always 0.0)")
		("z", boost::program_options::value<std::vector<double> >(&in_z)->multitoken()->default_value(std::vector<double>{-5.0, 0.0, 5.0}, "-3.0, 0.0, 3.0"), "position of each layer boundary")
		("output_sample", boost::program_options::value<std::string>(&in_sample)->default_value("sample.npy"), "output the sample as the input for scattervolume")
		("output", boost::program_options::value<std::string>(&in_outfile)->default_value("a.cw"), "output filename for the coupled wave structure")
		("alpha", boost::program_options::value<double>(&in_alpha)->default_value(1), "angle used to focus the incident field")
		("beta", boost::program_options::value<double>(&in_beta)->default_value(0.0), "internal obscuration angle (for simulating reflective optics)")
		("na", boost::program_options::value<double>(&in_na), "focus angle expressed as a numerical aperture (overrides --alpha)")
		("samples", boost::program_options::value<std::vector<unsigned int> >(&in_samples)->multitoken()->default_value(std::vector<unsigned int>{64, 64}, "64 64"), "number of samples (can be specified in 2 dimensions)")
		("wavemask", boost::program_options::value<std::vector<bool> >(&in_wavemask)->multitoken()->default_value(std::vector<bool>{1, 1, 1}, "1 1 1"), "waves simulated (boolean value for incident, reflected, and transmitted)")
		("mode", boost::program_options::value<std::string>(&in_mode)->default_value("polar"), "sampling mode (polar, montecarlo)")
		("log", "produce a log file")
		;
	boost::program_options::variables_map vm;
	//boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc, boost::program_options::command_line_style::unix_style ^ boost::program_options::command_line_style::allow_short), vm);
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).style(
		boost::program_options::command_line_style::unix_style ^ boost::program_options::command_line_style::allow_short
	).run(), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}


	if (vm.count("log")) {									// if a log is requested, begin output
		std::stringstream ss;
		ss << std::time(0) << "_scatterlayer.log";
		logfile.open(ss.str());
	}

	// override alpha with NA if specified
	if (vm.count("na")) {
		in_alpha = asin(in_na / in_n[0]);
	}

	// calculate the number of layers based on input parameters (take the maximum of all layer-specific command-line options)
	L = in_n.size();
	if (in_kappa.size() + 1 > L)														// if more absorption coefficients are provided
		L = in_kappa.size() + 1;														// add additional layers
	if (in_z.size() + 1 > L)															// if more z coordinates are provided
		L = in_z.size() + 1;															// add additional layers


	// update parameter lists so that all represent the same number of layers
	in_n.resize(L, in_n.back());														// add additional layers (append copies of the previous refractive index)
	in_kappa.resize(L - 1, 0.0);														// add additional layers (append values of zero absorbance as necessary)
	if (in_z.size() + 1 < L) {															// if there isn't a z coordinate specified for each layer
		for (size_t l = in_z.size(); l < L - 1; l++) {
			in_z.push_back(in_z.back() + 10.0);											// add additional layers in increments of 10 units
		}
	}

	glm::tvec3<double> dir = glm::normalize(glm::tvec3<double>(in_dir[0], in_dir[1], in_dir[2]));				// set the direction of the incoming source field
	glm::tvec3<std::complex<double>> e = glm::tvec3<std::complex<double>>(std::complex<double>(in_ex[0], in_ex[1]),
		std::complex<double>(in_ey[0], in_ey[1]),
		std::complex<double>(in_ez[0], in_ez[1]));				// set the input electrical field
	orthogonalize(e, dir);
	dir = dir * in_n[0];
	k_vac = 2 * M_PI / in_lambda;
	k = k_vac;										// calculate the wavenumber (2 pi * n / lambda) in the incident plane (accounting for refractive index)
	tira::planewave<double> i_ref(dir[0] * k, dir[1] * k, dir[2] * k, e[0], e[1], e[2]);

	unsigned int N[2];											// calculate the number of samples
	if (in_samples.size() == 1) {
		if (in_mode == "montecarlo") {
			N[0] = in_samples[0];
			N[1] = 1;
		}
		else {
			N[0] = N[1] = std::sqrt(in_samples[0]);
		}
	}
	else {
		N[0] = in_samples[0];
		N[1] = in_samples[1];
	}
	if (in_alpha == 0) {
		N[0] = 1;
		N[1] = 1;
	}


	// Create an array of incident plane waves based on the sampling angle parameters
	std::vector< tira::planewave<double> > I;

	if (in_alpha == 0 || N[0] * N[1] == 1) {
		I.push_back(i_ref);
	}
	else if (in_mode == "montecarlo")
		I = tira::planewave<double>::SolidAngleMC(in_alpha, k * dir[0], k * dir[1], k * dir[2], e[0], e[1], e[2], N[0] * N[1], in_beta, glm::vec<3, double>(0, 0, -1));
	else if (in_mode == "polar")
		I = tira::planewave<double>::SolidAnglePolar(in_alpha, k * dir[0], k * dir[1], k * dir[2], e[0], e[1], e[2], N[0], N[1], in_beta, glm::vec<3, double>(0, 0, -1));


	CoupledWaveStructure<double> cw;																// allocate a coupled wave structure to store simulation results
	cw.Layers.resize(L - 1);

	for (size_t p = 0; p < I.size(); p++) {															// for each incident plane wave
		tira::planewave<double> i = I[p];															// store the incident plane wave in i
		InitMatrices();
		InitLayerProperties();
		InitSz(i, in_n[0]);																				// initialize the model matrix
		SetGaussianConstraints(i);
		SetBoundaryConditions(i);
		SetBoundaryConstraints(i);

		Eigen::VectorXcd x = A.colPivHouseholderQr().solve(b);												// solve the linear system
		std::vector<tira::planewave<double>> P = mat2waves(i, x);									// generate plane waves from the solution vector
		//std::cout << x << std::endl;
		if (in_wavemask[0])
			cw.Pi.push_back(i);

		for (size_t l = 0; l < L - 1; l++) {														// for each layer
			cw.Layers[l].z = z[l];
			tira::planewave<double> r = P[1 + l * 2 + 0].wind(0.0, 0.0, -z[l]);
			if (in_wavemask[1])
				cw.Layers[l].Pr.push_back(r);
			tira::planewave<double> t = P[1 + l * 2 + 1].wind(0.0, 0.0, -z[l]);
			if (in_wavemask[2])
				cw.Layers[l].Pt.push_back(t);

			if (logfile) {
				logfile << "LAYER " << l << "==========================" << std::endl;
				logfile << "i (" << p << ") ------------" << std::endl << i.str() << std::endl;
				logfile << "r (" << p << ") ------------" << std::endl << r.str() << std::endl;
				logfile << "t (" << p << ") ------------" << std::endl << t.str() << std::endl;
				logfile << std::endl;
			}

		}
	}

	if (in_outfile != "") {
		cw.save(in_outfile);
	}

	if (in_sample != "") {
		std::complex<double>* ref = new std::complex<double>[ri.size() - 2];
		for (int i = 0; i < ri.size() - 2; i++) {
			ref[i] = ri[i + 1];
		}
		const std::vector<long unsigned> shape{ (unsigned long)(ri.size() - 2), 1, 1 };
		const bool fortran_order{ false };
		npy::SaveArrayAsNumpy(in_sample, fortran_order, shape.size(), shape.data(), ref);
	}

	// calculate the reference wave and output results
	InitSz(i_ref, in_n[0]);
	SetBoundaryConditions(i_ref);															// set the matrix boundary conditions
	SetGaussianConstraints(i_ref);
	SetBoundaryConstraints(i_ref);
	Eigen::VectorXcd x = A.partialPivLu().solve(b);												// solve the linear system

	if (logfile) {
		logfile << "Reference Wave Matrix:" << std::endl;
		logfile << A << std::endl;
		logfile << "Reference Wave RHS:" << std::endl;
		logfile << b << std::endl;
		logfile << "Reference Wave Solution:" << std::endl;
		logfile << x << std::endl;
		logfile << std::endl;
	}
	std::vector<tira::planewave<double>> P = mat2waves(i_ref, x);

	int namewidth = 20;
	int numwidth = 10;
	int spacing3 = 20;
	// incident field parameters
	std::cout << std::setw(namewidth) << std::left << "vacuum lambda:";
	std::cout << std::setw(numwidth) << std::left << in_lambda << std::endl;

	// optics
	if (in_alpha != 0.0) {
		std::cout << std::setw(namewidth) << std::left << "focusing angle:";
		std::cout << std::setw(numwidth) << in_alpha;
		std::cout << " (" << std::sin(in_alpha) * in_n[0] << " NA)" << std::endl;
		if (in_beta > 0.0) {
			std::cout << std::setw(namewidth) << std::left << "obscuration angle:";
			std::cout << std::setw(numwidth) << std::left << in_beta;
			std::cout << " (" << std::sin(in_beta) * in_n[0] << " NA)" << std::endl;
		}
	}
	std::cout << std::endl;
	std::cout << std::setw(namewidth) << std::left << "layers: " << L << std::endl;
	for (size_t l = 0; l < L; l++) {													// for each layer
		float z_start, z_end;
		if (l == 0) z_start = -INFINITY;
		else z_start = z[l - 1];


		if (l == L - 1) z_end = INFINITY;
		else z_end = z[l];

		std::cout << std::setw(4) << std::right << l;
		std::stringstream layer_range;
		layer_range << "[" << z_start << ", " << z_end << "]:";
		std::cout << std::setw(namewidth - 4) << std::left << layer_range.str();
		std::cout << ri[l].real() << " + " << ri[l].imag() << "i" << std::endl;
	}

	std::cout << std::endl;

	std::cout << std::setw(namewidth) << std::left << "samples: " << N[0] << " x " << N[1] << " = " << N[0] * N[1] << std::endl;
	std::cout << std::setw(namewidth) << std::left << "sampling mode: " << in_mode << std::endl;
	std::cout << std::endl;

	glm::vec<3, double> i_k_real = i_ref.getKreal();
	std::cout << std::setw(namewidth) << std::left << "vvvvvvvv    k:" << vec2str(i_k_real, numwidth) << std::endl;
	std::cout << std::setw(namewidth) << std::left << "vvvvvvvv  |k|:" << glm::length(i_k_real) << std::endl;
	glm::vec<3, std::complex<double>> i_E = i_ref.getE0();
	std::cout << std::setw(namewidth) << std::left << "vvvvvvvv E(0):" << vec2str(i_E, numwidth) << std::endl << std::endl;
	std::vector< tira::planewave<double> > R;
	std::vector< tira::planewave<double> > T;
	T.push_back(i_ref);								// push the incident plane wave to the back
	for (size_t l = 0; l < L - 1; l++) {

		tira::planewave<double> r = P[1 + l * 2 + 0].wind(0.0, 0.0, -z[l]);
		R.push_back(r);
		tira::planewave<double> t = P[1 + l * 2 + 1].wind(0.0, 0.0, -z[l]);
		T.push_back(t);
		glm::vec<3, std::complex<double>> r_k = r.getK();
		std::cout << std::setw(namewidth) << std::left << "^^^^^^^^    k:" << vec2str(r_k, numwidth) << std::endl;
		//std::cout << std::setw(spacing1) << std::left << "^^^^^ |k|:" << glm::length(r_k) << std::endl;
		glm::vec<3, std::complex<double>> r_E = r.getE0();
		std::cout << std::setw(namewidth) << std::left << "^^^^^^^^ E(0):" << vec2str(r_E, numwidth) << std::endl;

		std::cout << std::endl;
		std::cout << "n = " << ri[l].real() << " + " << ri[l].imag() << "i=========================" << std::endl;
		//std::cout << "                            z = " << z[l] << std::endl;
		glm::vec<3, std::complex<double>> Er = R[l].getE(0, 0, z[l]);
		glm::vec<3, std::complex<double>> Et = T[l].getE(0, 0, z[l]);
		glm::vec<3, std::complex<double>> sum = Er + Et;
		std::cout << "-------E(" << z[l] << std::setw(numwidth) << std::left << "):" << vec2str(sum, numwidth) << std::endl;
		std::cout << "n = " << ri[l + 1].real() << " + " << ri[l + 1].imag() << "i=========================" << std::endl;
		std::cout << std::endl;

		glm::vec<3, std::complex<double>> t_k = t.getK();
		std::cout << std::setw(namewidth) << std::left << "vvvvvvvv    k:" << vec2str(t_k, numwidth) << std::endl;
		//std::cout << std::setw(spacing1) << std::left << "vvvvv |k|:" << glm::length(t_k) << std::endl;
		glm::vec<3, std::complex<double>> t_E = t.getE0();
		std::cout << std::setw(namewidth) << std::left << "vvvvvvvv E(0):" << vec2str(t_E, numwidth) << std::endl;
		std::cout << std::endl << std::endl;


	}

}