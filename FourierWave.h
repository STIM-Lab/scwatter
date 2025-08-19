#pragma once

#include "tira/optics/planewave.h"
#include "CoupledWaveStructure.h"
#include <vector>
#include <fstream>
#include <complex>
#include "fftw3.h"
#include "glm/glm.hpp"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include <unsupported/Eigen/MatrixFunctions>
#include "tira/field.h"
#include "fftw3.h"

#define PI 3.141592653
extern std::vector<double> in_size;
extern std::ofstream logfile;
extern bool LOG;
extern std::chrono::duration<double> elapsed_seconds;

inline glm::tvec3<std::complex<double>> cross(glm::tvec3<std::complex<double>> E, glm::tvec3 < std::complex<double>> d) {
	glm::tvec3<std::complex<double>> out(3);
	out[0] = E[1] * d[2] - E[2] * d[1];
	out[1] = E[2] * d[0] - E[0] * d[2];
	out[2] = E[0] * d[1] - E[1] * d[0];
	return out;
}

inline void orthogonalize(glm::tvec3<std::complex<double>>& E, glm::tvec3 < double>& d) {
	std::complex<double> mag = std::sqrt(pow(E[0], 2) + pow(E[1], 2) + pow(E[2], 2));
	E = E / mag;
	glm::tvec3<std::complex<double>> d_complex = glm::tvec3<std::complex<double>>(std::complex<double>(d[0], 0),
		std::complex<double>(d[1], 0), std::complex<double>(d[2], 0) );
	glm::tvec3<std::complex<double>> s = cross(E, d_complex);
	E = cross(d_complex, s);
}
// Similar function with numpy.arange.
// Special adjustment on Oct.16: Different from np.arange, the step in this function need to match with all the steps in spuEvaluate.cpp.
inline Eigen::VectorXd arange(unsigned int points, double start, double end) {
	Eigen::VectorXd Res(points);
	double step = (end - start) / (points-1);
	for (int i = 0; i < points; i++) {
		Res[i] = start + i * step;
	}
	return Res;
}

/// <summary>
/// Function: Same as the numpy.meshgrid()
/// https://blog.csdn.net/weixin_41661099/article/details/105011027
/// </summary>
/// <param name="vecX"></param>
/// <param name="vecY"></param>
/// <param name="meshX"></param>
/// <param name="meshY"></param>
inline void meshgrid(Eigen::VectorXcd& vecX, Eigen::VectorXcd& vecY, Eigen::MatrixXcd& meshX, Eigen::MatrixXcd& meshY) {
	int vecXLength = vecX.size();
	int vecYLength = vecY.size();
	for (size_t i = 0; i < vecYLength; i++) {
		meshX.row(i) = vecX;
	}
	for (size_t i = 0; i < vecXLength; i++) {
		meshY.col(i) = vecY.transpose();
	}
}

inline void meshgrid_d(Eigen::VectorXd& vecX, Eigen::VectorXd& vecY, Eigen::MatrixXd& meshX, Eigen::MatrixXd& meshY) {
	int vecXLength = vecX.size();
	int vecYLength = vecY.size();
	for (size_t i = 0; i < vecYLength; i++) {
		meshX.row(i) = vecX;
	}
	for (size_t i = 0; i < vecXLength; i++) {
		meshY.col(i) = vecY.transpose();
	}
}

// function to compute 2d-fftshift (like MATLAB) of a matrix
inline Eigen::MatrixXcd fftShift2d(Eigen::MatrixXcd mat)
{
	int m, n, p, q;
	m = mat.rows();
	n = mat.cols();

	// matrix to store fftshift data
	Eigen::MatrixXcd mat_fftshift(m, n);

	// odd # of rows and cols
	if ((int)fmod(m, 2) == 1)
	{
		p = (int)floor(m / 2.0);
		q = (int)floor(n / 2.0);
	}
	else // even # of rows and cols
	{
		p = (int)ceil(m / 2.0);
		q = (int)ceil(n / 2.0);
	}

	// vectors to store swap indices
	Eigen::RowVectorXi indx(m), indy(n);

	// compute swap indices
	if ((int)fmod(m, 2) == 1) // # of rows odd
	{
		for (int i = 0; i < m - p - 1; i++)
			indx(i) = (m - p) + i;
		for (int i = m - p - 1; i < m; i++)
			indx(i) = i - (m - p - 1);
	}
	else // # of rows even
	{
		for (int i = 0; i < m - p; i++)
			indx(i) = p + i;
		for (int i = m - p; i < m; i++)
			indx(i) = i - (m - p);
	}

	if ((int)fmod(n, 2) == 1) // # of cols odd
	{
		for (int i = 0; i < n - q - 1; i++)
			indy(i) = (n - q) + i;
		for (int i = n - q - 1; i < n; i++)
			indy(i) = i - (n - q - 1);
	}
	else // # of cols even
	{
		for (int i = 0; i < n - q; i++)
			indy(i) = q + i;
		for (int i = n - q; i < n; i++)
			indy(i) = i - (n - q);
	}

	// rearrange the matrix elements by swapping the elements
	// according to the indices computed above.
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			mat_fftshift(i, j) = mat(indx(i), indy(j));

	// return fftshift matrix
	return mat_fftshift;
}

/// </summary>
/// <param name="A">input image specified as Eiegn::MatrixXcd format</param>
/// <param name="M1">The kept number of Fourier coefficients along x</param>
/// <param name="M2">The kept number of Fourier coefficients along y</param>
/// <returns></returns>
inline Eigen::MatrixXcd fftw_fft2(Eigen::MatrixXcd A, int M1, int M2) {
	int N1 = A.rows();
	int N2 = A.cols();

	// Matrix to store Fourier domain data
	Eigen::MatrixXcd B(N1, N2);

	// For Fourier transform
	fftw_complex* in;
	fftw_complex* out;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N1 * N2);

	// convert a 2d-matrix into a 1d array
	for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
		{
			in[i * N2 + j][0] = A(i, j).real();
			in[i * N2 + j][1] = A(i, j).imag();
		}

	// allocate 1d array to store fft data
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N1 * N2);

	fftw_plan plan = fftw_plan_dft_2d(N1, N2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);

	// covert 1d fft data to a 2d-matrix
	for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
		{
			B(i, j) = std::complex<double>(out[i * N2 + j][0], out[i * N2 + j][1]);
		}

	// Scale the output
	B = B / N1 / N2;

	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);

	Eigen::MatrixXcd outnew;
	outnew = fftShift2d(B);
	return outnew.block(N1 / 2 - M1 / 2, N2 / 2 - M2 / 2, M1, M2);
}

/// </summary>
/// <param name="A">input image specified as Eiegn::MatrixXcd format</param>
/// <param name="M1">The kept number of Fourier coefficients along x</param>
/// <param name="M2">The kept number of Fourier coefficients along y</param>
/// <returns></returns>
inline Eigen::MatrixXcd fftw_ift2(Eigen::MatrixXcd A, double* X, double* Y, unsigned int points, std::complex<double>* s, std::complex<double> k) {
	Eigen::MatrixXcd E;

	int MF_x = A.cols();
	int MF_y = A.rows();

	Eigen::MatrixXd XX;
	Eigen::MatrixXd YY;

	XX.resize(points, points);
	YY.resize(points, points);
	E.resize(points, points);
	E.setZero();
	Eigen::VectorXd X_vec;
	Eigen::VectorXd Y_vec;
	X_vec = arange(points, X[0], X[1]);
	Y_vec = arange(points, Y[0], Y[1]);
	meshgrid_d(X_vec, Y_vec, XX, YY);
	double u, w;
	for (int q = 0; q < MF_y; q++) {
		w = 2 * PI * (q - MF_y / 2) / (Y[1] - Y[0]) + (s[1] * k).real();
		for (int p = 0; p < MF_x; p++) {
			u = 2 * PI * (p - MF_x / 2) / (X[1] - X[0]) + (s[0] * k).real();
			E = E + A(q, p) * (std::complex<double>(0, 1) * (std::complex<double>(u) * XX.cast<std::complex<double>>().array() + std::complex<double>(w) * YY.cast<std::complex<double>>().array())).exp().matrix();
		}
	}
	return E;
}

template <class T>
class volume : public tira::field<T> {
public:
	std::vector<Eigen::MatrixXcd> NIf;
	Eigen::MatrixXcd Nif;
	std::vector<Eigen::MatrixXcd> _Sample;
	std::vector<size_t> _shape;

	Eigen::VectorXcd _n_layers;
	std::vector<double> _center;
	std::vector<unsigned int> _num_pixels;
	std::vector<double> _size;
	std::complex<double> _n_volume;

	std::vector<int> _flag;
	//std::vector<int> _fz;
	Eigen::VectorXd _Z;
	std::vector<Eigen::MatrixXcd> _Phi;
	int* _M = new int[2];
	double _k;

	Eigen::VectorXd _p_series;
	Eigen::VectorXd _q_series;
	Eigen::VectorXd _up;
	Eigen::VectorXd _wq;
	Eigen::VectorXcd _Sx;			// Fourier coefficients of x component of direction
	Eigen::VectorXcd _Sy;			// Fourier coefficients of y component of direction
	std::vector<Eigen::MatrixXcd> _Sz;		// 2D vector for the Fourier coefficients of z component for the upper and lower boundaries
	Eigen::MatrixXcd _meshS0, _meshS1;

	Eigen::VectorXd _dir;

	volume(std::string filename,
		Eigen::VectorXcd n_layers,
		std::vector<double> center,
		std::vector<double> size,
		double k,
		std::complex<double> n_volume) {
		tira::field<T>::template load_npy<std::complex<double>>(filename);

		// Necessary parameters
		_shape = tira::field<T>::_shape;
		_n_layers = n_layers;
		_center = center;
		_size = size;
		_k = k;
		_n_volume = n_volume;
	}

	/// <summary>
	/// Read data from .npy file as std::vector<double> and reformat it to be std::vector<Eigen::MatrixXcd>
	/// </summary>
	std::vector<size_t> reformat() {
		if (_shape.size() == 3) {
			_Sample.resize(_shape[0]);
			for (size_t i = 0; i < _shape[0]; i++) {
				_Sample[i].resize(_shape[1], _shape[2]); // Orders for resize(): (row, col)
				_Sample[i] = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(&tira::field<T>::_data[i * _shape[1] * _shape[2]], _shape[1], _shape[2]);
			}
		}
		else if (_shape.size() == 2) {
			_Sample.resize(1);
			_shape.insert(_shape.begin(), 1);
			for (size_t i = 0; i < _shape[0]; i++) {
				_Sample[i].resize(_shape[1], _shape[2]); // Orders for resize(): (row, col)
				_Sample[i] = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(&tira::field<T>::_data[i * _shape[1] * _shape[2]], _shape[1], _shape[2]);

			}
		}
		return  _shape;
	}

	void CalculateD(int* M, Eigen::VectorXd dir) {
		_M = M;
		_dir = dir;
		//std::cout << "		The property matrix D starts forming..." << std::endl;
		clock_t Phi1 = clock();
		for (size_t i = 0; i < _shape[0]; i++) {
			//std::cout << "_Sample[i]: " << _Sample[i] << std::endl;
			_Phi.push_back(phi(_Sample[i]));
			if (LOG)
				logfile << "        Finished pushing D[i] the property matrix array (layer " << i << ")." << std::endl;
		}
		clock_t Phi2 = clock();
	}

private:

	void UpWq_Cal() {
		_p_series.setLinSpaced(_M[0], -double(_M[0] / 2), double((_M[0] - 1) / 2));
		_q_series.setLinSpaced(_M[1], -double(_M[1] / 2), double((_M[1] - 1) / 2));
		_up = 2 * PI * _p_series / in_size[0] + _dir[0] * _k * Eigen::VectorXd::Ones(_M[0]);
		_wq = 2 * PI * _q_series / in_size[1] + _dir[1] * _k * Eigen::VectorXd::Ones(_M[1]);
		_Sx = (_up / _k).template cast<std::complex<double>>();
		_Sy = (_wq / _k).template cast<std::complex<double>>();

		_meshS0.setZero(_M[1], _M[0]);
		_meshS1.setZero(_M[1], _M[0]);
		// The z components for propagation direction. _Sz[0] is for the upper region while _Sz[1] is for the lower region
		meshgrid(_Sx, _Sy, _meshS0, _meshS1);
		_Sz.resize(2);
		_Sz[0] = (pow(_n_layers(0), 2) * Eigen::MatrixXcd::Ones(_M[1], _M[0]).array() - _meshS0.array().pow(2) - _meshS1.array().pow(2)).cwiseSqrt();
		_Sz[1] = (pow(_n_layers(1), 2) * Eigen::MatrixXcd::Ones(_M[1], _M[0]).array() - _meshS0.array().pow(2) - _meshS1.array().pow(2)).cwiseSqrt();
	}

	Eigen::MatrixXcd phi(Eigen::MatrixXcd sample) {
		int idx = 0;
		UpWq_Cal();

		std::chrono::time_point<std::chrono::system_clock> fft_before = std::chrono::system_clock::now();
		Eigen::MatrixXcd Nf = fftw_fft2(sample.array().pow(2), _M[1], _M[0]);
		//std::cout << "Nf: " << Nf << std::endl;
		Nif = fftw_fft2(sample.cwiseInverse().array().pow(2), _M[1], _M[0]);
		std::chrono::time_point<std::chrono::system_clock> fft_after = std::chrono::system_clock::now();
		elapsed_seconds = fft_after - fft_before;
		if (LOG)
			logfile<< "		Time to perform FFT: " << elapsed_seconds.count() << " s. " << std::endl;

		NIf.push_back(Nif);
		int MF = _M[0] * _M[1];

		// Calculate phi
		Eigen::MatrixXcd phi;
		phi.setZero(4 * MF, 4 * MF);

		std::vector<Eigen::VectorXcd> A(3, Eigen::VectorXcd(MF));
		double k_inv = 1.0 / pow(_k, 2);
		size_t indR, indC;
		for (size_t qi = 0; qi < _M[1]; qi++) {
			std::complex<double> wq = _wq[qi];
			for (size_t pi = 0; pi < _M[0]; pi++) {
				std::complex<double> up = _up[pi];
				size_t li = 0;
				for (size_t qj = 0; qj < _M[1]; qj++) {
					std::complex<double> wqj = _wq[qj];
					indR = ((int)_q_series[qi] % _M[1] - (int)_q_series[qj] % _M[1] + _M[1]) % _M[1];
					for (size_t pj = 0; pj < _M[0]; pj++) {
						std::complex<double> upj = _up[pj];
						indC = ((int)_p_series[pi] % _M[0] - (int)_p_series[pj] % _M[0] + _M[0]) % _M[0];			// % has different meanings in C++ and Python
						A[0](li) = Nf((indR + (_M[1] / 2)) % _M[1], (indC + (_M[0] / 2)) % _M[0]);
						A[1](li) = upj * (Nif((indR + (_M[1] / 2)) % _M[1], (indC + (_M[0] / 2)) % _M[0]));
						A[2](li) = wqj * (Nif((indR + (_M[1] / 2)) % _M[1], (indC + (_M[0] / 2)) % _M[0]));
						li += 1;
					}
				}

				// Dense storage
				phi.row(qi * _M[0] + pi).segment(2 * MF, MF) = up * k_inv * A[2];
				phi.row(qi * _M[0] + pi).segment(3 * MF, MF) = -up * k_inv * A[1];
				phi(qi * _M[0] + pi, 3 * MF + qi * _M[0] + pi) += 1;

				phi.row(qi * _M[0] + pi + MF).segment(2 * MF, MF) = wq * k_inv * A[2];
				phi.row(qi * _M[0] + pi + MF).segment(3 * MF, MF) = -wq * k_inv * A[1];
				phi(qi * _M[0] + pi + MF, 2 * MF + qi * _M[0] + pi) += -1;

				phi.row(qi * _M[0] + pi + 2 * MF).segment(MF, MF) = -A[0];
				phi(qi * _M[0] + pi + 2 * MF, qi * _M[0] + pi) += -up * wq * k_inv;
				phi(qi * _M[0] + pi + 2 * MF, MF + qi * _M[0] + pi) += up * up * k_inv;

				phi.row(qi * _M[0] + pi + 3 * MF).segment(0, MF) = A[0];
				phi(qi * _M[0] + pi + 3 * MF, qi * _M[0] + pi) += -wq * wq * k_inv;
				phi(qi * _M[0] + pi + 3 * MF, MF + qi * _M[0] + pi) += up * wq * k_inv;
			}
		}
		return phi;
		//phi.resize(0, 0);
	}
};
