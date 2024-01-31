#pragma once

// This file contains a set of wrapper functions that are linked to the corresponding functions in CLAPACK
#include <stdlib.h>
#include <stdio.h>
#include <complex>
#include "Eigen/Eigen"
#include "mkl.h"
#include "mkl_lapacke.h"
//
//extern "C" {
//#include "f2c.h"
//#include "clapack.h"
//}
//
//// A is the matrix to be eigen-solved. N is the number of rows or columns.
//void LINALG_eigensolve(std::complex<double>* A, std::complex<double>* eigenvalues, std::complex<double>* eigenvectors, int N) {
//	int* IPIV = new int[N + (size_t)1];
//	char jobvl = 'N';	// Not computed
//	char jobvr = 'V';	// Computed
//	integer LDA = N;	// Order of the matrix A. First dim for A.
//	doublecomplex* VL = new doublecomplex[N];	// Output for left eigenvectors
//	integer LDVL = 1;	// 1 for 'N'. N for 'V'
//	integer LDVR = N;	// 1 for 'N'. N for 'V'
//	integer LWORK = 2 * N;	//generally>= max(1,2N)
//	double* RWORK = new double[2 * N];	// Dim: 2N
//	std::complex<double>* WORK = new std::complex<double>[LWORK];
//	integer INFO;
//	// 'N' means the left eigenvectors of A are not computed
//	zgeev_(&jobvl, &jobvr, (integer*)&N, (doublecomplex*)A, &LDA, (doublecomplex*)eigenvalues, VL, &LDVL, (doublecomplex*)eigenvectors, &LDVR, (doublecomplex*)WORK, &LWORK, (doublereal*)RWORK, &INFO);
//	delete[] IPIV;
//	delete[] VL;
//	delete[] RWORK;
//	delete[] WORK;
//}

// A is the matrix to be eigen-solved. N is the number of rows or columns.
inline void MKL_eigensolve(std::complex<double>* A, std::complex<double>* eigenvalues, std::complex<double>* eigenvectors, int N) {
	int LDA = N, LDVL = N, LDVR = N;
	MKL_INT n = N, lda = LDA, ldvl = LDVL, ldvr = LDVR, info;
	MKL_Complex16* w, * vl, * vr;
	vl = new MKL_Complex16[LDVL * N];
	info = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'V', n, (MKL_Complex16*)A, lda, (MKL_Complex16*)eigenvalues, vl, ldvl, (MKL_Complex16*)eigenvectors, ldvr);
	delete[] vl;
}

inline void MKL_linearsolve(Eigen::MatrixXcd& A, Eigen::VectorXcd& b) {
	int N = b.size();
	MKL_INT* x;
	x = new MKL_INT[N * 2];
	int Nb = 1;
	// 'N' means no trans
	int info = LAPACKE_zgesv(LAPACK_COL_MAJOR, N, Nb, (MKL_Complex16*)A.data(), N, x, (MKL_Complex16*)b.data(), N);
}

inline Eigen::MatrixXcd MKL_inverse(Eigen::MatrixXcd A) {
	int N = A.rows();
	Eigen::MatrixXcd B = Eigen::MatrixXcd::Identity(N, N);
	MKL_INT* x;
	x = new MKL_INT[N * 2];
	// 'N' means no trans
	int info = LAPACKE_zgesv(LAPACK_COL_MAJOR, N, N, (MKL_Complex16*)A.data(), N, x, (MKL_Complex16*)B.data(), N);
	return B;
}
//inline Eigen::MatrixXcd MKL_inverse(Eigen::MatrixXcd A) {
//	int N = A.rows();
//	Eigen::MatrixXcd B = Eigen::MatrixXcd::Identity(N, N);
//	MKL_INT* x;
//	x = new MKL_INT[N * 2];
//	// 'N' means no trans
//	int info = LAPACKE_zgesv(LAPACK_COL_MAJOR, N, N, (MKL_Complex16*)A.data(), N, x, (MKL_Complex16*)B.data(), N);
//	return B;
//}

inline Eigen::MatrixXcd MKL_multiply(Eigen::MatrixXcd A, Eigen::MatrixXcd B, std::complex<double> alpha) {
	// This example computes real matrix C=alpha*A*B+beta*C using zgemm.
	int M = A.rows();
	int K = A.cols();
	int N = B.cols();
	std::complex<double> beta = 0;
	Eigen::MatrixXcd C = Eigen::MatrixXcd::Zero(M, N);
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, (MKL_Complex16*)A.data(), M, (MKL_Complex16*)B.data(), K, &beta, (MKL_Complex16*)C.data(), M);

	return C;
}


//Eigen::VectorXcd CuspIterativeSolve(Eigen::MatrixXcd& A, Eigen::VectorXcd& b);