#pragma once
#include "CoupledWaveStructure.h"
#include "FourierWave.h"
#include "third_Lapack.h"

template <typename T>
struct UnpackedWave {
    std::complex<T> E0[3];
    std::complex<T> k[3];
};

extern int layers;
extern std::vector<float> z_layers;
extern std::vector<int> waves_begin;
extern std::vector<int> waves_end;
extern std::vector<UnpackedWave<float>> W;

void cw_allocate(CoupledWaveStructure<double>* cw);
void cw_unpack(CoupledWaveStructure<double>* cw);

void EvaluateSample(std::vector <std::vector< Eigen::MatrixXcd>>& E, float* center, float Extent, unsigned int N);
void cpu_cw_evaluate_sample(glm::vec<3, std::complex<float>>* E_xy, glm::vec<3, std::complex<float>>* E_xz, glm::vec<3, std::complex<float>>* E_yz,
    std::vector<std::vector<Eigen::MatrixXcd>> E,
    float x_start, float y_start, float z_start, float x, float y, float z, float d);

void cpu_cw_evaluate_xy(glm::vec<3, std::complex<float>>* E_xy,
    float x_start, float y_start,
    float z, float d, size_t N);

void cpu_cw_evaluate_yz(glm::vec<3, std::complex<float>>* E_xy,
    float y_start, float z_start,
    float x, float d, size_t N);

void cpu_cw_evaluate_xz(glm::vec<3, std::complex<float>>* E_xz,
    float x_start, float z_start,
    float y, float d, size_t N);