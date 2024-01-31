#pragma once

#include "cpuEvaluator.h"

#include <vector>
#include <thrust/complex.h>

extern int layers;
extern std::vector<float> z_layers;
extern std::vector<int> waves_begin;
extern std::vector<int> waves_end;
extern std::vector<UnpackedWave<float>> W;

void gpu_initialize();

void gpu_cw_evaluate(thrust::complex<float>* E_xy, thrust::complex<float>* E_xz, thrust::complex<float>* E_yz,
    float x_start, float y_start, float z_start, float x, float y, float z, float d,
    int N, int device);