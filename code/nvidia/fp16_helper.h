#include <curand.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <device_functions.h>

__global__
void float_2_fp16(float *d_f, half *d_h, const int size){
	
	int idx =  (blockIdx.x * blockDim.x * blockDim.y) + blockDim.y * threadIdx.y +  threadIdx.x;
        if (idx < size){
		float flt = d_f[idx];
		d_h[idx] = __float2half(flt);
	}
	return;  
}

__global__
void float_2_half(float *d_flt,  float *d_h){
	int idx = (blockIdx.x * blockDim.x * blockDim.y) + blockDim.y * threadIdx.y +  threadIdx.x;
	auto half =  __float2half(d_flt[idx]);
	d_h[idx] = __half2float(half);
}

__global__
void half_2_float(const half h,  float *d_float){
	d_float[0] = __half2float(h);
}
