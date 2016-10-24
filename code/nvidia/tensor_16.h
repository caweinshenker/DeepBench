#pragma once

#include <vector>
#include <numeric>
#include <memory>

#include <curand.h>
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include "fp16_helper.h"
#include "cuda_helper.h"

template <typename  T>
class Tensor {
    std::vector<int> dims_;
    int size_;

    struct deleteCudaPtr {
        void operator()(T *p) const {
            CHECK_CUDA_ERROR(cudaFree(p));
        }
    };

    std::shared_ptr<T> ptr_;

public:

    Tensor() {}

    Tensor(std::vector<int> dims) : dims_(dims) {
        T* tmp_ptr;
        size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
        CHECK_CUDA_ERROR(cudaMalloc(&tmp_ptr, sizeof(T) * size_));
        ptr_.reset(tmp_ptr, deleteCudaPtr());
    }

    static Tensor<T> fill(std::vector<int> dims, float  val);
    static Tensor<T> zeros(std::vector<int> dims);
    static Tensor<T> rand(std::vector<int> dims, curandGenerator_t curand_gen); 
    T* begin() const { return ptr_.get(); }
    T* end()   const { return ptr_.get() + size_; }
    int size() const { return size_; }
    dim3 gridSize(std::vector<int> dims) {dim3 d(dims[0]/32, 1, 1); return d;}
    dim3 blockSize(std::vector<int> dims) {dim3 d(dims[0] / (dims[0] / 32), dims[1], 1); return d;}
    std::vector<int> dims() const { return dims_; }
    void print();
};


template <>
Tensor<float> Tensor<float>::fill(std::vector<int> dims, float val) {
     Tensor<float> tensor(dims);
     thrust::fill(thrust::device_ptr<float>(tensor.begin()),
                  thrust::device_ptr<float>(tensor.end()), val);
     return tensor;
}

template <>
Tensor<half> Tensor<half>::fill(std::vector<int> dims, float  val){
     Tensor<float> float_tensor = Tensor<float>::fill(dims, val);
     Tensor<half> half_tensor(dims);
     float_2_fp16<<<half_tensor.gridSize(dims), half_tensor.blockSize(dims)>>>(float_tensor.begin(), half_tensor.begin(), half_tensor.size()); 
     return half_tensor;
}  

template <>
Tensor<float> Tensor<float>::zeros(std::vector<int> dims) {
    Tensor<float> tensor(dims);
    try
	{
		thrust::fill(thrust::device_ptr<float>(tensor.begin()),
                 thrust::device_ptr<float>(tensor.end()), 0.f);
	}
    catch(thrust::system_error &e)
	{
		std::cerr << "CUDA ERROR after thrust::fill " << e.what() << std::endl;
				
		//oops, recover
		cudaSetDevice(0);
		cudaMemset(tensor.begin(), 0, tensor.size() * sizeof(float));

	}
    return tensor;
}

template <>
Tensor<half> Tensor<half>::zeros(std::vector<int> dims){
    Tensor<half> half_tensor(dims);
    Tensor<float> float_tensor = Tensor<float>::zeros(dims);
    float_2_fp16<<<half_tensor.gridSize(dims), half_tensor.blockSize(dims)>>>(float_tensor.begin(), half_tensor.begin(), half_tensor.size()); 
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return half_tensor;

}

template <>
Tensor<float> Tensor<float>::rand(std::vector<int> dims, curandGenerator_t curand_gen) {
    Tensor<float> tensor(dims);
    curandGenerateUniform(curand_gen, tensor.begin(), tensor.size());
    return tensor;
}

template <>
Tensor<half> Tensor<half>::rand(std::vector<int> dims, curandGenerator_t curand_gen){ 
    Tensor <float> float_tensor = Tensor<float>::rand(dims, curand_gen);
    Tensor<half> half_tensor(dims);
    float_2_fp16<<<half_tensor.gridSize(dims), half_tensor.blockSize(dims)>>>(float_tensor.begin(), half_tensor.begin(), half_tensor.size());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return half_tensor;
} 



template <>
void Tensor<float>::print(){
    float h_out[size_];
    CHECK_CUDA_ERROR(cudaMemcpy(h_out, begin(), sizeof(float) * size_, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    for (int i = 0; i < dims_[0]; i++){
	printf("[");
	for (int j = 0; j < dims_[1] - 1; j++){
		printf("%f,", h_out[i * dims_[1] + j]);
	}
	printf("%f]\n", h_out[i * dims_[1] + dims_[1] - 1]);
    }

}


template <>
void Tensor<half>::print(){
    float h_out[size_];
    float *d_buff;

    CHECK_CUDA_ERROR(cudaMalloc(&d_buff, sizeof(float) * size_));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
   
    half_2_float<<<std::max(1, std::min(1, size_/1024)), 1024>>>(begin(), d_buff, size_);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_buff, sizeof(float) * size_, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    for (int i = 0; i < dims_[0]; i++){
	printf("%d\n", i);
	printf("[");
	for (int j = 0; j < dims_[1] - 1; j++){
		printf("%f,", h_out[i * dims_[1] + j]);
	}
	printf("%f]\n", h_out[i * dims_[1] + dims_[1] - 1]);
    }
}

