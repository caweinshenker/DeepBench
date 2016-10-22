#pragma once

#include <vector>
#include <numeric>
#include <memory>

#include <curand.h>
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "fp16_helper.h"

template <typename  T>
class Tensor {
    std::vector<int> dims_;
    int size_;

    struct deleteCudaPtr {
        void operator()(T *p) const {
            cudaFree(p);
        }
    };

    std::shared_ptr<T> ptr_;

public:

    Tensor() {}

    Tensor(std::vector<int> dims) : dims_(dims) {
        T* tmp_ptr;
        size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
        cudaMalloc(&tmp_ptr, sizeof(T) * size_);
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
     cudaFree(float_tensor.begin());
     return half_tensor;
}  

template <>
Tensor<float> Tensor<float>::zeros(std::vector<int> dims) {
    Tensor<float> tensor(dims);
    //thrust::fill(thrust::device_ptr<float>(tensor.begin()),
    //             thrust::device_ptr<float>(tensor.end()), 0.f);
    cudaMemset(tensor.begin(), 0, tensor.size() * sizeof(float));
    return tensor;
}

template <>
Tensor<half> Tensor<half>::zeros(std::vector<int> dims){
    Tensor<half> half_tensor(dims);
    Tensor<float> float_tensor = Tensor<float>::zeros(dims);
    float_2_fp16<<<half_tensor.gridSize(dims), half_tensor.blockSize(dims)>>>(float_tensor.begin(), half_tensor.begin(), half_tensor.size()); 
    cudaFree(float_tensor.begin());
    return half_tensor;
}

template <>
Tensor<float> Tensor<float>::rand(std::vector<int> dims, curandGenerator_t curand_gen) {
    Tensor<float> tensor;
    curandGenerateUniform(curand_gen, tensor.begin(), tensor.size());
    return tensor;
}

template <>
Tensor<half> Tensor<half>::rand(std::vector<int> dims, curandGenerator_t curand_gen){ 
    Tensor <float> float_tensor = Tensor<float>::rand(dims, curand_gen);
    Tensor<half> half_tensor(dims);
    float_2_fp16<<<half_tensor.gridSize(dims), half_tensor.blockSize(dims)>>>(float_tensor.begin(), half_tensor.begin(), half_tensor.size());
    cudaFree(float_tensor.begin()); 
    return half_tensor;
} 
