#include <chrono>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <tuple>

#include <cuda.h>
#include <curand.h>
#include <cuda_fp16.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "tensor_16.h"
#include "cudnn_helper.h"
#include "half.hpp"
#include <assert.h>
#include <cmath>
#include <cublas_v2.h>
#define TOL 1e-3


cudnnHandle_t cudnn_handle;
curandGenerator_t curand_gen;
std::vector<int> dims = {1024, 16 * 25};
int hidden_size = 1760;
int batch_size = 16;
int time_steps = 50;



void float_tensor_fill_test(){
	Tensor<float>tensor = Tensor<float>::fill(dims, 1.5);
	float f_host[tensor.size()];
	CHECK_CUDA_ERROR(cudaMemcpy(f_host, tensor.begin(), sizeof(float) * tensor.size(), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	for (int i = 0; i < tensor.size(); i++){
		if (f_host[i] == 0) printf("%d: %f\n", i, f_host[i]);
		assert(f_host[i] != 0);
	}
	std::cout << "Fill float tensor test passed\n";

}


void float_tensor_zeros_test(){
	Tensor<float> tensor = Tensor<float>::zeros(dims);
	float f_host[tensor.size()];
	CHECK_CUDA_ERROR(cudaMemcpy(f_host, tensor.begin(), sizeof(float) * tensor.size(), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	for (int i = 0; i < tensor.size(); i++){
		if (f_host[i] != 0) printf("%d: %f\n", i, f_host[i]);
		assert(f_host[i] == 0);
	}
	std::cout << "Zeros float tensor test passed\n";
}


void float_tensor_rand_test(){
	Tensor<float> tensor = Tensor<float>::rand(dims, curand_gen);
	float f_host[tensor.size()];
	CHECK_CUDA_ERROR(cudaMemcpy(f_host, tensor.begin(), sizeof(float) * tensor.size(), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	for (int i = 0; i < tensor.size(); i++){
		if (f_host[i] == 0) printf("%d: %f\n", i, f_host[i]);
		assert(f_host[i] != 0);
	}
	std::cout << "Random float tensor test passed\n";

}


void half_same_test(){
	int truth = std::is_same<half,half>::value;
	assert(truth == 1);
	std::cout << "Half is half\n";
}


void half_tensor_rand_test(){
	Tensor<half> tensor = Tensor<half>::rand(dims, curand_gen);
	printf("Random half tensor test passed\n");	
}


void half_tensor_zeros_test(){
	std::vector<int> dims2 = {50, 10};
	Tensor<half> tensor = Tensor<half>::zeros(dims);
	float *d_f;
        float h_host[tensor.size()];
	CHECK_CUDA_ERROR(cudaMalloc(&d_f, sizeof(float) * tensor.size()));
	
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	half_2_float<<<50, 50>>>(tensor.begin(), d_f, tensor.size()); 

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());	
	CHECK_CUDA_ERROR(cudaMemcpy(h_host, d_f, sizeof(float) * tensor.size(), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	tensor.print();
	for (int i =0; i < tensor.size() / 2; i++){
		if (std::abs(h_host[i]) >= TOL) printf("%d: %f\n", i, h_host[i]);
		assert (std::abs(h_host[i]) < TOL);
	}
	std::cout << "Half tensor zeros passed\n";
} 

//Check that tensor of half types generate
void types_test(){
	TensorDescriptorNdArray<half> xDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps);
	TensorDescriptorNd<half> hx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1});
	std::cout << "Types test passed\n";
}

void float_2_half_test(){
	float h_f[1];
	h_f[0] = 0.487;
	float h_h[1];
	float *d_f;
	float *d_h;
	cudaMalloc(&d_f, sizeof(float) * 1);
	cudaMalloc(&d_h, sizeof(float) * 1);
	cudaMemcpy(d_f, h_f, sizeof(float) * 1, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize(); 
	float_2_half<<<(1,1,1), (1,1,1)>>>(d_f, d_h, 1);
	cudaDeviceSynchronize();
	cudaMemcpy(h_h, d_h, 1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_h);
	assert(std::abs(h_f[0] - h_h[0]) < 1e-3);
	printf("Float to half test passed\n");

}    

void rand_half_test(){
	const int size = 32;
	float  *d_float;
	float  *d_half;
	float  h_float[size];
	float  h_half[size];
	dim3 gridSize(1,1,1);
    	dim3 blockSize(32, 1,  1);

	cudaMalloc(&d_float, sizeof(float) * size);
	cudaMalloc(&d_half, sizeof(float) * size);
	cudaDeviceSynchronize();
        
	curandGenerateUniform(curand_gen, d_float, size);	
	cudaDeviceSynchronize();
	float_2_half<<<gridSize, blockSize>>>(d_float, d_half, size);

	cudaDeviceSynchronize();
	cudaMemcpy(h_float, d_float, sizeof(float) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_half, d_half, sizeof(float) * size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_float);
	cudaFree(d_half);
	
	for (int i =0; i < size; i++){
		assert(std::abs(h_float[i] - h_half[i]) < 1e-3);
	}
	printf("Rand half test passed\n");
}


void half_multiply_test(){
	Tensor<half> A = Tensor<half>::rand({2, 1}, curand_gen);
	Tensor<half> B = Tensor<half>::rand({1, 2}, curand_gen);
	Tensor<half> C = Tensor<half>::zeros({2, 1});	
	printf("Tensors initialized\n");
	//A.print();
	//B.print();
	//C.print();



	//Create a CUBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	//Multiplication params
	int m = dims[0];
	int n = dims[1];
	int k = dims[1];
	int lda=m, ldb=k,ldc=m;
	float h_alf[1] = {1.0f};
	float h_bet[1] = {0.0f};
	float *d_alf;
	float *d_bet;
	half  *d_h_alf;
	half  *d_h_bet;


	//Move alpha and beta  onto device
	cudaMalloc(&d_alf, sizeof(float) * 1);
	cudaMalloc(&d_bet, sizeof(float) * 1);
	cudaMalloc(&d_h_alf, sizeof(half) * 1);
	cudaMalloc(&d_h_bet, sizeof(half) * 1);
	cudaDeviceSynchronize();
	cudaMemcpy(d_alf, h_alf, sizeof(float) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bet, h_bet, sizeof(float) * 1, cudaMemcpyHostToDevice);
	float_2_fp16<<<1,1>>>(d_alf, d_h_alf, 1);
	float_2_fp16<<<1,1>>>(d_bet, d_h_bet, 1);

	
	cudaDeviceSynchronize();


	//Multiply
	cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, d_h_alf,  A.begin(), lda, B.begin(), ldb, d_h_bet, C.begin(), ldc);
	cudaDeviceSynchronize();
	printf("Half multiply test passed\n");

}


int main(int argc, char **argv) {
    cudaFree(0);
    CHECK_CUDNN_ERROR( cudnnCreate(&cudnn_handle) );

    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);
   
    float_tensor_fill_test(); 
    float_tensor_zeros_test();
    float_tensor_rand_test();
    float_2_half_test();
    rand_half_test();
    //half_tensor_fill_test();
    half_tensor_zeros_test();
    half_tensor_rand_test();
    //types_test();
    half_multiply_test();
    


    cudnnDestroy(cudnn_handle);
    curandDestroyGenerator(curand_gen);
    return 0;
}
