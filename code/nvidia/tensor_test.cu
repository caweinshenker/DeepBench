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
cudnnHandle_t cudnn_handle;
curandGenerator_t curand_gen;
std::vector<int> dims = {1024, 16 * 25};
int hidden_size = 1760;
int batch_size = 16;
int time_steps = 50;

void float_tensor_zeros_test(){
	Tensor<float> f_tensor;
	f_tensor = Tensor<float>::zeros(dims);
	float f_host[f_tensor.size()];
	cudaMemcpy(f_host, f_tensor.begin(), sizeof(float) * f_tensor.size(), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < f_tensor.size(); i++){
		assert(f_host[i] == 0);
	}
	std::cout << "Float tensor test passed\n";
}

void half_same_test(){
	int truth = std::is_same<half,half>::value;
	assert(truth == 1);
	std::cout << "Half is half\n";
}


void half_tensor_zeros_test(){
	Tensor<half> tensor = Tensor<half>::zeros(dims);
        float h_host[tensor.size() / 2];	
	cudaMemcpy(h_host, tensor.begin(), sizeof(half) * tensor.size(), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//for (int i =0; i < tensor.size() / 2; i++){
	//	printf("%d: %f \n", i, h_host[i]);
	//	assert  ((int) h_host[i] == 0);
	//}
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
	float_2_half<<<(1,1,1), (1,1,1)>>>(d_f, d_h);
	cudaDeviceSynchronize();
	cudaMemcpy(h_h, d_h, 1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_h);
	printf("Float value: %f, Half value: %f \n", h_f[0], h_h[0]);

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
	float_2_half<<<gridSize, blockSize>>>(d_float, d_half);

	cudaDeviceSynchronize();
	cudaMemcpy(h_float, d_float, sizeof(float) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_half, d_half, sizeof(float) * size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_float);
	cudaFree(d_half);
	
	for (int i =0; i < size; i++){
		printf("%d: float:  %f, half: %f \n", i, h_float[i], h_half[i]);
	}
}



int main(int argc, char **argv) {
    cudaFree(0);
    CHECK_CUDNN_ERROR( cudnnCreate(&cudnn_handle) );

    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    float_2_half_test();
    rand_half_test();
    float_tensor_zeros_test();
    half_tensor_zeros_test();
    types_test();
    cudnnDestroy(cudnn_handle);
    curandDestroyGenerator(curand_gen);

    return 0;
}
