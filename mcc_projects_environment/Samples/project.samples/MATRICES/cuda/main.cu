#include "mx.cuh"
#include <chrono>
#include <vector>
// #include <fstream>

template <typename T>
__global__ void test(mx<T> A, T v){
	std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
	std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	if(row < A.len() && col < A.len())
		A.set_val(row + 1, col + 1, v);
}

int main(int argc, char **argv){
  
	int DIMS, THREADS, ITERATIONS;
	DIMS = argc > 1 ? atoi(argv[1]) : 5;
	THREADS = argc > 2 ? atoi(argv[2]) : 16;
	ITERATIONS = argc > 3 ? atoi(argv[3]) : 1;
		
	printf("DIMS=%d, THREADS=%d, ITERATIONS=%d\n", DIMS, THREADS, ITERATIONS);

	dim3 threads_per_block(THREADS, THREADS, 1);
  	dim3 number_of_blocks((DIMS / threads_per_block.x) + 1, (DIMS / threads_per_block.y) + 1, 1);

	// std::ofstream fout("out.txt");

	gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaFree(0));

	for(int i = 1; i <= ITERATIONS; i++){
		mx<int> A;
		A.devAlloc(DIMS);
		int v = 5;
		test<<< number_of_blocks, threads_per_block >>>(A, v);
		gpuErrchk(cudaPeekAtLastError());
    	gpuErrchk(cudaDeviceSynchronize());
		std::vector<int> B(DIMS * DIMS);
		gpuErrchk(cudaMemcpy(&B[0], A.val, A.len(), cudaMemcpyDeviceToHost));
		A.devFree();
		for(int i = 0; i < DIMS * DIMS; i++)
			std::cout << i << " = " << B[i] << std::endl;
	}

	// fout.close();
	return 0;
}