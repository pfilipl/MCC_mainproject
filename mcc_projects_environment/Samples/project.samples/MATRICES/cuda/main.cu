#include "mx.cuh"
#include <chrono>
#include <fstream>

// #define cudaCheckErrors(msg) \
//     do{ \
//         cudaError_t __err = cudaGetLastError(); \
//         if (__err != cudaSuccess) { \
//             fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
//                 msg, cudaGetErrorString(__err), \
//                 __FILE__, __LINE__); \
//             fprintf(stderr, "*** FAILED - ABORTING\n"); \
//             exit(1); \
//         } \
//     }while(0)

__global__ void cudaMain(std::ostream& out, int DIMS){
	mx<int> A(DIMS);
	__syncthreads();
	A.random_int(10);
	__syncthreads();
	A.print(out);
	__syncthreads();
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

	for(int i = 1; i <= ITERATIONS; i++){
		cudaMain<<< number_of_blocks, threads_per_block >>>(std::cout, DIMS);
		cudaDeviceSynchronize();
	}

	// fout.close();
}