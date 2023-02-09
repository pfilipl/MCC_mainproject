#include "mx.cuh"
#include <chrono>

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

__global__ void cudaMain(std::ostream& out){
	mx<int> A(10);
	A.random_int(10);
	A.print(out);
}

int main(int argc, char **argv){
  
	int N, THREADS, ITERATIONS, DIMS;
	N = argc > 1 ? atoi(argv[1]) : 64;
	THREADS = argc > 2 ? atoi(argv[2]) : 16;
	ITERATIONS = argc > 3 ? atoi(argv[3]) : 1;
	DIMS = argc > 4 ? atoi(argv[4]) : 5;
		
	printf("N=%d, THREADS=%d, ITERATIONS=%d, DIMS=%d\n", N, THREADS, ITERATIONS, DIMS);

	dim3 threads_per_block(THREADS, THREADS, 1);
  	dim3 number_of_blocks((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

	for(int i = 1; i <= ITERATIONS; i++){
		cudaMain<<< number_of_blocks, threads_per_block >>>(std::cout);
	}
}
