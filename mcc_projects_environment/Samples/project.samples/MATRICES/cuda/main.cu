#include "mx.cuh"
#include <chrono>

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
		mx<int> A(DIMS, 5);
		mx<int>* a = &A;
		// a -> random_int(10);
		// cudaDeviceSynchronize();
		A.print();
		cudaDeviceSynchronize();
	}
}
