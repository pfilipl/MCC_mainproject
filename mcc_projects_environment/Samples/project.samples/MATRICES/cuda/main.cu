#include "mx.cuh"
#include <chrono>
// #include <fstream>

int main(int argc, char **argv){
  
	int DIMS, THREADS, ITERATIONS;
	DIMS = argc > 1 ? atoi(argv[1]) : 5;
	THREADS = argc > 2 ? atoi(argv[2]) : 16;
	ITERATIONS = argc > 3 ? atoi(argv[3]) : 1;
		
	printf("DIMS=%d, THREADS=%d, ITERATIONS=%d\n", DIMS, THREADS, ITERATIONS);

	srand(time(NULL));

	dim3 threads_per_block(THREADS, THREADS, 1);
  	dim3 number_of_blocks((DIMS / threads_per_block.x) + 1, (DIMS / threads_per_block.y) + 1, 1);

	// std::ofstream fout("out.txt");

	gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaFree(0));

	for(int i = 1; i <= ITERATIONS; i++){
		mx<double> A(number_of_blocks, threads_per_block, DIMS, 5);
		std::cout << A;
		A.identity(number_of_blocks, threads_per_block);
		std::cout << A;
		A.random();
		std::cout << A;
		A.random_int(10);
		std::cout << A;
		mx<double> B(number_of_blocks, threads_per_block, A, 2, 3);
		std::cout << B;
		mx<double> C;
		C.copy(number_of_blocks, threads_per_block, B);
		std::cout << C;
		B.minor(number_of_blocks, threads_per_block, 2, 3);
		std::cout << B;

		A.devFree();
		B.devFree();
	}

	// fout.close();
	return 0;
}