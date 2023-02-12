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

	set_NOB(number_of_blocks);
	set_TPB(threads_per_block);

	dim3 NOB = number_of_blocks;
	dim3 TPB = threads_per_block;

	// std::ofstream fout("out.txt");

	gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaFree(0));

	for(int i = 1; i <= ITERATIONS; i++){
		mx<double> A(NOB, TPB, DIMS);
		A.h_set_val(2, 4, 1);
		A.h_set_val(1, 3, -3);
		A.h_set_val(4, 1, 4);
		A.h_set_val(3, 2, 6);
		mx<double> B(NOB, TPB, DIMS);
		B.random_int(10, -5);
		std::cout << A << B;
		// std::size_t r = 0, n_max = 0;
		// std::size_t p_r = &r, p_n_max = &n_max;
		// B.find_best_row(NOB, TPB, p_r, p_n_max);
		// std::cout << r << " " << n_max << std::endl;

		A.devFree();
		B.devFree();
	}

	// fout.close();
	return 0;
}