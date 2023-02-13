#include "mx.cuh"
#include <chrono>
#include <fstream>

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

	std::ifstream fin("in.txt");
	// std::ofstream foutin("in.txt");
	std::ofstream fout("out.txt");

	gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaFree(0));

	mx<double> A(NOB, TPB, DIMS);
	mx<double> C(NOB, TPB, DIMS);
	// A.random_int(10, -5);
	// foutin << A;

	double x;
	for(int j = 0; j < DIMS * DIMS; j++){
		fin >> x;
		A.h_set_val(j, x);
	}

	auto start = std::chrono::steady_clock::now();
	for(int i = 1; i <= ITERATIONS; i++){
		mx<double> B(NOB, TPB, A.inverse(NOB, TPB));
		if(i == ITERATIONS)
			C = B;
		B.devFree();		
	}
	auto stop = std::chrono::steady_clock::now();
	std::chrono::duration<double> time = stop - start;

	fout << A << C << std::endl << time.count() / ITERATIONS << std::endl;
	A.devFree();
	C.devFree();

	fin.close();
	// foutin.close();
	fout.close();
	return 0;
}