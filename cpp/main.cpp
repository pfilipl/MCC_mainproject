#include "mx.hpp"
#include <cstring>
#include <chrono>
#include <fstream>

int main(int argc, char **argv) {
	int DIMS;
	std::string fout_name = "mul_out";
	if(argc > 1){
		DIMS = atoi(argv[1]);
		fout_name += argv[1];
	}
	else{
		DIMS = 5;
		fout_name += "5";
	}
	fout_name += ".txt";
		
	printf("DIMS=%d\n", DIMS);

	std::ifstream fin("in.txt");
	// std::ofstream foutin("in.txt");
	std::ofstream fout(fout_name);

	mx<double> A(DIMS);
	mx<double> B(DIMS);
	mx<double> C(DIMS);
	// A.random_int(10, -5);
	// foutin << A;

	double x, y;
	for(int j = 0; j < DIMS * DIMS; j++){
		fin >> x >> y;
		A.set_val(j, x);
		B.set_val(j, y);
	}

	auto start = std::chrono::steady_clock::now();
	C = A * B;
	auto stop = std::chrono::steady_clock::now();
	std::chrono::duration<double> time = stop - start;

	fout << A << B << C << std::endl << time.count() << std::endl;

	fin.close();
	// foutin.close();
	fout.close();
	return 0;
}