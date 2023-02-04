#include "mx.hpp"
#include <cstring>

int main(int argc, char **argv) {
	// std::cout << "Hello world!" << std::endl;

	// mx<int> A(2);
	// A.init(5);
	// A.print();

	// mx<double> B(6, 3.14);
	// B.print();

	// mx<int> C(3);
	// C.identity();
	// C.print();
	// std::cout << std::endl;

	// std::cout << "To jest macierz C:" << C << "B:" << B << "A:" << A <<
	// std::endl;

	mx<double> A(3, 3);
	mx<double> B(3, 4);
	mx<double> C(2, 3);
	mx<double> I(3);
	I.identity();
	
	// std::cout << A << B << std::endl;
	// A = B;
	// std::cout << A << B << std::endl;
	// if(A == B)
	// 	std::cout << "A is equal to B" << std::endl;
	// else
	// 	std::cout << "A is not equal to B" << std::endl;

	// std::cout << A << B << std::endl;
	// A.add(B);
	// std::cout << A << B << std::endl;
	// A = A + B;
	// std::cout << A << B << std::endl;
	// A += I;
	// std::cout << A + I << std::endl;

	// std::cout << A << B << std::endl;
	// A.subtract(B);
	// std::cout << A << B << std::endl;
	// A = A - B;
	// std::cout << A << B << std::endl;
	// std::cout << A - I << std::endl;

	// std::cout << B << std::endl;
	// B.enter_val(3, 2, 2.1);
	// B.enter_val(1, 3, 8.2);
	// B += I;
	// std::cout << B << std::endl;
	// B.transpoze();
	// std::cout << B << std::endl;

	B.enter_val(2, 3, 0);
	B.enter_val(1, 3, 1);
	B.enter_val(2, 1, -5);	
	B.enter_val(3, 2, 0);	
	B.enter_val(2, -1, 0);
	C.enter_val(1, 2, 0);
	C.enter_val(2, 2, 10);
	// A.enter_val(2, 2, 0);
	// A.enter_val(2, 3, -6);
	// std::cout << A << B << std::endl;
	// A.multiply_matrix(B);
	// std::cout << A << std::endl;
	// A *= B;
	// std::cout << A << std::endl;
	// std::cout << A * B << std::endl;

	// mx<double> D(1, 3.14);
	// mx<double> E(4, 5);
	// mx<double> Z;
	// std::cout << B.det() << " " << C.det() << " " << E.det() << " " << Z.det() << std::endl;

	// mx<double> R(3);
	// R.random_int(10, 5);
	// std::cout << R << std::endl;
	// R.random();
	// std::cout << R << std::endl;

	// mx<int> M(E, 2, 3);
	// std::cout << E << M << std::endl;
	// std::cout << E << E.det() << std::endl;

	if(argc > 1){
		mx<double> E(std::stoi(argv[1]));
		E.random_int(10, -5);
		// std::cout << E << E.det() << std::endl;
		// E.invert();
		// std::cout << E << std::endl;
		std::cout << E << E.det() << E.inverse() << std::endl;
	}
	else{
		mx<double> E(5);
		E.random_int(10);
		// std::cout << E << E.det() << std::endl;
		// std::cout << E << E.det() << std::endl;
		// E.invert();
		// std::cout << E << std::endl;
		std::cout << E << E.det() << E.inverse() << std::endl;
	}

	return 0;
}