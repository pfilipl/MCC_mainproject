#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__ __STDC_HOSTED__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifndef MX_H_
#define MX_H_

#include <iostream>
#include <ctime>

template <typename T, int BLOCK_SIZE>
class mx{
	std::size_t dim;
	T* val;
	public:
		// constructors
		CUDA_CALLABLE_MEMBER mx(std::size_t n = 0, T v = 0){
			dim = n;
			if(dim == 0)
				val = nullptr;
			else{
				val = new T[dim * dim];
				this -> init(v);
			}
		}
		CUDA_CALLABLE_MEMBER mx(const mx<T, BLOCK_SIZE>& A, std::size_t r, std::size_t c){
			std::size_t dimA = A.get_dim();
			dim = dimA - 1;
			val = new T[dim * dim];
            std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
            if(row > r && col > c)
                this -> enter_val(row, col, A.get_val(row + 1, col + 1));
            else if(row > r)
                this -> enter_val(row, col + 1, A.get_val(row + 1, col + 1));
            else if(col > c)
                this -> enter_val(row + 1, col, A.get_val(row + 1, col + 1));
            else if(row != r && col != c)
                this -> enter_val(row + 1, col + 1, A.get_val(row + 1, col + 1));
		}

		// deconstructor
		CUDA_CALLABLE_MEMBER ~mx() { delete [] val; }

		// initializing
		CUDA_CALLABLE_MEMBER void init(T v = 0){
            std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
			val[row * dim + col] = v;
		}

		// random initializing
		CUDA_CALLABLE_MEMBER void random(){
			srand(time(NULL));
            std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
			val[row * dim + col] = rand();
		}
		CUDA_CALLABLE_MEMBER void random_int(int range, int val_min = 0){
			srand(time(NULL));
			std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
			val[row * dim + col] = rand() % range + val_min;
		}

		// making identity matrix
		CUDA_CALLABLE_MEMBER void identity(){
			this -> init();
			std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
			val[row * dim + col] = 1;
		}

		// geting the 'dim' value
		CUDA_CALLABLE_MEMBER std::size_t get_dim() const { return dim; }

		// geting 'val[]' value
		CUDA_CALLABLE_MEMBER T operator[](std::size_t n) const{
			return val[n];
		}
		CUDA_CALLABLE_MEMBER T get_val(std::size_t r, std::size_t c) const{
			return val[(r - 1) * dim + (c - 1)];
		}

		// entering 'val[]' value
		CUDA_CALLABLE_MEMBER void enter_val(std::size_t r, std::size_t c, T x){
			val[(r - 1) * dim + (c - 1)] = x;
		}

		// printing matrix to the stream
		CUDA_CALLABLE_MEMBER void print(std::ostream& out = std::cout) const{
            std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
            if((row * dim + col) % dim == 0)
                out << std::endl << val[row * dim + col] << " ";
            else
                out << val[row * dim + col] << " ";
			out << std::endl;
		}

		// copying matrix
		CUDA_CALLABLE_MEMBER void copy(const mx<T, BLOCK_SIZE>& A){
			if(this != &A){
				if(dim == A.get_dim()){
                    std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
                    std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
                    val[row * dim + col] = A[row * dim + col];
                }		
				else
					std::cout << "[E]: Invalid dimention! Copying is not possible!" << std::endl;
			}
		}

		// assigning matrix
		CUDA_CALLABLE_MEMBER mx& operator=(const mx<T, BLOCK_SIZE>& A){
			if(this == &A)
				return *this;
			if(dim != A.get_dim()){
				dim = A.get_dim();
				delete [] val;
				val = new T[dim * dim];
			}
			this -> copy(A);
			return *this;
		}

		// matrices comparison
		CUDA_CALLABLE_MEMBER bool operator==(const mx<T, BLOCK_SIZE>& A) const{
			if(this == &A)
				return true;
			if(dim != A.get_dim())
				return false;
			std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
            if(val[row * dim + col] != A[row * dim + col])
                return false;
			return true;
		}

		// adding matrix
		CUDA_CALLABLE_MEMBER void add(const mx<T, BLOCK_SIZE>& A){
			if(dim == A.get_dim()){
                std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
                std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
				val[row * dim + col] += A[row * dim + col];
            }
			else
				std::cout << "[E]: Invalid dimention! Addition is not possible!" << std::endl;
		}
		CUDA_CALLABLE_MEMBER mx operator+(const mx<T, BLOCK_SIZE>& A) const{
			mx<T, BLOCK_SIZE> result(dim);
			result.copy(*this);
			result.add(A);
			return result;
		}
		CUDA_CALLABLE_MEMBER mx& operator+=(const mx<T, BLOCK_SIZE>& A){
			this -> add(A);
			return *this;
		}

		// multiplying by scalar
		CUDA_CALLABLE_MEMBER void multiply_scalar(double x){
			std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
			val[row * dim + col] *= x;
		}
		CUDA_CALLABLE_MEMBER mx operator*(double x) const{
			mx<T, BLOCK_SIZE> result(dim);
			result.copy(*this);
			result.multiply_scalar(-1.0);
			return result;
		}
		CUDA_CALLABLE_MEMBER mx& operator*=(double x){
			this -> multiply_scalar(x);
			return *this;
		}

		// subtracting matrix
		CUDA_CALLABLE_MEMBER void subtract(const mx<T, BLOCK_SIZE>& A){
			this -> add(A*-1.0);
		}
		CUDA_CALLABLE_MEMBER mx operator-(const mx<T, BLOCK_SIZE>& A) const{
			mx<T, BLOCK_SIZE> result(dim);
			result.copy(*this);
			result.subtract(A);
			return result;
		}
		CUDA_CALLABLE_MEMBER mx& operator-=(const mx<T, BLOCK_SIZE>& A){
			this -> subtract(A);
			return *this;
		}

		// transpozition
		CUDA_CALLABLE_MEMBER void transpoze(){
			mx<T, BLOCK_SIZE> temp(dim);
			temp.copy(*this);
			this -> init();
			std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
			this -> enter_val(col + 1, row + 1, temp.get_val(row + 1, col + 1));
		}

		// multiplying by matrix
		CUDA_CALLABLE_MEMBER void multiply_matrix(const mx<T, BLOCK_SIZE>& A){
			if(dim != A.get_dim())
				std::cout << "[E]: Invalid dimention! Multiplying is not possible!" <<std::endl;
			else{
				mx<T, BLOCK_SIZE> result(dim);
				T x;
				std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            	std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
				for(int k = 1; k <= dim; k++){
					if(k == 1)
						x = 0;
						x += (this -> get_val(row + 1, k)) * A.get_val(k, col + 1);
				}
				result.enter_val(row + 1, col + 1, x);
				this -> copy(result);
			}	
		}
		CUDA_CALLABLE_MEMBER mx operator*(const mx<T, BLOCK_SIZE>& A) const{
			mx<T, BLOCK_SIZE> result(dim);
			result.copy(*this);
			result.multiply_matrix(A);
			return result;
		}
		CUDA_CALLABLE_MEMBER mx& operator*=(const mx<T, BLOCK_SIZE>& A){
			this -> multiply_matrix(A);
			return *this;
		}

		// determinant calculation helper
		CUDA_CALLABLE_MEMBER int find_best_row(std::size_t& r) const{
			int n, n_max = 0;
			std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
			n = 0;
			for(int j = 1; j <= dim; j++)
				if((this -> get_val(row + 1, j)) == 0)
					n++;
			if(n > n_max){
				n_max = n;
				r = i;
			}	
			return n_max;
		}

		CUDA_CALLABLE_MEMBER int find_best_column(std::size_t& c) const{
			mx<T, BLOCK_SIZE> temp(dim);
			temp.copy(*this);
			temp.transpoze();
			return temp.find_best_row(c);
		}

		// determinant
		CUDA_CALLABLE_MEMBER T det() const{
			T x = 0;
			switch(dim){
				case 0:
					std::cout << "[E]: Invalid dimention!" << std::endl;
					break;
				case 1:
					x = val[0];
					break;
				case 2:
					x = val[0] * val[3] - val[1] * val[2];
					break;
				case 3:
					x += val[0] * val[4] * val[8];
					x += val[3] * val[7] * val[2];
					x += val[6] * val[1] * val[5];
					x -= val[2] * val[4] * val[6];
					x -= val[5] * val[7] * val[0];
					x -= val[8] * val[1] * val[3];
					break;
				default:
					std::size_t r = 1, c = 1, n;
					int zero_count_r = this -> find_best_row(r);
					int zero_count_c = this -> find_best_column(c);
					if(zero_count_r >= zero_count_c){
						n = r;
						for(int i = 1; i <= dim; i++){
							mx<T, BLOCK_SIZE> M(*this, n, i);
							if((1 + i) % 2 == 0)
								x += (this -> get_val(n, i)) * M.det();
							else
								x -= (this -> get_val(n, i)) * M.det();
						}
					}
					else{
						n = c;
						for(int i = 1; i <= dim; i++){
							mx<T, BLOCK_SIZE> M(*this, i, n);
							if((1 + i) % 2 == 0)
								x += (this -> get_val(i, n)) * M.det();
							else
								x -= (this -> get_val(i, n)) * M.det();
						}
					}
			}
			return x;
		}

		// inverse matrix
		CUDA_CALLABLE_MEMBER void invert(){
			T det = this -> det();
			if(!det)
				std::cout << "Matrix is noninversable!" << std::endl;
			else{
				mx<T, BLOCK_SIZE> cofactor(dim);
				std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            	std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
				mx<T, BLOCK_SIZE> M(*this, row + 1, col + 1);
				cofactor.enter_val(row + 1, col + 1, M.det());
				cofactor.transpoze();
				cofactor.multiply_scalar(1/det);
				this -> copy(cofactor);
			}
		}
		CUDA_CALLABLE_MEMBER mx inverse() const{
			mx<T, BLOCK_SIZE> result(dim);
			result.copy(*this);
			result.invert();
			return result;
		}
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const mx<T, BLOCK_SIZE>& M){
	M.print(out);
	return out;
}

#endif