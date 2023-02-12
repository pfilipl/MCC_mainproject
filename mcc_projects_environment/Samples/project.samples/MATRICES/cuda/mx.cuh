#ifndef MX_H_
#define MX_H_

#include <iostream>
#include <cstdlib>
#include <ctime>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <typename T>
void printMatrix(T& val, std::size_t dim){
	printf("Hello world!");
	// for(int i = 0; i < dim * dim; i++)
	// 	if(i % dim == 0)
	// 		std::cout << std::endl << val[i] << " ";
	// 	else
	// 		std::cout << val[i] << " ";
	// std::cout << std::endl;
}

template <class T>
class mx{
	std::size_t dim;
	T* val;

	public:
		T** val_p = &val;

		__host__ void devAlloc(std::size_t n){
			dim = n;
			gpuErrchk(cudaMalloc(&val, dim * dim * sizeof(T)));
		}

		__host__ void devFree(){
			gpuErrchk(cudaFree(val));
			dim = 0;
		}

		// geting the 'dim' value
		__host__ __device__ std::size_t get_dim() const { return dim; }

		// geting the length of 'val'
		__host__ __device__ std::size_t len() const { return dim * dim * sizeof(T); }

		// geting 'val[]' value
		__host__ __device__ T operator[](std::size_t n) const { return val[n]; }
		__host__ __device__ T get_val(std::size_t r, std::size_t c) const { return val[(r - 1) * dim + (c - 1)]; }

		// setting 'val[]' value
		__host__ __device__ void set_val(std::size_t r, std::size_t c, T x) { val[(r - 1) * dim + (c - 1)] = x; }

		// printing matrix to the stream
		__host__ void print(std::ostream& out) const{
			T* p = new T[dim * dim];
			gpuErrchk(cudaMemcpy(p, val, this -> len(), cudaMemcpyDeviceToHost));
			for(int i = 0; i < dim * dim; i++)
				if(i % dim == 0)
					out << std::endl << p[i] << " ";
				else
					out << p[i] << " ";
			out << std::endl;
			delete p;
		}

		// constructors
		__host__ __device__ mx(): dim{0}, val{nullptr} {}

		__host__ mx(dim3 NOB, dim3 TPB, std::size_t n, T v = 0){
			this -> devAlloc(n);
			this -> init(NOB, TPB, v);
		}

		__host__ mx(dim3 NOB, dim3 TPB, mx<T> A, std::size_t r = 0, std::size_t c = 0){
			if(r * c == 0){
				this -> devAlloc(A.get_dim());
				this -> copy(NOB, TPB, A);
			}
			else{
				this -> devAlloc(A.get_dim() - 1);
				this -> minor(NOB, TPB, r, c, A);
			}
		}

		// deconstructor
		__host__ __device__ ~mx() {}

		// initializing
		__host__ void init(dim3 NOB, dim3 TPB, T v = 0){
			init_gpu<<< NOB, TPB >>>(*this, v);
			gpuErrchk(cudaPeekAtLastError());
    		gpuErrchk(cudaDeviceSynchronize());
		}

		__host__ void identity(dim3 NOB, dim3 TPB){
			identity_gpu<<< NOB, TPB >>>(*this);
			gpuErrchk(cudaPeekAtLastError());
    		gpuErrchk(cudaDeviceSynchronize());
		}

		// random initializing
		__host__ void random(){
			srand(time(NULL));
			T* p = new T[dim * dim];
			for(int i = 0; i < dim * dim; i++)
				p[i] = rand();
			gpuErrchk(cudaMemcpy(val, p, this -> len(), cudaMemcpyHostToDevice));
			delete p;
		}

		__host__ void random_int(int range, int val_min = 0){
			srand(time(NULL));
			T* p = new T[dim * dim];
			for(int i = 0; i < dim * dim; i++)
				p[i] = rand() % range + val_min;
			gpuErrchk(cudaMemcpy(val, p, this -> len(), cudaMemcpyHostToDevice));
			delete p;
		}

		// making minor
		__host__ void minor(dim3 NOB, dim3 TPB, std::size_t r, std::size_t c, mx<T> A){
			minor_gpu<<< NOB, TPB >>>(*this, r, c, A);
			gpuErrchk(cudaPeekAtLastError());
    		gpuErrchk(cudaDeviceSynchronize());
		}

		__host__ void minor(dim3 NOB, dim3 TPB, std::size_t r, std::size_t c){
			mx<T> temp(NOB, TPB, *this);
			dim--;
			minor_gpu<<< NOB, TPB >>>(*this, r, c, temp);
			gpuErrchk(cudaPeekAtLastError());
    		gpuErrchk(cudaDeviceSynchronize());
			temp.devFree();
		}

		// copying matrix
		__host__ void copy(dim3 NOB, dim3 TPB, mx<T> A){
			if(this != &A){
				if(val == nullptr)
					this -> devAlloc(A.get_dim());
				if(dim == A.get_dim()){
					copy_gpu<<< NOB, TPB >>>(*this, A);
                    gpuErrchk(cudaPeekAtLastError());
    				gpuErrchk(cudaDeviceSynchronize());
                }		
				else
					std::cout << "[E]: Invalid dimention! Copying is not possible!" << std::endl;
			}
		}

		// assigning matrix
		// __host__ __device__ mx& operator=(const mx<T>& A){
		// 	if(this == &A)
		// 		return *this;
		// 	if(dim != A.get_dim()){
		// 		dim = A.get_dim();
		// 		cudaFreeHost(val); // delete [] val;
		// 		cudaMallocManaged(&val, dim * dim * sizeof(T)); // val = new T[dim * dim];
		// 	}
		// 	this -> copy(A);
		// 	return *this;
		// }

		// matrices comparison
		__host__ __device__ bool operator==(const mx<T>& A) const{
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
		__host__ __device__ void add(const mx<T>& A){
			if(dim == A.get_dim()){
                std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
                std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
				val[row * dim + col] += A[row * dim + col];
            }
			else
				std::cout << "[E]: Invalid dimention! Addition is not possible!" << std::endl;
		}
		// __host__ __device__ mx operator+(const mx<T>& A) const{
		// 	mx<T> result(dim);
		// 	result.copy(*this);
		// 	result.add(A);
		// 	return result;
		// }
		// __host__ __device__ mx& operator+=(const mx<T>& A){
		// 	this -> add(A);
		// 	return *this;
		// }

		// multiplying by scalar
		__host__ __device__ void multiply_scalar(double x){
			std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
            std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
			val[row * dim + col] *= x;
		}
		// __host__ __device__ mx operator*(double x) const{
		// 	mx<T> result(dim);
		// 	result.copy(*this);
		// 	result.multiply_scalar(-1.0);
		// 	return result;
		// }
		// __host__ __device__ mx& operator*=(double x){
		// 	this -> multiply_scalar(x);
		// 	return *this;
		// }

		// subtracting matrix
		// __host__ __device__ void subtract(const mx<T>& A){
		// 	this -> add(A*-1.0);
		// }
		// __host__ __device__ mx operator-(const mx<T>& A) const{
		// 	mx<T> result(dim);
		// 	result.copy(*this);
		// 	result.subtract(A);
		// 	return result;
		// }
		// __host__ __device__ mx& operator-=(const mx<T>& A){
		// 	this -> subtract(A);
		// 	return *this;
		// }

		// transpozition
		// __host__ __device__ void transpoze(){
		// 	mx<T> temp(dim);
		// 	temp.copy(*this);
		// 	this -> init();
		// 	std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
        //     std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
		// 	this -> set_val(col + 1, row + 1, temp.get_val(row + 1, col + 1));
		// }

		// multiplying by matrix
		// __host__ __device__ void multiply_matrix(const mx<T>& A){
		// 	if(dim != A.get_dim())
		// 		std::cout << "[E]: Invalid dimention! Multiplying is not possible!" <<std::endl;
		// 	else{
		// 		mx<T> result(dim);
		// 		T x;
		// 		std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
        //     	std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
		// 		for(int k = 1; k <= dim; k++){
		// 			if(k == 1)
		// 				x = 0;
		// 				x += (this -> get_val(row + 1, k)) * A.get_val(k, col + 1);
		// 		}
		// 		result.set_val(row + 1, col + 1, x);
		// 		this -> copy(result);
		// 	}	
		// }
		// __host__ __device__ mx operator*(const mx<T>& A) const{
		// 	mx<T> result(dim);
		// 	result.copy(*this);
		// 	result.multiply_matrix(A);
		// 	return result;
		// }
		// __host__ __device__ mx& operator*=(const mx<T>& A){
		// 	this -> multiply_matrix(A);
		// 	return *this;
		// }

		// determinant calculation helper
		__host__ __device__ int find_best_row(std::size_t& r) const{
			int n, n_max = 0;
			std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
			n = 0;
			for(int j = 1; j <= dim; j++)
				if((this -> get_val(row + 1, j)) == 0)
					n++;
			if(n > n_max){
				n_max = n;
				r = row;
			}	
			return n_max;
		}

		// __host__ __device__ int find_best_column(std::size_t& c) const{
		// 	mx<T> temp(dim);
		// 	temp.copy(*this);
		// 	temp.transpoze();
		// 	return temp.find_best_row(c);
		// }

		// determinant
		// __host__ __device__ T det() const{
		// 	T x = 0;
		// 	switch(dim){
		// 		case 0:
		// 			std::cout << "[E]: Invalid dimention!" << std::endl;
		// 			break;
		// 		case 1:
		// 			x = val[0];
		// 			break;
		// 		case 2:
		// 			x = val[0] * val[3] - val[1] * val[2];
		// 			break;
		// 		case 3:
		// 			x += val[0] * val[4] * val[8];
		// 			x += val[3] * val[7] * val[2];
		// 			x += val[6] * val[1] * val[5];
		// 			x -= val[2] * val[4] * val[6];
		// 			x -= val[5] * val[7] * val[0];
		// 			x -= val[8] * val[1] * val[3];
		// 			break;
		// 		default:
		// 			std::size_t r = 1, c = 1, n;
		// 			int zero_count_r = this -> find_best_row(r);
		// 			int zero_count_c = this -> find_best_column(c);
		// 			if(zero_count_r >= zero_count_c){
		// 				n = r;
		// 				for(int i = 1; i <= dim; i++){
		// 					mx<T> M(*this, n, i);
		// 					if((1 + i) % 2 == 0)
		// 						x += (this -> get_val(n, i)) * M.det();
		// 					else
		// 						x -= (this -> get_val(n, i)) * M.det();
		// 				}
		// 			}
		// 			else{
		// 				n = c;
		// 				for(int i = 1; i <= dim; i++){
		// 					mx<T> M(*this, i, n);
		// 					if((1 + i) % 2 == 0)
		// 						x += (this -> get_val(i, n)) * M.det();
		// 					else
		// 						x -= (this -> get_val(i, n)) * M.det();
		// 				}
		// 			}
		// 	}
		// 	return x;
		// }

		// inverse matrix
		// __host__ __device__ void invert(){
		// 	T det = this -> det();
		// 	if(!det)
		// 		std::cout << "Matrix is noninversable!" << std::endl;
		// 	else{
		// 		mx<T> cofactor(dim);
		// 		std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
        //     	std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
		// 		mx<T> M(*this, row + 1, col + 1);
		// 		cofactor.set_val(row + 1, col + 1, M.det());
		// 		cofactor.transpoze();
		// 		cofactor.multiply_scalar(1/det);
		// 		this -> copy(cofactor);
		// 	}
		// }
		// __host__ __device__ mx inverse() const{
		// 	mx<T> result(dim);
		// 	result.copy(*this);
		// 	result.invert();
		// 	return result;
		// }
};

template<typename T>
__global__ void init_gpu(mx<T> A, T v = 0){
	std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
	std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	std::size_t dim = A.get_dim();
	if(row < dim && col < dim)
		A.set_val(row + 1, col + 1, v);
}

template <typename T>
__global__ void copy_gpu(mx<T> R, mx<T> A){
	std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
	std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	std::size_t dim = R.get_dim();
	if(row < dim && col < dim)
		R.set_val(row + 1, col + 1, A.get_val(row + 1, col + 1));
}

template <typename T>
__global__ void minor_gpu(mx<T> M, std::size_t r, std::size_t c, mx<T> A){
	r--;
	c--;
	std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
	std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	std::size_t dim = M.get_dim();
	if(row < dim + 1 && col < dim + 1){
		if(row > r && col > c)
			M.set_val(row, col, A.get_val(row + 1, col + 1));
		else if(row > r)
			M.set_val(row, col + 1, A.get_val(row + 1, col + 1));
		else if(col > c)
			M.set_val(row + 1, col, A.get_val(row + 1, col + 1));
		else if(row != r && col != c)
			M.set_val(row + 1, col + 1, A.get_val(row + 1, col + 1));
	}
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const mx<T>& A){
	A.print(out);
	return out;
}

// making identity matrix
template <typename T>
__global__ void identity_gpu(mx<T> A){
	std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
	std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	std::size_t dim = A.get_dim();
	if(row < dim && col < dim){
		A.set_val(row + 1, col + 1, 0);
		if(row == col)
			A.set_val(row + 1, col + 1, 1);
	}
}

// template <typename T>
// __global__ void test(mx<T> A, T v){
// 	std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;
// 	std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
// 	if(row < A.get_dim() && col < A.get_dim())
// 		A.set_val(row + 1, col + 1, v);
// }

#endif