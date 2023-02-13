#ifndef MX_H_
#define MX_H_

#include <iostream>
#include <ctime>

template <typename T>
class mx{
	std::size_t dim;
	T* val;
	public:
		// constructors
		mx(std::size_t n = 0, T v = 0){
			dim = n;
			if(dim == 0)
				val = nullptr;
			else{
				val = new T[dim * dim];
				this -> init(v);
			}
		}
		mx(const mx<T>& A, std::size_t r, std::size_t c){
			std::size_t dimA = A.get_dim();
			dim = dimA - 1;
			val = new T[dim * dim];
			for(int i = 1; i <= dimA; i++)
				for(int j = 1; j <= dimA; j++){
					if(i == r || j == c)
						continue;
					if(i > r && j > c)
						this -> set_val(i - 1, j - 1, A.get_val(i, j));
					else if(i > r)
						this -> set_val(i - 1, j, A.get_val(i, j));
					else if(j > c)
						this -> set_val(i, j - 1, A.get_val(i, j));
					else
						this -> set_val(i, j, A.get_val(i, j));
				}
		}

		// deconstructor
		~mx() { delete [] val; }

		// initializing
		void init(T v = 0){
			for(int i = 0; i < dim * dim; i++)
				val[i] = v;
		}

		// random initializing
		void random(){
			srand(time(NULL));
			for(int i = 0; i < dim * dim; i++)
				val[i] = rand();
		}
		void random_int(int range, int val_min = 0){
			srand(time(NULL));
			for(int i = 0; i < dim * dim; i++)
				val[i] = rand() % range + val_min;
		}

		// making identity matrix
		void identity(){
			this -> init();
			for(int i = 0; i < dim * dim; i += dim + 1)
				val[i] = 1;
		}

		// geting the 'dim' value
		std::size_t get_dim() const { return dim; }

		// geting 'val[]' value
		T operator[](std::size_t n) const{
			return val[n];
		}
		T get_val(std::size_t r, std::size_t c) const{
			return val[(r - 1) * dim + (c - 1)];
		}

		// entering 'val[]' value
		void set_val(std::size_t r, std::size_t c, T x){
			val[(r - 1) * dim + (c - 1)] = x;
		}

		// printing matrix to the stream
		void print(std::ostream& out = std::cout) const{
			for(int i = 0; i < dim * dim; i++)
				if(i % dim == 0)
					out << std::endl << val[i] << " ";
				else
					out << val[i] << " ";
			out << std::endl;
		}

		// copying matrix
		void copy(const mx<T>& A){
			if(this != &A){
				if(dim == A.get_dim())
					for(int i = 0; i < dim * dim; i++)
						val[i] = A[i];
				else
					std::cout << "[E]: Invalid dimention! Copying is not possible!" << std::endl;
			}
		}

		// assigning matrix
		mx& operator=(const mx<T>& A){
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
		bool operator==(const mx<T>& A) const{
			if(this == &A)
				return true;
			if(dim != A.get_dim())
				return false;
			for(int i = 0; i < dim * dim; i++)
				if(val[i] != A[i])
					return false;
			return true;
		}

		// adding matrix
		void add(const mx<T>& A){
			if(dim == A.get_dim())
				for(int i = 0; i < dim * dim; i++)
					val[i] += A[i];
			else
				std::cout << "[E]: Invalid dimention! Addition is not possible!" << std::endl;
		}
		mx operator+(const mx<T>& A) const{
			mx<T> result(dim);
			result.copy(*this);
			result.add(A);
			return result;
		}
		mx& operator+=(const mx<T>& A){
			this -> add(A);
			return *this;
		}

		// multiplying by scalar
		void multiply_scalar(double x){
			for(int i = 0; i < dim * dim; i++)
				val[i] *= x;
		}
		mx operator*(double x) const{
			mx<T> result(dim);
			result.copy(*this);
			result.multiply_scalar(-1.0);
			return result;
		}
		mx& operator*=(double x){
			this -> multiply_scalar(x);
			return *this;
		}

		// subtracting matrix
		void subtract(const mx<T>& A){
			this -> add(A*-1.0);
		}
		mx operator-(const mx<T>& A) const{
			mx<T> result(dim);
			result.copy(*this);
			result.subtract(A);
			return result;
		}
		mx& operator-=(const mx<T>& A){
			this -> subtract(A);
			return *this;
		}

		// transpozition
		void transpoze(){
			mx<T> temp(dim);
			temp.copy(*this);
			this -> init();
			for(int i = 1; i <= dim; i++)
				for(int j = 1; j <= dim; j++)
					this -> set_val(j, i, temp.get_val(i, j));
		}

		// multiplying by matrix
		void multiply_matrix(const mx<T>& A){
			if(dim != A.get_dim())
				std::cout << "[E]: Invalid dimention! Multiplying is not possible!" <<std::endl;
			else{
				mx<T> result(dim);
				T x;
				for(int i = 1; i <= dim; i++)
					for(int j = 1; j <= dim; j++){
						x = 0;
						for(int k = 1; k <= dim; k++)
							x += (this -> get_val(i, k)) * A.get_val(k, j);
						result.set_val(i, j, x);
					}
				this -> copy(result);
			}	
		}
		mx operator*(const mx<T>& A) const{
			mx<T> result(dim);
			result.copy(*this);
			result.multiply_matrix(A);
			return result;
		}
		mx& operator*=(const mx<T>& A){
			this -> multiply_matrix(A);
			return *this;
		}

		// determinant calculation helper
		int find_best_row(std::size_t& r) const{
			int n, n_max = 0;
			for(int i = 1; i <= dim; i++){
				n = 0;
				for(int j = 1; j <= dim; j++)
					if((this -> get_val(i, j)) == 0)
						n++;
				if(n > n_max){
					n_max = n;
					r = i;
				}
			}	
			return n_max;
		}

		int find_best_column(std::size_t& c) const{
			mx<T> temp(dim);
			temp.copy(*this);
			temp.transpoze();
			return temp.find_best_row(c);
		}

		// determinant
		T det() const{
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
							mx<T> M(*this, n, i);
							if((1 + i) % 2 == 0)
								x += (this -> get_val(n, i)) * M.det();
							else
								x -= (this -> get_val(n, i)) * M.det();
						}
					}
					else{
						n = c;
						for(int i = 1; i <= dim; i++){
							mx<T> M(*this, i, n);
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
		void invert(){
			T det = this -> det();
			if(!det)
				std::cout << "Matrix is noninversable!" << std::endl;
			else{
				mx<T> cofactor(dim);
				for(int i = 1; i <= dim; i++)
					for(int j = 1; j <= dim; j++){
						mx<T> M(*this, i, j);
						cofactor.set_val(i, j, M.det());
					}
				cofactor.transpoze();
				cofactor.multiply_scalar(1/det);
				this -> copy(cofactor);
			}
		}
		mx inverse() const{
			mx<T> result(dim);
			result.copy(*this);
			result.invert();
			return result;
		}
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const mx<T>& M){
	M.print(out);
	return out;
}



#endif