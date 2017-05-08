// MMSP.complex.h
// Class definition for the MMSP complex data structure
// Questions/comments to trevor.keller@gmail.com (Trevor Keller)

#ifndef MMSP_COMPLEX
#define MMSP_COMPLEX
#include <complex>
#include"MMSP.utility.h"

namespace MMSP {

template <typename T> class complex {
public:
	// constructors
	complex() {}
	complex(const std::complex<T>& value) {
		data = value;
	}
	complex(const T& re, const T& im) {
		data = std::complex<T>(re, im);
	}
	complex(const complex& s) {
		data = s.data;
	}
	template <typename U> complex(const std::complex<U>& s) {
		data = std::complex<T>(s);
	}

	// data access operators
	operator std::complex<T>&() {
		return data;
	}
	operator const std::complex<T>&() const {
		return data;
	}
	std::complex<T>& value() {
		return data;
	}
	const std::complex<T>& value() const {
		return data;
	}

	// assignment operators
	complex& operator=(const std::complex<T>& value) {
		data = value;
		return *this;
	}
	complex& operator=(const complex& s) {
		data = s.value();
		return *this;
	}
	template <typename U> complex& operator=(const U& value) {
		data = std::complex<T>(value);
		return *this;
	}
	template <typename U> complex& operator=(const complex<U>& s) {
		data = std::complex<T>(s.value());
		return *this;
	}

	// buffer I/O functions
	int buffer_size() const {
		return sizeof(std::complex<T>);
	}
	int to_buffer(char* buffer) const {
		memcpy(buffer, &data, sizeof(std::complex<T>));
		return sizeof(std::complex<T>);
	}
	int from_buffer(const char* buffer) {
		memcpy(&data, buffer, sizeof(std::complex<T>));
		return sizeof(std::complex<T>);
	}

	// file I/O functions
	void write(std::ofstream& file) const {
		file.write(reinterpret_cast<const char*>(&data), sizeof(std::complex<T>));
	}
	void read(std::ifstream& file) {
		file.read(reinterpret_cast<char*>(&data), sizeof(std::complex<T>));
	}

	// utility functions
	double norm() const {
		return std::norm(data);
	}
	complex conj() const {
		return complex(std::conj(data));
	}
	int length() const {
		return 1;
	}
	void resize(int n) {}
	void copy(const complex& s) {
		memcpy(data, s.data, sizeof(std::complex<T>));
	}
	void swap(complex& s) {
		std::complex<T> temp = data;
		data = s.data;
		s.data = temp;
	}

private:
	// object data
	std::complex<T> data;
};

// buffer I/O functions
template <typename T> int buffer_size(const complex<T>& s) {
	return s.buffer_size();
}
template <typename T> int to_buffer(const complex<T>& s, char* buffer) {
	return s.to_buffer(buffer);
}
template <typename T> int from_buffer(complex<T>& s, const char* buffer) {
	return s.from_buffer(buffer);
}

// file I/O functions
template <typename T> void write(const complex<T>& s, std::ofstream& file) {
	return s.write(file);
}
template <typename T> void read(complex<T>& s, std::ifstream& file) {
	return s.read(file);
}

// utility functions
template <typename T> int length(const complex<T>& s) {
	return s.length();
}
template <typename T> void resize(complex<T>& s, int n) {
	s.resize(n);
}
template <typename T> void copy(complex<T>& s, const complex<T>& t) {
	s.copy(t);
}
template <typename T> void swap(complex<T>& s, complex<T>& t) {
	s.swap(t);
}
template <typename T> std::string name(const complex<T>& s) {
	return std::string("complex:") + name(T());
}

// mathematical operators
template <typename T, typename U> bool operator==(const complex<T>& x, const complex<U>& y) {
	return x.value() == y.value();
}
template <typename T> bool operator==(const complex<T>& x, const complex<T>& y) {
	return x.value() == y.value();
}
template <typename T, typename U> complex<T>& operator+=(complex<T>& x, const complex<U>& y) {
	x.value() += y.value();
	return x;
}
template <typename T, typename U> complex<T> operator+(const complex<T>& x, const complex<U>& y) {
	complex<T> z(x);
	z += y;
	return z;
}
template <typename T, typename U> complex<T>& operator-=(complex<T>& x, const complex<U>& y) {
	x.value() -= y.value();
	return x;
}
template <typename T, typename U> complex<T> operator-(const complex<T>& x, const complex<U>& y) {
	complex<T> z(x);
	z -= y;
	return z;
}
template <typename T, typename U> complex<T>& operator*=(complex<T>& x, const complex<U>& y) {
	x.value() *= y.value();
	return x;
}
template <typename T, typename U> complex<T>& operator*=(complex<T>& x, const U& value) {
	x.value() *= value;
	return x;
}
template <typename T, typename U> complex<T> operator*(const complex<T>& x, const U& value) {
	complex<T> z(x);
	z.value() *= value;
	return z;
}
template <typename T, typename U> complex<T> operator*(const U& value, const complex<T>& x) {
	complex<T> z(x);
	z.value() *= value;
	return z;
}
template <typename T> complex<T> operator*(const complex<T>& x, const std::complex<T>& y) {
	complex<T> z(x);
	z *= y;
	return z;
}
template <typename T> complex<T> operator*(const std::complex<T>& x, const complex<T>& y) {
	complex<T> z(x);
	z *= y;
	return z;
}
template <typename T, typename U> complex<T>& operator/=(complex<T>& x, const complex<U>& y) {
	x.value() /= y.value();
	return x;
}
template <typename T, typename U> complex<T>& operator/=(complex<T>& x, const U& value) {
	x.value() /= value;
	return x;
}
template <typename T> complex<T> operator/(const complex<T>& x, const std::complex<T>& y) {
	complex<T> z(x);
	z /= y;
	return z;
}
template <typename T> complex<T> operator*(const complex<T>& x, const complex<T>& y) {
	complex<T> z(x);
	z.value() *= y.value();
	return z;
}
template <typename T> complex<T> operator/(const complex<T>& x, const complex<T>& y) {
	complex<T> z(x);
	z /= y;
	return z;
}


// target class: dim = 0 specialization for complex class
template <int ind, typename T>
class target<0, ind, complex<T> > {
public:
	// constructor
	target(complex<T>* DATA, const int* S0, const int* SX, const int* X0, const int* X1, const int* B0, const int* B1) {
		data = DATA;
		s0 = S0;
		sx = SX;
		x0 = X0;
		x1 = X1;
		b0 = B0;
		b1 = B1;
	}

	// data access operators
	operator std::complex<T>&() {
		return *data;
	}
	operator const std::complex<T>&() const {
		return *data;
	}

	// assignment operators
	complex<T>& operator=(const T& value) const {
		return data->operator=(value);
	}
	complex<T>& operator=(const complex<T>& s) const {
		return data->operator=(s);
	}
	template <typename U> complex<T>& operator=(const U& value) const {
		return data->operator=(value);
	}
	template <typename U> complex<T>& operator=(const complex<U>& s) const {
		return data->operator=(s);
	}

	// buffer I/O functions
	int buffer_size() const {
		return data->buffer_size();
	}
	int to_buffer(char* buffer) const {
		return data->to_buffer(buffer);
	}
	int from_buffer(const char* buffer) const {
		return data->from_buffer(buffer);
	}

	// file I/O functions
	void write(std::ofstream& file) const {
		data->write(file);
	}
	void read(std::ifstream& file) const {
		data->read(file);
	}

	// utility functions
	int length() const {
		return data->length();
	}
	int resize(int n) const {
		return data->resize(n);
	}
	void copy(const target& t) const {
		data->copy(t->data);
	}
	void swap(const target& t) const {
		data->swap(t->data);
	}

	// object data
	complex<T>* data;
	const int* s0;
	const int* sx;
	const int* x0;
	const int* x1;
	const int* b0;
	const int* b1;
};

// buffer I/O functions
template <int ind, typename T> int buffer_size(const target<0, ind, complex<T> >& s) {
	return s.buffer_size();
}
template <int ind, typename T> int to_buffer(const target<0, ind, complex<T> >& s, char* buffer) {
	return s.to_buffer(buffer);
}
template <int ind, typename T> int from_buffer(const target<0, ind, complex<T> >& s, const char* buffer) {
	return s.from_buffer(buffer);
}

// file I/O functions
template <int ind, typename T> void write(const target<0, ind, complex<T> >& s, std::ofstream& file) {
	return s.write(file);
}
template <int ind, typename T> void read(const target<0, ind, complex<T> >& s, std::ifstream& file) {
	return s.read(file);
}

// utility functions
template <int ind, typename T> int length(const target<0, ind, complex<T> >& s) {
	return s.length();
}
template <int ind, typename T> void resize(const target<0, ind, complex<T> >& s, int n) {
	s.resize(n);
}
template <int ind, typename T> void copy(const target<0, ind, complex<T> >& s, const target<0, ind, complex<T> >& t) {
	s.copy(t);
}
template <int ind, typename T> void swap(const target<0, ind, complex<T> >& s, const target<0, ind, complex<T> >& t) {
	s.swap(t);
}
template <int ind, typename T> std::string name(const target<0, ind, complex<T> >& s) {
	return std::string("complex:") + name(T());
}

} // namespace MMSP

#endif
