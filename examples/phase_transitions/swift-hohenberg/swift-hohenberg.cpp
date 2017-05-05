// swift-hohenberg.cpp
// Algorithms for 2D and 3D Swift-Hohenberg amplitude model
// Questions/comments to trevor.keller@gmail.com (Trevor Keller)

#ifndef SWIFTHOHENBERG_UPDATE
#define SWIFTHOHENBERG_UPDATE
#include "MMSP.hpp"
#include "MMSP.complex.hpp"
#include <cmath>
#include <complex>
#include <cassert>
#include "swift-hohenberg.hpp"

namespace MMSP{

const int NF = 3; // number of amplitude fields

template<int dim>
double radius(const vector<int> & P, const vector<int> & Q, const double dx, const double dy=0.0, const double dz=0.0)
{
	int diffsq = 0;
	for (int d=0; d<dim; d++)
		diffsq += std::pow(Q[d] - P[d], 2);
	return std::sqrt(dx * diffsq);
}

void generate(int dim, const char* filename)
{
	const complex<double> solid(0.35, 0.0);
	const complex<double> liquid(0.0, 0.0);
	const double meshres[NF] = {M_PI/2, M_PI/2, M_PI/2};

	// Before using it for science, let's make sure our complex class obeys the identities.
	if (1) {
		// Equality
		const complex<double> a( 0.5, -0.5);
		const complex<double> b( 0.5, -0.5);
		assert(a==b);
	}
	if (1) {
		// Addition
		const complex<double> a( 0.5, -0.5);
		const complex<double> b( 1.0, -1.0);
		const complex<double> c( 1.5, -1.5);
		assert(a+b == c);

		complex<double> d(a);
		d += b;
		assert(d == c);
	}
	if (1) {
		// Subtraction
		const complex<double> a( 0.5, -0.5);
		const complex<double> b( 1.0, -1.0);
		const complex<double> c(-0.5, 0.5);
		assert(a-b == c);

		complex<double> d(a);
		d -= b;
		assert(d == c);
	}
	if (1) {
		// Multiplication
		const complex<int> a( 2, -4);
		const complex<int> b( 3,  5);
		const complex<int> c(26, -2);
		assert(a*b == c);

		complex<int> d(a);
		d *= b;
		assert(d == c);
	}
	if (1) {
		// Division
		const complex<int> a( 40, -20);
		const complex<int> b( 5,  3);
		const complex<int> c(140/34, -220/34);
		const complex<int> d = a/b;
		assert(a/b == c);

		complex<int> e(a);
		e /= b;
		assert(e == c);
	}

	if (dim==1) {
		int L=1024;
		GRID1D initGrid(NF, 0,L);
		for (int d=0; d<dim; d++)
			dx(initGrid, d) = meshres[d];
		const vector<int> origin(1, L/2);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			initGrid(n) = (radius<1>(origin, x, meshres[0])<10*meshres[0]) ? solid : liquid;
		}

		output(initGrid,filename);
	}

	if (dim==2) {
		int L=256;
		GRID2D initGrid(NF, 0,L, 0,L);
		for (int d=0; d<dim; d++)
			dx(initGrid, d) = meshres[d];
		const vector<int> origin(2, L/2);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			initGrid(n) = (radius<2>(origin, x, meshres[0], meshres[1])<10*meshres[0]) ? solid : liquid;
		}

		output(initGrid,filename);
	}

	if (dim==3) {
		int L=64;
		GRID3D initGrid(NF, 0,L, 0,L, 0,L);
		for (int d=0; d<dim; d++)
			dx(initGrid, d) = meshres[d];
		const vector<int> origin(3, L/2);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			initGrid(n) = (radius<3>(origin, x, meshres[0], meshres[1], meshres[2])<10*meshres[0]) ? solid : liquid;
		}

		output(initGrid,filename);
	}
}

template<int dim, typename T>
vector<complex<T> > covariantGradient(const grid<dim,vector<complex<T> >  > & GRID, const vector<int> & x, const vector<vector<complex<T> > >& k)
{
	// Encode the nonlinear Laplacian operator, (∇² + 2ⅈ k·∇)

	// Take standard vector derivatives on n field variables with d dimensions
	vector<complex<T> > lap = laplacian(GRID, x); // (1 x d) complex vector
	const vector<vector<complex<T> > > grad = gradient(GRID, x); // (d x n) complex matrix... we want the (n x d) version

	// MMSP knows about vectors, but not matrices, so we have to transpose it manually.
	const complex<T> blank_c(0.0, 0.0);
	const vector<complex<T> > blank_v(dim, blank_c);
	vector<vector<complex<T> > > gradTranspose(fields(GRID), blank_v);
	for (int d=0; d<dim; d++)
		for (int i=0; i<fields(GRID); i++)
			gradTranspose[i][d] = grad[d][i];

	const complex<T> coeff(0.0, 2.0);

	for (int i=0; i<fields(GRID); i++) {
		lap[i] += coeff * (gradTranspose[i] * k[i]);
	}

	return lap;
}

template <int dim, typename T>
void update(grid<dim,vector<complex<T> > >& oldGrid, int steps)
{
	// Update the grid using the equation of motion,
	// which looks a lot like the Newell-Whitehead-Segel equation for traveling waves
	int rank=0;
    #ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif

	ghostswap(oldGrid);

	grid<dim,vector<complex<T> > > newGrid(oldGrid);
	grid<dim,vector<complex<T> > > tempGrid(oldGrid);

	const T dt = 1e-2;
	const T gam = 1.0;
	const complex<T> eps(0.5, 0.0);

	// Populate k with reals expressed in complex form
	complex<T> r0i0(0.0, 0.0);
	vector<complex<T> > k0(2, r0i0); // k0 = (0,0)
	vector<vector<complex<T> > > k(3, k0); // k = ((0,0),(0,0),(0,0)).
	k[0][0] = ( std::sqrt(3.0)/2, 0.0);
	k[0][1] = (-0.5,              0.0);
	k[1][0] = ( 0.0,              0.0);
	k[1][1] = ( 1.0,              0.0);
	k[2][0] = (-std::sqrt(3.0)/2, 0.0);
	k[2][1] = (-0.5,              0.0);

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n); // index of this mesh point
			tempGrid(n) = covariantGradient(oldGrid, x, k);
		}
		ghostswap(tempGrid);

		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n); // index of this mesh point
			const vector<complex<T> >& oldGridN = oldGrid(n);
			vector<complex<T> >& newGridN = newGrid(n);
			const vector<complex<T> > lap = covariantGradient(tempGrid, x, k);

			for (int i=0; i<fields(oldGrid); i++) {
				const complex<T>& A = oldGridN[i];
				newGridN[i] = A + dt * (eps - (gam*A.norm()*A.norm())*A - lap[i]);
			}
		}

		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}
}

} // namespace MMSP

#endif

#include"MMSP.main.hpp"
