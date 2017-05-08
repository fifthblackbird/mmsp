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
		diffsq += (Q[d] - P[d]) * (Q[d] - P[d]);
	return std::sqrt(dx * diffsq);
}

void generate(int dim, const char* filename)
{
	const complex<double> solid(0.35, 0.0);
	const complex<double> liquid(0.0, 0.0);
	const double meshres[NF] = {M_PI/2, M_PI/2, M_PI/2};
	const double R = 15.0 * meshres[0];

	if (dim==1) {
		int L=1024;
		GRID1D initGrid(NF+1, 0,L);
		for (int d=0; d<dim; d++) {
			dx(initGrid, d) = meshres[d];
			if (x0(initGrid,d) == g0(initGrid,d))
				b0(initGrid,d) = Neumann;
			else if (x1(initGrid,d) == g1(initGrid,d))
				b1(initGrid,d) = Neumann;
		}
		const vector<int> origin(1, L/2);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			initGrid(n) = (radius<1>(origin, x, meshres[0])<R) ? solid : liquid;
		}

		output(initGrid,filename);
	}

	if (dim==2) {
		int L=256;
		GRID2D initGrid(NF+1, 0,L, 0,L);
		for (int d=0; d<dim; d++) {
			dx(initGrid, d) = meshres[d];
			if (x0(initGrid,d) == g0(initGrid,d))
				b0(initGrid,d) = Neumann;
			else if (x1(initGrid,d) == g1(initGrid,d))
				b1(initGrid,d) = Neumann;
		}
		const vector<int> origin(2, L/2);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			initGrid(n) = (radius<2>(origin, x, meshres[0], meshres[1])<R) ? solid : liquid;
		}

		output(initGrid,filename);
	}

	if (dim==3) {
		int L=64;
		GRID3D initGrid(NF+1, 0,L, 0,L, 0,L);
		for (int d=0; d<dim; d++) {
			dx(initGrid, d) = meshres[d];
			if (x0(initGrid,d) == g0(initGrid,d))
				b0(initGrid,d) = Neumann;
			else if (x1(initGrid,d) == g1(initGrid,d))
				b1(initGrid,d) = Neumann;
		}
		const vector<int> origin(3, L/2);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			initGrid(n) = (radius<3>(origin, x, meshres[0], meshres[1], meshres[2])<R) ? solid : liquid;
		}

		output(initGrid,filename);
	}
}

template<int dim, typename T>
vector<complex<T> > covariantGradient(const grid<dim,vector<complex<T> >  > & GRID, const vector<int> & x, const vector<vector<T> >& k)
{
	// Encode the nonlinear Laplacian operator, (∇² + 2ⅈ k·∇), a.k.a. covariant gradient operator

	const complex<T> coeff(0.0, 2.0);

	// Take standard vector derivatives on n field variables with d dimensions
	vector<complex<T> > lap = laplacian(GRID, x); // (1 x d) complex vector
	const vector<vector<complex<T> > > grad = gradient(GRID, x); // (d x n) complex matrix... we want the (n x d) version

	// Combine Laplacian with gradient projected onto lattice vectors
	for (int d=0; d<dim; d++)
		for (int i=0; i<NF; i++)
			lap[i] += coeff * (k[i][d] * grad[d][i]);

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

	const T dt = 0.001;
	const T bet =  0.0;
	const T eps = -1.0;
	const T psi = -0.1; // average density

	// Populate k (lattice vectors) with reals expressed in complex form
	complex<T> r0i0(0.0, 0.0);
	vector<T> k0(2, 0.0); // k0 = (0,0)
	vector<vector<T> > k(NF, k0); // k = ((0,0),(0,0),(0,0)).
	k[0][0] =  std::sqrt(3.0)/2;
	k[0][1] = -0.5;
	k[1][0] =  0.0;
	k[1][1] =  1.0;
	k[2][0] = -std::sqrt(3.0)/2;
	k[2][1] = -0.5;

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

		#pragma omp parallel for
		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n); // index of this mesh point
			tempGrid(n) = covariantGradient(oldGrid, x, k);
		}
		ghostswap(tempGrid);

		#pragma omp parallel for
		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n); // index of this mesh point
			const vector<complex<T> >& oldGridN = oldGrid(n);
			vector<complex<T> >& newGridN = newGrid(n);
			const vector<complex<T> > lap = covariantGradient(tempGrid, x, k);

			// Store sum of norms-squared
			T allNormSq = 0.0;
			for (int i=0; i<NF; i++) {
				const T norm = oldGridN[i].norm();
				allNormSq += 2.0 * norm*norm;
			}

			// Store pairwise conjugates
			vector<complex<T> > conjugates(NF, r0i0);
			conjugates[0] = oldGridN[1].conj() * oldGridN[2].conj();
			conjugates[1] = oldGridN[0].conj() * oldGridN[2].conj();
			conjugates[2] = oldGridN[0].conj() * oldGridN[1].conj();

			for (int i=0; i<NF; i++) {
				const complex<T>& A = oldGridN[i];
				const T norm = A.norm();
				newGridN[i] = A - dt * ((eps - 2.0*bet*psi + 3.0*psi*psi) * A
				                        + (2.0*bet - 6.0*psi)*conjugates[i]
				                        + 3.0*(allNormSq - norm*norm) * A
				                        + lap[i]);
			}
		}

		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}

	// Convert amplitudes to density
	#pragma omp parallel for
	for (int n=0; n<nodes(oldGrid); n++) {
		vector<complex<T> >& oldGridN = oldGrid(n);
		vector<double> r = position(oldGrid, n);
		for (int d=0; d<dim; d++)
			r[d] *= dx(oldGrid, d);

		T density = psi;
		for (int i=0; i<NF; i++) {
			const T a = std::real(oldGridN[i].value());
			const T b = std::imag(oldGridN[i].value());
			density += 2.0 * (a*std::cos(k[i] * r) - b*std::sin(k[i] * r));
		}
		oldGridN[NF] = density;
	}
}

} // namespace MMSP

#endif

#include"MMSP.main.hpp"
