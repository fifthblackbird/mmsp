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

namespace par{
	// Define numerical constants in a "parameter" namespace

	// Number of fields
	int NF() { return 3; }

	// "Temperature" or system internal energy pareter
	double eps(){ return -0.2; }

	// Ternary energy coefficient
	double bet(){ return 0.0; }

	// Average field density
	double psi0(){ return -0.0; }

	// Timestep
	double dt() { return 0.01; }

	// Mesh resolution
	double dx(){ return M_PI/2; }

	// Coefficient of grad product on covariant gradient (2ⅈ)
	MMSP::complex<double> twoeye() { return MMSP::complex<double>(0.0, 2.0); }


}

namespace MMSP{

template<int dim>
double radius(const vector<int> & P, const vector<int> & Q, const double dx, const double dy=0.0, const double dz=0.0)
{
	int diffsq = 0;
	for (int d=0; d<dim; d++)
		diffsq += (Q[d] - P[d]) * (Q[d] - P[d]);
	return std::sqrt(dx * diffsq);
}

template <class T>
double bulk_energy(const vector<complex<T> >& dat, const double eps, const double bet, const double psi0)
{
	const double psiSq = psi0*psi0;
	double well = (eps/2 - bet/3)*psiSq + 0.25*psiSq*psiSq;

	double norms = 0;
	double pairs = 0;
	for (int i=0; i<par::NF(); i++) {
		const T normi = dat[i].norm();
		norms += (eps - 2.0*bet*psi0 + 3.0*psiSq) * normi*normi;

		T allNorms = 0.0;
		for (int j=i+1; j<par::NF(); j++) {
			const T normj = dat[j].norm();
			allNorms += 4.0*normj*normj;
		}
		pairs += 1.5 * normi*normi * (normi*normi + allNorms);
	}

	complex<double> ternaries(dat[0]);
	complex<double> conjugates(dat[0].conj());
	for (int i=1; i<par::NF(); i++) {
		ternaries *= dat[i];
		conjugates *= dat[i].conj();
	}
	double triplets = (2.0*bet - 6.0*psi0)*(std::real(ternaries.value()) + std::real(conjugates.value()));

	return well + norms + triplets + pairs;
}

void generate(int dim, const char* filename)
{
	const complex<double> l(0.0, 0.0);
	const complex<double> s(0.35, 0.0);

	vector<complex<double> > liquid(par::NF()+2, l);
	liquid[par::NF()] = par::psi0();

	vector<complex<double> > solid(par::NF()+2, s);
	solid[par::NF()] = par::psi0();
	solid[par::NF()+1] = l;

	const vector<double> meshres(3, par::dx());
	const double R = 30.0 * meshres[0];

	if (dim==1) {
		int L=1024;
		GRID1D initGrid(par::NF()+2, -L/2,L/2);
		for (int d=0; d<dim; d++) {
			dx(initGrid, d) = meshres[d];
		}
		const vector<int> origin(1, 0);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			initGrid(n) = (radius<1>(origin, x, meshres[0])<R) ? solid : liquid;
		}

		output(initGrid,filename);
	}

	if (dim==2) {
		int L=256;
		GRID2D initGrid(par::NF()+2, -L,L, -L/2,L/2);
		for (int d=0; d<dim; d++) {
			dx(initGrid, d) = meshres[d];
		}
		const vector<int> origin(2, 0);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			initGrid(n) = (radius<2>(origin, x, meshres[0], meshres[1])<R) ? solid : liquid;
		}

		output(initGrid,filename);
	}

	if (dim==3) {
		int rank=0;
		#ifdef MPI_VERSION
		rank = MPI::COMM_WORLD.Get_rank();
		#endif
		if (rank==0)
			std::cout<<"Warning: 3D is not implemented. This function exports free energy to "<<filename<<std::endl;

		FILE* ef = NULL;
		ef = fopen(filename, "w"); // file will be overwritten
		fprintf(ef, "eps,psi0,nrg\n");

		complex<double> a(0.35, 0.0);
		vector<complex<double> > A(par::NF(), a);

		for (double eps=0.0; eps>-4*M_PI; eps-=M_PI/16) {
			for (double psi=0.0; psi>-2*M_PI; psi-=M_PI/16) {
				double nrg = bulk_energy(A, eps, par::bet(), psi);
				fprintf(ef, "%f,%f,%f\n", eps,psi,nrg);
			}
		}
		fclose(ef);
		ef = NULL;
	}
}

template<int dim, typename T>
vector<complex<T> > covariantGradient(const grid<dim,vector<complex<T> >  > & GRID, const vector<int> & x, const vector<vector<T> >& k)
{
	// Encode the nonlinear Laplacian operator, (∇² + 2ⅈk·∇), a.k.a. covariant gradient operator

	// Take standard vector derivatives on n field variables with d dimensions
	vector<complex<T> > lap = laplacian(GRID, x); // (1 x d) complex vector
	const vector<vector<complex<T> > > grad = gradient(GRID, x); // (d x n) complex matrix... we want the (n x d) version

	// Amplitude fields
	for (int d=0; d<dim; d++)
		for (int i=0; i<par::NF(); i++)
			lap[i] += par::twoeye() * (k[i][d] * grad[d][i]);

	// Density field
	lap[par::NF()] += GRID(x)[par::NF()];

	return lap;
}

template <int dim, typename T>
void update(grid<dim,vector<complex<T> > >& oldGrid, int steps)
{
	// Update the grid using the coarse-grained equations of motion for a triangular lattice
	int rank=0;
    #ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif

	ghostswap(oldGrid);

	grid<dim,vector<complex<T> > > newGrid(oldGrid);
	grid<dim,vector<complex<T> > > lapGrid(oldGrid);
	grid<dim,complex<T> > psiGrid(oldGrid, 1);

	// Populate k (lattice vectors) with reals expressed in complex form
	complex<T> r0i0(0.0, 0.0);
	vector<T> k0(2, 0.0); // k0 = (0,0)
	vector<vector<T> > k(par::NF(), k0); // k = ((0,0),(0,0),(0,0)).
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
			lapGrid(n) = covariantGradient(oldGrid, x, k);
		}

		ghostswap(lapGrid);

		#pragma omp parallel for
		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n); // index of this mesh point
			const vector<complex<T> >& oldGridN = oldGrid(n);
			vector<complex<T> >& newGridN = newGrid(n);
			const vector<complex<T> > lap = covariantGradient(lapGrid, x, k);

			// Store sum of norms-squared
			T allNormSq = 0.0;
			for (int i=0; i<par::NF(); i++) {
				const T norm = oldGridN[i].norm();
				allNormSq += 2.0 * norm*norm;
			}

			// Store pairwise conjugates
			vector<complex<T> > conjpairs(par::NF(), r0i0);
			conjpairs[0] = oldGridN[1].conj() * oldGridN[2].conj();
			conjpairs[1] = oldGridN[0].conj() * oldGridN[2].conj();
			conjpairs[2] = oldGridN[0].conj() * oldGridN[1].conj();

			// Update amplitudes
			for (int i=0; i<par::NF(); i++) {
				const complex<T>& A = oldGridN[i];
				const T norm = A.norm();
				newGridN[i] = A - par::dt() * ((par::eps() - 2.0*par::bet()*par::psi0() + 3.0*par::psi0()*par::psi0()) * A
				                        + (2.0*par::bet() - 6.0*par::psi0())*conjpairs[i]
				                        + 3.0*(allNormSq - norm*norm) * A
				                        + lap[i]);
			}

			// Store ternary amplitudes and conjugates
			complex<T> ternaries = oldGridN[0];
			complex<T> conjugates = oldGridN[0].conj();
			for (int i=1; i<par::NF(); i++) {
				ternaries *= oldGridN[i];
				conjugates *= oldGridN[i].conj();
			}

			// Update density
			const complex<T>& psi = oldGridN[par::NF()];
			psiGrid(n) = par::eps() * psi - par::bet()*psi*psi + psi*psi*psi
			             + (6.0*psi - 2.0*complex<T>(par::bet(),0.0)) * allNormSq
			             - 6.0 * (ternaries + conjugates)
			             + lap[par::NF()];
		}

		ghostswap(psiGrid);

		#pragma omp parallel for
		for (int n=0; n<nodes(psiGrid); n++) {
			vector<int> x = position(oldGrid, n); // index of this mesh point
			vector<complex<T> >& newGridN = newGrid(n);
			const vector<complex<T> >& oldGridN = oldGrid(n);
			const complex<T> lap = laplacian(psiGrid, x);

			newGridN[par::NF()] = oldGridN[par::NF()] + par::dt() * lap;
		}

		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}

	// Convert amplitudes to density
	#pragma omp parallel for
	for (int n=0; n<nodes(oldGrid); n++) {
		vector<complex<T> >& oldGridN = oldGrid(n);
		vector<double> r = position(oldGrid, n);
		// Reconstruct the phases with a finer resolution
		// than the amplitude computation, just to get a
		// prettier picture of the lattice
		for (int d=0; d<dim; d++)
			r[d] *= dx(oldGrid, d)/2;

		complex<T> psi = oldGridN[par::NF()];
		for (int i=0; i<par::NF(); i++) {
			const T a = std::real(oldGridN[i].value());
			const T b = std::imag(oldGridN[i].value());
			psi += 2.0 * (a*std::cos(k[i] * r) - b*std::sin(k[i] * r));
		}
		oldGridN[par::NF()+1] = psi;
	}
}

} // namespace MMSP

#endif

#include"MMSP.main.hpp"
