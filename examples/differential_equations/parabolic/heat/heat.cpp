// heat.cpp
// Algorithms for 2D and 3D heat model
// Questions/comments to gruberja@gmail.com (Jason Gruber)

#ifndef HEAT_UPDATE
#define HEAT_UPDATE
#include"MMSP.hpp"
#include<cmath>
#include<cassert>
#include"heat.hpp"
using std::cos;

namespace MMSP{


double MS_T(const vector<double>& xcoeff, // spatial coefficients
            const vector<double>& tcoeff, // temporal coefficients
            double D,                     // time coefficient
            const vector<double>& h,      // grid spacing
            const vector<int>& x,         // spatial indices
            double t)                     // time
{
	// Return the manufactured solution T(x,y,z,t)
	double temp = cos(D*t);
	for (int i=0; i<x.size(); i++)
		temp *= cos(xcoeff[i]*h[i]*x[i] + tcoeff[i]*t);
	return temp;
}

void SourceQ (const vector<int> xidx, vector<double> h, double t,
              vector<double> Ax, // spatial amplitudes
              vector<double> At, // temporal amplitudes
              vector<double> k,  // thermal diffusivities
              vector<double> Cp, // heat capacities
              double rho)
{
	vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.size(); d++)
		x[d] = h[d]*xidx[d];
	assert(Ax.size() == 3);
	assert(k.size() == 3);
	assert(Cp.size() == 3);
	assert(At.size() == 4);

	double Q_T;       // thermal source term
	double T_an;      // analytical temperature
	double *gradT_an; // analytical temperature gradient

	// Define the thermal source term. Transformed for MMSP accessors and restructured for clarity and linewidth
	// after https://github.com/manufactured-solutions/analytical/blob/master/heat_equation/C_code/Temp_3d_unsteady_k_cp_var.C
	// For best results, generate this yourself using a symbolic math package, e.g. SymPy, Mathematica, or Maple.
	// All this floating-point math in a single expression makes me nervous.
	Q_T = - rho * Cp[0] * At[0] * sin(Ax[0] * x[0] + At[0] * t)
	                            * cos(Ax[1] * x[1] + At[1] * t)
	                            * cos(Ax[2] * x[2] + At[2] * t)
	                            * cos(At[3] * t)
	      - rho * Cp[0] * At[1] * cos(Ax[0] * x[0] + At[0] * t)
	                            * sin(Ax[1] * x[1] + At[1] * t)
	                            * cos(Ax[2] * x[2] + At[2] * t)
	                            * cos(At[3] * t)
	      - rho * Cp[0] * At[2] * cos(Ax[0] * x[0] + At[0] * t)
	                            * cos(Ax[1] * x[1] + At[1] * t)
	                            * sin(Ax[2] * x[2] + At[2] * t)
	                            * cos(At[3] * t)
	      - rho * Cp[0] * At[3] * cos(Ax[0] * x[0] + At[0] * t)
	                            * cos(Ax[1] * x[1] + At[1] * t)
	                            * cos(Ax[2] * x[2] + At[2] * t)
	                            * sin(At[3] * t)
	      - k[1] * Ax[0] * Ax[0] * pow(cos(Ax[1] * x[1] + At[1] * t), 2.0)
	                             * pow(cos(Ax[2] * x[2] + At[2] * t), 2.0)
	                             * pow(cos(At[3] * t), 2.0)
	      - k[1] * Ax[1] * Ax[1] * pow(cos(Ax[0] * x[0] + At[0] * t), 2.0)
	                             * pow(cos(Ax[2] * x[2] + At[2] * t), 2.0)
	                             * pow(cos(At[3] * t), 2.0)
	      - k[1] * Ax[2] * Ax[2] * pow(cos(Ax[0] * x[0] + At[0] * t), 2.0)
	                             * pow(cos(Ax[1] * x[1] + At[1] * t), 2.0)
	                             * pow(cos(At[3] * t), 2.0)
	      + 3.0 * k[2] * ( Ax[0] * Ax[0]
	                     + Ax[1] * Ax[1]
	                     + Ax[2] * Ax[2]
	                     ) * pow(cos(Ax[0] * x[0] + At[0] * t), 3.0)
	                       * pow(cos(Ax[1] * x[1] + At[1] * t), 3.0)
	                       * pow(cos(Ax[2] * x[2] + At[2] * t), 3.0)
	                       * pow(cos(At[3] * t), 3.0)
	      + (
	        - rho * Cp[1] * At[0] * sin(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cp[1] * At[1] * cos(Ax[0] * x[0] + At[0] * t)
	                              * sin(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cp[1] * At[2] * cos(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * sin(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cp[1] * At[3] * cos(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * sin(At[3] * t)
	        + k[0] * Ax[0] * Ax[0] + k[0] * Ax[1] * Ax[1] + k[0] * Ax[2] * Ax[2]
	        - 2.0 * k[2] * Ax[0] * Ax[0] * pow(cos(Ax[1] * x[1] + At[1] * t), 2.0)
	                                     * pow(cos(Ax[2] * x[2] + At[2] * t), 2.0)
	                                     * pow(cos(At[3] * t), 2.0)
	        - 2.0 * k[2] * Ax[1] * Ax[1] * pow(cos(Ax[0] * x[0] + At[0] * t), 2.0)
	                                     * pow(cos(Ax[2] * x[2] + At[2] * t), 2.0)
	                                     * pow(cos(At[3] * t), 2.0)
	        - 2.0 * k[2] * Ax[2] * Ax[2] * pow(cos(Ax[0] * x[0] + At[0] * t), 2.0)
	                                     * pow(cos(Ax[1] * x[1] + At[1] * t), 2.0)
	                                     * pow(cos(At[3] * t), 2.0)
	        ) * cos(Ax[0] * x[0] + At[0] * t)
	          * cos(Ax[1] * x[1] + At[1] * t)
	          * cos(Ax[2] * x[2] + At[2] * t)
	          * cos(At[3] * t)
	      + (
	        - rho * Cp[2] * At[0] * sin(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cp[2] * At[1] * cos(Ax[0] * x[0] + At[0] * t)
	                              * sin(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cp[2] * At[2] * cos(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * sin(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cp[2] * At[3] * cos(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * sin(At[3] * t)
	        + 2.0 * k[1] * Ax[0] * Ax[0]
	        + 2.0 * k[1] * Ax[1] * Ax[1]
	        + 2.0 * k[1] * Ax[2] * Ax[2]
	        ) * pow(cos(Ax[0] * x[0] + At[0] * t), 2.0)
	          * pow(cos(Ax[1] * x[1] + At[1] * t), 2.0)
	          * pow(cos(Ax[2] * x[2] + At[2] * t), 2.0)
	          * pow(cos(At[3] * t), 2.0);

	T_an = cos(Ax[0] * x[0] + At[0] * t) * cos(Ax[1] * x[1] + At[1] * t) * cos(Ax[2] * x[2] + At[2] * t) * cos(At[3] * t);
	gradT_an[0] = -Ax[0] * cos(At[3] * t) * cos(Ax[1] * x[1] + At[1] * t) * cos(Ax[2] * x[2] + At[2] * t) * sin(Ax[0] * x[0] + At[0] * t);
	gradT_an[1] = -Ax[1] * cos(At[3] * t) * cos(Ax[0] * x[0] + At[0] * t) * cos(Ax[2] * x[2] + At[2] * t) * sin(Ax[1] * x[1] + At[1] * t);
	gradT_an[2] = -Ax[2] * cos(At[3] * t) * cos(Ax[0] * x[0] + At[0] * t) * cos(Ax[1] * x[1] + At[1] * t) * sin(Ax[2] * x[2] + At[2] * t);
}

void generate(int dim, const char* filename)
{
	if (dim==1) {
		int L=1024;
		GRID1D initGrid(0,0,L);

		for (int i=0; i<nodes(initGrid); i++)
			initGrid(i) = 1.0-2.0*double(rand())/double(RAND_MAX);

		output(initGrid,filename);
	}

	if (dim==2) {
		int L=256;
		GRID2D initGrid(0,0,2*L,0,L);

		for (int i=0; i<nodes(initGrid); i++)
			initGrid(i) = 1.0-2.0*double(rand())/double(RAND_MAX);

		output(initGrid,filename);
	}

	if (dim==3) {
		int L=64;
		GRID3D initGrid(0,0,2*L,0,L,0,L/4);

		for (int i=0; i<nodes(initGrid); i++)
			initGrid(i) = 1.0-2.0*double(rand())/double(RAND_MAX);

		output(initGrid,filename);
	}
}

template <int dim, typename T> void update(grid<dim,T>& oldGrid, int steps)
{
	int rank=0;
    #ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif

	ghostswap(oldGrid);

	grid<dim,T> newGrid(oldGrid);

	double r = 1.0;
	double u = 1.0;
	double K = 1.0;
	double M = 1.0;
	double dt = 0.01;

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

		for (int i=0; i<nodes(oldGrid); i++) {
			T phi = oldGrid(i);
			newGrid(i) = phi-dt*M*(-r*phi+u*pow(phi,3)-K*laplacian(oldGrid,i));
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}
}

} // namespace MMSP

#endif

#include"MMSP.main.hpp"
