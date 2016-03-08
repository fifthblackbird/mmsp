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

coefficients::coefficients() : // constructor
		// call initializing constructor on vector-valued members to enforce 3 spatial dimensions (otherwise, high risk of segfaulting)
		N(3,0),    // grid points
		h(3,0.0),  // spatial resolution
		Ax(3,0.0), // amplitudes in space
		At(4,0.0), // amplitudes in time
		Ck(3,0.0), // thermal conductivity polynomial coefficients
		Cc(3,0.0)  // heat capacity polynomial coefficients
{
		// Set spatially-independent coefficients
		dt = 0.01;
		rho = 1.0;   // What units?
		At[3] = 1.0; // fourth dimension is time -- cannot be zero, unless you implement an iterative solver for steady-state
		Ck[0] = 1.0;    Ck[1] = 1.0;    Ck[2] = 1.0;
		Cc[0] = 1.0;    Cc[1] = 1.0;    Cc[2] = 1.0;

		// Set coeffients for each dimension. If a dimension does not exist, leave its value zero.
		if (dim>0) { // x-axis
			N[0]=1024;  h[0]=1.0/N[0];    Ax[0] = 1.0;    At[0] = 1.0;
		}
		if (dim>1) { // y-axis
			N[0]=256;   h[0]=1.0/N[0];
			N[1]=256;   h[1]=1.0/N[1];    Ax[1] = 1.0;    At[1] = 1.0;
		}
		if (dim>2) { // z-axis
			N[0]=64;    h[0]=1.0/N[0];
			N[1]=64;    h[1]=1.0/N[1];
			N[2]=64;    h[2]=1.0/N[2];    Ax[2] = 1.0;    At[2] = 1.0;
		}
}

void generate(int dim, const char* filename)
{
	if (dim==1) {
		coefficients<1> vars;
		GRID1D initGrid(0,0,vars.N[0]);
		dx(initGrid,0)=vars.h[0];

		for (int i=0; i<nodes(initGrid); i++) {
			vector<int> x = MMSP::position(initGrid,n);
			initGrid(n) = MS_T(x, 0, vars.h, vars.Ax, vars.At);
		}

		output(initGrid,filename);
	}

	if (dim==2) {
		coefficients<2> vars;
		GRID2D initGrid(0,0,vars.N[0],0,vars.N[1]);
		dx(initGrid)=vars.h[0];
		dy(initGrid)=vars.h[1];

		for (int i=0; i<nodes(initGrid); i++) {
			vector<int> x = MMSP::position(initGrid,n);
			initGrid(n) = MS_T(x, 0, vars.h, vars.Ax, vars.At);
		}

		output(initGrid,filename);
	}

	if (dim==3) {
		coefficients<3> vars;
		GRID3D initGrid(0,0,vars.N[0],0,vars.N[1],0,vars.N[2]);
		dx(initGrid)=vars.h[0];
		dy(initGrid)=vars.h[1];
		dz(initGrid)=vars.h[2];

		for (int i=0; i<nodes(initGrid); i++) {
			vector<int> x = MMSP::position(initGrid,n);
			initGrid(n) = MS_T(x, 0, vars.h, vars.Ax, vars.At);
		}

		output(initGrid,filename);
	}
}

template <int dim, typename T> void update(grid<dim,T>& oldGrid, int steps)
{
	int rank=0;
    #ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif
    static double elapsed = 0.0;

	coefficients<dim> vars;

	ghostswap(oldGrid);

	grid<dim,T> newGrid(oldGrid);

	// March forward in time
	for (int step=0; step<steps; step++) {
		//if (rank==0)
		//	print_progress(step, steps);
		grid<dim,vector<T> > gradGrid(oldGrid,dim);
		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid,n);
			double kappa = MS_k(vars.Ck,oldGrid(n));
			gradGrid(x) = kappa*gradient(oldGrid,x);
		}

		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid,n);
			T temp = oldGrid(n);
			double prefactor = vars.dt/(vars.rho * MS_Cp(vars.Cc, temp));
			double divKgrad = divergence(gradGrid,x);
			double source = MS_Q(x, elapsed, vars.rho, vars.h, vars.Ax, vars.At, vars.Ck, vars.Cc);
			newGrid(x) = temp + prefactor*(divKgrad + Q);
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}

	elapsed += vars.dt;

	// Compute global error w.r.t. manufactured solution
	double error = 0.0;
	int npts = nodes(oldGrid);
	for (int n=0; n<npts; n++) {
		vector<int> x = position(oldGrid,n);
		double analytical = MS_T(x, elapsed, vars.h, vars.Ax, vars.At);
		double numerical = oldGrid(n);
		error += pow(numerical - analytical, 2.0);
	}
	#ifdef MPI_VERSION
	double myErr(error);
	int myPts(npts);
	MPI::COMM_WORLD.Allreduce(&myErr,&error,1,MPI_DOUBLE,MPI_SUM);
	MPI::COMM_WORLD.Allreduce(&myPts,&npts,1,MPI_INT,MPI_SUM);
	#endif
	error = sqrt(error/npts);
	std::cout<<elapsed<<'\t'<<error<<std::endl;

}


double MS_k (const vector<double>& Ck, double temp)
{
	return Ck[0] + Ck[1]*temp + Ck[2]*temp*temp;
}

double MS_Cp (const vector<double>& Cc, double temp)
{
	return Cc[0] + Cc[1]*temp + Cc[2]*temp*temp;
}

double MS_T (const vector<int>& xidx, double t, const vector<double>& h, const vector<double>& Ax, const vector<double>& At)
{
	// Analytical temperature
	vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.size(); d++)
		x[d] = h[d]*xidx[d];
	assert(Ax.size() == 3);
	assert(At.size() == 4);
	return (
		cos(Ax[0] * x[0] + At[0] * t) * cos(Ax[1] * x[1] + At[1] * t) * cos(Ax[2] * x[2] + At[2] * t) * cos(At[3] * t)
	);
}

vector<double> MS_GradT (const vector<int>& xidx, double t, const vector<double>& h, const vector<double>& Ax, const vector<double>& At)
{
	// Analytical temperature gradient
	vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.size(); d++)
		x[d] = h[d]*xidx[d];
	assert(Ax.size() == 3);
	assert(At.size() == 4);
	gradT_an[0] = -Ax[0] * cos(At[3] * t) * cos(Ax[1] * x[1] + At[1] * t) * cos(Ax[2] * x[2] + At[2] * t) * sin(Ax[0] * x[0] + At[0] * t);
	gradT_an[1] = -Ax[1] * cos(At[3] * t) * cos(Ax[0] * x[0] + At[0] * t) * cos(Ax[2] * x[2] + At[2] * t) * sin(Ax[1] * x[1] + At[1] * t);
	gradT_an[2] = -Ax[2] * cos(At[3] * t) * cos(Ax[0] * x[0] + At[0] * t) * cos(Ax[1] * x[1] + At[1] * t) * sin(Ax[2] * x[2] + At[2] * t);
}

double MS_Q (const vector<int>& xidx, double t, double rho, const vector<double>& h,
             const vector<double>& Ax, const vector<double>& At,
             const vector<double>& Ck, const vector<double>& Cc)
{
	vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.size(); d++)
		x[d] = h[d]*xidx[d];
	assert(At.size() == 4);
	assert(Ax.size() == 3);
	assert(Ck.size() == 3);
	assert(Cc.size() == 3);

	// Define the thermal source term. Transformed for MMSP accessors and restructured for clarity and linewidth
	// after https://github.com/manufactured-solutions/analytical/tree/master/heat_equation/C_code/
	// For best results, generate this yourself using a symbolic math package, e.g. SymPy, Mathematica, or Maple.
	// All this floating-point math in a single expression makes me nervous.
	double Q_T = 0.0;
	Q_T = - rho * Cc[0] * At[0] * sin(Ax[0] * x[0] + At[0] * t)
	                            * cos(Ax[1] * x[1] + At[1] * t)
	                            * cos(Ax[2] * x[2] + At[2] * t)
	                            * cos(At[3] * t)
	      - rho * Cc[0] * At[1] * cos(Ax[0] * x[0] + At[0] * t)
	                            * sin(Ax[1] * x[1] + At[1] * t)
	                            * cos(Ax[2] * x[2] + At[2] * t)
	                            * cos(At[3] * t)
	      - rho * Cc[0] * At[2] * cos(Ax[0] * x[0] + At[0] * t)
	                            * cos(Ax[1] * x[1] + At[1] * t)
	                            * sin(Ax[2] * x[2] + At[2] * t)
	                            * cos(At[3] * t)
	      - rho * Cc[0] * At[3] * cos(Ax[0] * x[0] + At[0] * t)
	                            * cos(Ax[1] * x[1] + At[1] * t)
	                            * cos(Ax[2] * x[2] + At[2] * t)
	                            * sin(At[3] * t)
	      - Ck[1] * Ax[0]*Ax[0] * pow(cos(Ax[1] * x[1] + At[1] * t), 2.0)
	                           * pow(cos(Ax[2] * x[2] + At[2] * t), 2.0)
	                           * pow(cos(At[3] * t), 2.0)
	      - Ck[1] * Ax[1]*Ax[1] * pow(cos(Ax[0] * x[0] + At[0] * t), 2.0)
	                           * pow(cos(Ax[2] * x[2] + At[2] * t), 2.0)
	                           * pow(cos(At[3] * t), 2.0)
	      - Ck[1] * Ax[2]*Ax[2] * pow(cos(Ax[0] * x[0] + At[0] * t), 2.0)
	                           * pow(cos(Ax[1] * x[1] + At[1] * t), 2.0)
	                           * pow(cos(At[3] * t), 2.0)
	      + 3.0 * Ck[2] * ( Ax[0]*Ax[0] + Ax[1]*Ax[1] + Ax[2]*Ax[2] )
	                   * pow(cos(Ax[0] * x[0] + At[0] * t), 3.0)
	                   * pow(cos(Ax[1] * x[1] + At[1] * t), 3.0)
	                   * pow(cos(Ax[2] * x[2] + At[2] * t), 3.0)
	                   * pow(cos(At[3] * t), 3.0)
	      + (
	        - rho * Cc[1] * At[0] * sin(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cc[1] * At[1] * cos(Ax[0] * x[0] + At[0] * t)
	                              * sin(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cc[1] * At[2] * cos(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * sin(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cc[1] * At[3] * cos(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * sin(At[3] * t)
	        + Ck[0] * Ax[0]*Ax[0] + Ck[0] * Ax[1]*Ax[1] + Ck[0] * Ax[2]*Ax[2]
	        - 2.0 * Ck[2] * Ax[0]*Ax[0] * pow(cos(Ax[1] * x[1] + At[1] * t), 2.0)
	                                     * pow(cos(Ax[2] * x[2] + At[2] * t), 2.0)
	                                     * pow(cos(At[3] * t), 2.0)
	        - 2.0 * Ck[2] * Ax[1]*Ax[1] * pow(cos(Ax[0] * x[0] + At[0] * t), 2.0)
	                                     * pow(cos(Ax[2] * x[2] + At[2] * t), 2.0)
	                                     * pow(cos(At[3] * t), 2.0)
	        - 2.0 * Ck[2] * Ax[2]*Ax[2] * pow(cos(Ax[0] * x[0] + At[0] * t), 2.0)
	                                     * pow(cos(Ax[1] * x[1] + At[1] * t), 2.0)
	                                     * pow(cos(At[3] * t), 2.0)
	        ) * cos(Ax[0] * x[0] + At[0] * t)
	          * cos(Ax[1] * x[1] + At[1] * t)
	          * cos(Ax[2] * x[2] + At[2] * t)
	          * cos(At[3] * t)
	      + (
	        - rho * Cc[2] * At[0] * sin(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cc[2] * At[1] * cos(Ax[0] * x[0] + At[0] * t)
	                              * sin(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cc[2] * At[2] * cos(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * sin(Ax[2] * x[2] + At[2] * t)
	                              * cos(At[3] * t)
	        - rho * Cc[2] * At[3] * cos(Ax[0] * x[0] + At[0] * t)
	                              * cos(Ax[1] * x[1] + At[1] * t)
	                              * cos(Ax[2] * x[2] + At[2] * t)
	                              * sin(At[3] * t)
	        + 2.0 * Ck[1] * Ax[0]*Ax[0]
	        + 2.0 * Ck[1] * Ax[1]*Ax[1]
	        + 2.0 * Ck[1] * Ax[2]*Ax[2]
	        ) * pow(cos(Ax[0] * x[0] + At[0] * t), 2.0)
	          * pow(cos(Ax[1] * x[1] + At[1] * t), 2.0)
	          * pow(cos(Ax[2] * x[2] + At[2] * t), 2.0)
	          * pow(cos(At[3] * t), 2.0);

	return Q_T;
}

} // namespace MMSP

#endif

#include"MMSP.main.hpp"
