// heat.cpp
// Algorithms for 2D and 3D heat model
// Questions/comments to trevor.keller@gmail.com (Trevor Keller)

#ifndef HEAT_UPDATE
#define HEAT_UPDATE
#include"MMSP.hpp"
#include<cmath>
#include<cassert>
#include"heat.hpp"
using std::cos;

namespace MMSP{

coefficients::coefficients(int dim) : // constructor
		// call initializing constructor on vector-valued members to enforce 3 spatial dimensions (otherwise, high risk of segfaulting)
		N(3,0),    // grid points
		h(3,0.0),  // spatial resolution
		Ax(3,0.0), // amplitudes in space
		At(4,0.0), // amplitudes in time
		Ck(3,0.0), // thermal conductivity polynomial coefficients
		Cc(3,0.0)  // heat capacity polynomial coefficients
{
		std::ifstream vfile("params.txt");

		// Set spatially-independent coefficients
		T0 = 1.0;
		dt = 0.000005;   // timestep
		noiseamp = 0.25;  // amplitude of random noise
		initscale = 1.625; // scaling factor for initial condition (should be "not too close to 1")
		rho = 1.0;
		At[3] = 3*M_PI/2.0; // fourth dimension is time -- cannot be zero, unless you implement an iterative solver for steady-state
		Ck[0] = 10.0;    //Ck[1] = 1.0;    Ck[2] = 0.01;
		Cc[0] = 10.0;    //Cc[1] = 2.0;    Cc[2] = 0.03;

		// Set coeffients for each dimension. If a dimension does not exist, leave its value zero.
		if (dim==1) {
			int L=512;
			if (vfile)
				vfile >> dt >> L;
			N[0]=L;      h[0]=1.0/N[0];    Ax[0] = 4.0*M_PI;    At[0] = 2.0*M_PI;
		} else if (dim==2) {
			int L=256;
			if (vfile)
				vfile >> dt >> L;
			N[0]=2*L;    h[0]=1.0/L;    Ax[0] = 4.0*M_PI;    At[0] = 2.0*M_PI;
			N[1]=L;      h[1]=1.0/L;    Ax[1] = 2.0*M_PI;    At[1] = 2.5*M_PI;
		} else if (dim==3) {
			int L=64;
			if (vfile)
				vfile >> dt >> L;
			N[0]=2*L;    h[0]=1.0/L;    Ax[0] = 4.0*M_PI;    At[0] = 2.0*M_PI;
			N[1]=L;      h[1]=1.0/L;    Ax[1] = 2.0*M_PI;    At[1] = 2.5*M_PI;
			N[2]=L/2;    h[2]=1.0/L;    Ax[2] = 1.0*M_PI;    At[2] = 3.0*M_PI;
		}
		if (vfile)
			vfile.close();
}

void generate(int dim, const char* filename)
{
	int rank=0;
    #ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif
	if (dim==1) {
		coefficients vars(1);
		GRID1D initGrid(0,0,vars.N[0]);
		dx(initGrid,0)=vars.h[0];

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = MMSP::position(initGrid,n);
			double noise = vars.noiseamp*double(rand()-(RAND_MAX/2))/RAND_MAX;
			initGrid(n) = vars.initscale*MS_T(x, 0, vars.T0, vars.h, vars.Ax, vars.At) + noise;
		}
		double prefactor = (vars.dt * MS_k(vars.Ck, initGrid(nodes(initGrid)/2))) /
		                   (vars.rho*MS_Cp(vars.Cc, initGrid(nodes(initGrid)/2)));
		if (rank==0)
			printf("CFL condition is %g. Take %d steps per unit time.\n",prefactor/pow(vars.h[0],2),int(1.0/vars.dt));

		output(initGrid,filename);
	}

	if (dim==2) {
		coefficients vars(2);
		GRID2D initGrid(0,0,vars.N[0],0,vars.N[1]);
		dx(initGrid)=vars.h[0];
		dy(initGrid)=vars.h[1];

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = MMSP::position(initGrid,n);
			double noise = vars.noiseamp*double(rand()-(RAND_MAX/2))/RAND_MAX;
			initGrid(n) = vars.initscale*MS_T(x, 0, vars.T0, vars.h, vars.Ax, vars.At) + noise;
		}
		double prefactor = (vars.dt * MS_k(vars.Ck, initGrid(nodes(initGrid)/2))) /
		                   (vars.rho*MS_Cp(vars.Cc, initGrid(nodes(initGrid)/2)));
		if (rank==0)
			printf("CFL condition is %g. Take %d steps per unit time.\n",prefactor*(1.0/pow(vars.h[0],2)+1.0/pow(vars.h[1],2)),int(1.0/vars.dt));

		output(initGrid,filename);
	}

	if (dim==3) {
		coefficients vars(3);
		GRID3D initGrid(0,0,vars.N[0],0,vars.N[1],0,vars.N[2]);
		dx(initGrid)=vars.h[0];
		dy(initGrid)=vars.h[1];
		dz(initGrid)=vars.h[2];

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = MMSP::position(initGrid,n);
			double noise = vars.noiseamp*double(rand()-(RAND_MAX/2))/RAND_MAX;
			initGrid(n) = vars.initscale*MS_T(x, 0, vars.T0, vars.h, vars.Ax, vars.At) + noise;
		}
		double prefactor = (vars.dt * MS_k(vars.Ck, initGrid(nodes(initGrid)/2)))/
		                   (vars.rho*MS_Cp(vars.Cc, initGrid(nodes(initGrid)/2)));
		if (rank==0)
			printf("CFL condition is %g. Take %d steps per unit time.\n",prefactor*(1.0/pow(vars.h[0],2)+1.0/pow(vars.h[1],2)+1.0/pow(vars.h[2],2)),int(1.0/vars.dt));

		output(initGrid,filename);
	}
}

template <int dim, typename T> void update(grid<dim,T>& oldGrid, int steps)
{
	int rank=0;
    #ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif

	// Setup variables and grids
    static double elapsed = 0.0;
	coefficients vars(dim);
	ghostswap(oldGrid);
	grid<dim,T> newGrid(oldGrid);

	if (rank==0 && elapsed<vars.dt)
		std::cout<<"time\terror\n";

	bool numerical=true;
	if (numerical) { // numerical solution
		for (int step=0; step<steps; step++) {
			// March forward in time
			elapsed += vars.dt;

			for (int n=0; n<nodes(oldGrid); n++) {
				vector<int> x = position(oldGrid,n);
				T temp = oldGrid(n);
				vector<T> gradT = gradient(oldGrid,x);
				// div.k(grad T) = grad(k).grad(T) + k.lap(T) = k'.grad(T).grad(T) + k.laplacian(T)
				double divkgradT = MS_dkdT(vars.Ck, temp)*(gradT*gradT) + MS_k(vars.Ck, temp)*laplacian(oldGrid, x);
				double prefactor = vars.dt/(vars.rho * MS_Cp(vars.Cc, temp));
				double source = MS_Q<dim>(x, elapsed, vars.T0, vars.rho, vars.h, vars.Ax, vars.At, vars.Ck, vars.Cc);
				newGrid(n) = temp + prefactor*(divkgradT + source);
			}
			swap(oldGrid,newGrid);
			ghostswap(oldGrid);
		}
	} else { // analytical solution
		// Jump forward in time
		elapsed += steps*vars.dt;
		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid,n);
			newGrid(n) = MS_T(x, elapsed, vars.T0, vars.h, vars.Ax, vars.At);
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}

	// Compute global error w.r.t. manufactured solution
	double error = 0.0;
	for (int n=0; n<nodes(oldGrid); n++) {
		vector<int> x = position(oldGrid,n);
		double analytical = MS_T(x, elapsed, vars.T0, vars.h, vars.Ax, vars.At);
		double numerical = oldGrid(n);
		error += pow(numerical - analytical, 2.0);
	}
	double npts(nodes(oldGrid));
	#ifdef MPI_VERSION
	double myErr(error);
	double myPts(npts);
	MPI::COMM_WORLD.Allreduce(&myErr,&error,1,MPI_DOUBLE,MPI_SUM);
	MPI::COMM_WORLD.Allreduce(&myPts,&npts,1,MPI_DOUBLE,MPI_SUM);
	#endif
	error = sqrt(error/npts);
	if (rank==0)
		std::cout<<elapsed<<'\t'<<error<<std::endl;

}


} // namespace MMSP

double MS_k (const MMSP::vector<double>& Ck, const double& temp)
{
	return Ck[0] + Ck[1]*temp + Ck[2]*temp*temp;
}

double MS_dkdT (const MMSP::vector<double>& Ck, const double& temp)
{
	return Ck[1] + 2.0*Ck[2]*temp;
}

double MS_Cp (const MMSP::vector<double>& Cc, const double& temp)
{
	return Cc[0] + Cc[1]*temp + Cc[2]*temp*temp;
}

double MS_T (const MMSP::vector<int>& xidx, const double& t, const double& T0,
             const MMSP::vector<double>& h, const MMSP::vector<double>& Ax, const MMSP::vector<double>& At)
{
	// Analytical temperature
	MMSP::vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.length(); d++)
		x[d] = h[d]*xidx[d];
	assert(Ax.length() == 3);
	assert(At.length() == 4);
	return (
		T0*(1.0 + cos(Ax[0]*x[0] + At[0]*t) * cos(Ax[1]*x[1] + At[1]*t) * cos(Ax[2]*x[2] + At[2]*t) * cos(At[3]*t))
	);
}

template<>
double MS_Q<1>(const MMSP::vector<int>& xidx, const double& t, const double& T0, const double& rho, const MMSP::vector<double>& h,
             const MMSP::vector<double>& Ax, const MMSP::vector<double>& At,
             const MMSP::vector<double>& Ck, const MMSP::vector<double>& Cc)
{
	MMSP::vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.length(); d++)
		x[d] = h[d]*xidx[d];
	assert(At.length() == 4);
	assert(Ax.length() == 3);
	assert(Ck.length() == 3);
	assert(Cc.length() == 3);

	// Define the thermal source term. Transformed for MMSP accessors and restructured for clarity and linewidth
	// after https://github.com/manufactured-solutions/analytical/tree/master/heat_equation/C_code/
	// For best results, generate this yourself using a symbolic math package, e.g. SymPy, Mathematica, or Maple.
	// All this floating-point math in a single expression makes me nervous.
	double Q_T = 0.0;
	Q_T = T0*(pow(Ax[0],2)*cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*(Ck[0] + Ck[1]*T0*(1 + cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])) + 
        Ck[2]*pow(T0 + T0*cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0]),2)) - 
     pow(Ax[0],2)*T0*pow(cos(At[3]*t),2)*(Ck[1] + 2*Ck[2]*T0 + Ck[2]*T0*cos(At[0]*t - At[3]*t + Ax[0]*x[0]) + Ck[2]*T0*cos(At[0]*t + At[3]*t + Ax[0]*x[0]))*
      pow(sin(At[0]*t + Ax[0]*x[0]),2) + rho*(Cc[0] + Cc[1]*T0*(1 + cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])) + 
        Cc[2]*pow(T0 + T0*cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0]),2))*
      (-(At[3]*cos(At[0]*t + Ax[0]*x[0])*sin(At[3]*t)) - At[0]*cos(At[3]*t)*sin(At[0]*t + Ax[0]*x[0])));
	return Q_T;
}

template<>
double MS_Q<2>(const MMSP::vector<int>& xidx, const double& t, const double& T0, const double& rho, const MMSP::vector<double>& h,
             const MMSP::vector<double>& Ax, const MMSP::vector<double>& At,
             const MMSP::vector<double>& Ck, const MMSP::vector<double>& Cc)
{
	MMSP::vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.length(); d++)
		x[d] = h[d]*xidx[d];
	assert(At.length() == 4);
	assert(Ax.length() == 3);
	assert(Ck.length() == 3);
	assert(Cc.length() == 3);

	// Define the thermal source term. Transformed for MMSP accessors and restructured for clarity and linewidth
	// after https://github.com/manufactured-solutions/analytical/tree/master/heat_equation/C_code/
	// For best results, generate this yourself using a symbolic math package, e.g. SymPy, Mathematica, or Maple.
	// All this floating-point math in a single expression makes me nervous.
	double Q_T = 0.0;
	Q_T = T0*(pow(Ax[0],2)*cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*cos(At[1]*t + Ax[1]*x[1])*
      (Ck[0] + Ck[1]*T0*(1 + cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*cos(At[1]*t + Ax[1]*x[1])) + 
        Ck[2]*pow(T0 + T0*cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*cos(At[1]*t + Ax[1]*x[1]),2)) + 
     pow(Ax[1],2)*cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*cos(At[1]*t + Ax[1]*x[1])*
      (Ck[0] + Ck[1]*T0*(1 + cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*cos(At[1]*t + Ax[1]*x[1])) + 
        Ck[2]*pow(T0 + T0*cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*cos(At[1]*t + Ax[1]*x[1]),2)) - 
     pow(Ax[0],2)*T0*pow(cos(At[3]*t),2)*pow(cos(At[1]*t + Ax[1]*x[1]),2)*
      (Ck[1] + 2*Ck[2]*T0 + 2*Ck[2]*T0*cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*cos(At[1]*t + Ax[1]*x[1]))*pow(sin(At[0]*t + Ax[0]*x[0]),2) - 
     pow(Ax[1],2)*T0*pow(cos(At[3]*t),2)*pow(cos(At[0]*t + Ax[0]*x[0]),2)*
      (Ck[1] + 2*Ck[2]*T0 + 2*Ck[2]*T0*cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*cos(At[1]*t + Ax[1]*x[1]))*pow(sin(At[1]*t + Ax[1]*x[1]),2) + 
     rho*(Cc[0] + Cc[1]*T0*(1 + cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*cos(At[1]*t + Ax[1]*x[1])) + 
        Cc[2]*pow(T0 + T0*cos(At[3]*t)*cos(At[0]*t + Ax[0]*x[0])*cos(At[1]*t + Ax[1]*x[1]),2))*
      (-(At[0]*cos(At[3]*t)*cos(At[1]*t + Ax[1]*x[1])*sin(At[0]*t + Ax[0]*x[0])) - 
        cos(At[0]*t + Ax[0]*x[0])*(At[3]*cos(At[1]*t + Ax[1]*x[1])*sin(At[3]*t) + At[1]*cos(At[3]*t)*sin(At[1]*t + Ax[1]*x[1]))));
	return Q_T;
}

template<>
double MS_Q<3>(const MMSP::vector<int>& xidx, const double& t, const double& T0, const double& rho, const MMSP::vector<double>& h,
             const MMSP::vector<double>& Ax, const MMSP::vector<double>& At,
             const MMSP::vector<double>& Ck, const MMSP::vector<double>& Cc)
{
	MMSP::vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.length(); d++)
		x[d] = h[d]*xidx[d];
	assert(At.length() == 4);
	assert(Ax.length() == 3);
	assert(Ck.length() == 3);
	assert(Cc.length() == 3);

	// Define the thermal source term. Transformed for MMSP accessors and restructured for clarity and linewidth
	// after https://github.com/manufactured-solutions/analytical/tree/master/heat_equation/C_code/
	// For best results, generate this yourself using a symbolic math package, e.g. SymPy, Mathematica, or Maple.
	// All this floating-point math in a single expression makes me nervous.
	double Q_T = 0.0;
	Q_T = T0*( pow(cos(Ax[1]*x[1] + At[1]*t),2.0)
	          *pow(cos(Ax[2]*x[2] + At[2]*t),2.0)
              *pow(cos(t*At[3]),2.0)
              *pow(sin(Ax[0]*x[0] + At[0]*t),2.0)
              *pow(Ax[0],2.0)*T0*
              (-Ck[1] - 2.0*(1.0 + cos(Ax[0]*x[0] + At[0]*t)
                                  *cos(Ax[1]*x[1] + At[1]*t)
                                  *cos(Ax[2]*x[2] + At[2]*t)
                                  *cos(t*At[3])
                            )*Ck[2]*T0
              )
              +pow(cos(Ax[0]*x[0] + At[0]*t),2.0)
              *pow(cos(Ax[2]*x[2] + At[2]*t),2.0)
              *pow(cos(t*At[3]),2.0)
              *pow(sin(Ax[1]*x[1] + At[1]*t),2.0)
              *pow(Ax[1],2.0)*T0*
              (-Ck[1] - 2.0*(1.0 + cos(Ax[0]*x[0] + At[0]*t)
                                   *cos(Ax[1]*x[1] + At[1]*t)
                                   *cos(Ax[2]*x[2] + At[2]*t)
                                   *cos(t*At[3])
                             )*Ck[2]*T0
              )
              +pow(cos(Ax[0]*x[0] + At[0]*t),2.0)
              *pow(cos(Ax[1]*x[1] + At[1]*t),2.0)
              *pow(cos(t*At[3]),2.0)
              *pow(sin(Ax[2]*x[2] + At[2]*t),2.0)
              *pow(Ax[2],2.0)*T0*
              (-Ck[1] - 2.0*(1.0 + cos(Ax[0]*x[0] + At[0]*t)
                                  *cos(Ax[1]*x[1] + At[1]*t)
                                  *cos(Ax[2]*x[2] + At[2]*t)
                                  *cos(t*At[3])
                            )*Ck[2]*T0
             )
             + rho*(
                    -( cos(Ax[1]*x[1] + At[1]*t)
                      *cos(Ax[2]*x[2] + At[2]*t)
                      *cos(t*At[3])
                      *sin(Ax[0]*x[0] + At[0]*t)
                      *At[0]
                     )
                    - cos(Ax[0]*x[0] + At[0]*t)
                     *( cos(Ax[2]*x[2] + At[2]*t)
                       *cos(t*At[3])
                       *sin(Ax[1]*x[1] + At[1]*t)
                       *At[1] + cos(Ax[1]*x[1] + At[1]*t)
                                *( cos(t*At[3])
                                  *sin(Ax[2]*x[2] + At[2]*t)
                                  *At[2]
                                  + cos(Ax[2]*x[2] + At[2]*t)
                                   *sin(t*At[3])
                                   *At[3]
                                 )
                      )
                   ) *
                   (Cc[0] + (1.0 + cos(Ax[0]*x[0] + At[0]*t)
                                  *cos(Ax[1]*x[1] + At[1]*t)
                                  *cos(Ax[2]*x[2] + At[2]*t)
                                  *cos(t*At[3])
                            ) *T0*
                   (Cc[1] + (1.0 + cos(Ax[0]*x[0] + At[0]*t)
                                  *cos(Ax[1]*x[1] + At[1]*t)
                                  *cos(Ax[2]*x[2] + At[2]*t)
                                  *cos(t*At[3])
                             )*Cc[2]*T0
                   )
             ) + cos(Ax[0]*x[0] + At[0]*t)
                *cos(Ax[1]*x[1] + At[1]*t)
                *cos(Ax[2]*x[2] + At[2]*t)
                *cos(t*At[3])
                *pow(Ax[0],2.0)*
                (Ck[0] + (1.0 + cos(Ax[0]*x[0] + At[0]*t)
                                *cos(Ax[1]*x[1] + At[1]*t)
                                *cos(Ax[2]*x[2] + At[2]*t)
                                *cos(t*At[3])
                          )*T0*
                (Ck[1] + (1.0 + cos(Ax[0]*x[0] + At[0]*t)
                                      *cos(Ax[1]*x[1] + At[1]*t)
                                      *cos(Ax[2]*x[2] + At[2]*t)
                                      *cos(t*At[3])
                          )*Ck[2]*T0
                )
             ) + cos(Ax[0]*x[0] + At[0]*t)
                *cos(Ax[1]*x[1] + At[1]*t)
                *cos(Ax[2]*x[2] + At[2]*t)
                *cos(t*At[3])
                *pow(Ax[1],2.0)*
                (Ck[0] + (1.0 + cos(Ax[0]*x[0] + At[0]*t)
                               *cos(Ax[1]*x[1] + At[1]*t)
                               *cos(Ax[2]*x[2] + At[2]*t)
                               *cos(t*At[3])
                         ) *T0*
                (Ck[1] + (1.0 + cos(Ax[0]*x[0] + At[0]*t)
                               *cos(Ax[1]*x[1] + At[1]*t)
                               *cos(Ax[2]*x[2] + At[2]*t)
                               *cos(t*At[3])
                         )*Ck[2]*T0
                )
             ) + cos(Ax[0]*x[0] + At[0]*t)
                *cos(Ax[1]*x[1] + At[1]*t)
                *cos(Ax[2]*x[2] + At[2]*t)
                *cos(t*At[3])
                *pow(Ax[2],2.0)*
                (Ck[0] + (1.0 + cos(Ax[0]*x[0] + At[0]*t)
                               *cos(Ax[1]*x[1] + At[1]*t)
                               *cos(Ax[2]*x[2] + At[2]*t)
                               *cos(t*At[3])
                         )*T0*
                (Ck[1] + (1.0 + cos(Ax[0]*x[0] + At[0]*t)
                               *cos(Ax[1]*x[1] + At[1]*t)
                               *cos(Ax[2]*x[2] + At[2]*t)
                               *cos(t*At[3])
                         )*Ck[2]*T0
                )
             )
           );

	return Q_T;
}

double MS_T_old (const MMSP::vector<int>& xidx, double t, const double& T0,
             const MMSP::vector<double>& h, const MMSP::vector<double>& Ax, const MMSP::vector<double>& At)
{
	// Analytical temperature
	MMSP::vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.length(); d++)
		x[d] = h[d]*xidx[d];
	assert(Ax.length() == 3);
	assert(At.length() == 4);
	return (
		cos(Ax[0]*x[0] + At[0]*t) * cos(Ax[1]*x[1] + At[1]*t) * cos(Ax[2]*x[2] + At[2]*t) * cos(At[3]*t)
	);
}

double MS_Q_old (const MMSP::vector<int>& xidx, double t, const double& T0, double rho, const MMSP::vector<double>& h,
             const MMSP::vector<double>& Ax, const MMSP::vector<double>& At,
             const MMSP::vector<double>& Ck, const MMSP::vector<double>& Cc)
{
	MMSP::vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.length(); d++)
		x[d] = h[d]*xidx[d];
	assert(At.length() == 4);
	assert(Ax.length() == 3);
	assert(Ck.length() == 3);
	assert(Cc.length() == 3);

	// Define the thermal source term. Transformed for MMSP accessors and restructured for clarity and linewidth
	// after https://github.com/manufactured-solutions/analytical/tree/master/heat_equation/C_code/
	// For best results, generate this yourself using a symbolic math package, e.g. SymPy, Mathematica, or Maple.
	// All this floating-point math in a single expression makes me nervous.
	double Q_T = 0.0;
	Q_T = - rho * Cc[0] * At[0] * sin(Ax[0]*x[0] + At[0]*t)
	                            * cos(Ax[1]*x[1] + At[1]*t)
	                            * cos(Ax[2]*x[2] + At[2]*t)
	                            * cos(At[3]*t)
	      - rho * Cc[0] * At[1] * cos(Ax[0]*x[0] + At[0]*t)
	                            * sin(Ax[1]*x[1] + At[1]*t)
	                            * cos(Ax[2]*x[2] + At[2]*t)
	                            * cos(At[3]*t)
	      - rho * Cc[0] * At[2] * cos(Ax[0]*x[0] + At[0]*t)
	                            * cos(Ax[1]*x[1] + At[1]*t)
	                            * sin(Ax[2]*x[2] + At[2]*t)
	                            * cos(At[3]*t)
	      - rho * Cc[0] * At[3] * cos(Ax[0]*x[0] + At[0]*t)
	                            * cos(Ax[1]*x[1] + At[1]*t)
	                            * cos(Ax[2]*x[2] + At[2]*t)
	                            * sin(At[3]*t)
	      - Ck[1] * Ax[0]*Ax[0] * pow(cos(Ax[1]*x[1] + At[1]*t), 2.0)
	                            * pow(cos(Ax[2]*x[2] + At[2]*t), 2.0)
	                            * pow(cos(At[3]*t), 2.0)
	      - Ck[1] * Ax[1]*Ax[1] * pow(cos(Ax[0]*x[0] + At[0]*t), 2.0)
	                            * pow(cos(Ax[2]*x[2] + At[2]*t), 2.0)
	                            * pow(cos(At[3]*t), 2.0)
	      - Ck[1] * Ax[2]*Ax[2] * pow(cos(Ax[0]*x[0] + At[0]*t), 2.0)
	                            * pow(cos(Ax[1]*x[1] + At[1]*t), 2.0)
	                            * pow(cos(At[3]*t), 2.0)
	      + 3.0 * Ck[2] * ( Ax[0]*Ax[0] + Ax[1]*Ax[1] + Ax[2]*Ax[2] )
	                    * pow(cos(Ax[0]*x[0] + At[0]*t), 3.0)
	                    * pow(cos(Ax[1]*x[1] + At[1]*t), 3.0)
	                    * pow(cos(Ax[2]*x[2] + At[2]*t), 3.0)
	                    * pow(cos(At[3]*t), 3.0)
	      + (
	        - rho * Cc[1] * At[0] * sin(Ax[0]*x[0] + At[0]*t)
	                              * cos(Ax[1]*x[1] + At[1]*t)
	                              * cos(Ax[2]*x[2] + At[2]*t)
	                              * cos(At[3]*t)
	        - rho * Cc[1] * At[1] * cos(Ax[0]*x[0] + At[0]*t)
	                              * sin(Ax[1]*x[1] + At[1]*t)
	                              * cos(Ax[2]*x[2] + At[2]*t)
	                              * cos(At[3]*t)
	        - rho * Cc[1] * At[2] * cos(Ax[0]*x[0] + At[0]*t)
	                              * cos(Ax[1]*x[1] + At[1]*t)
	                              * sin(Ax[2]*x[2] + At[2]*t)
	                              * cos(At[3]*t)
	        - rho * Cc[1] * At[3] * cos(Ax[0]*x[0] + At[0]*t)
	                              * cos(Ax[1]*x[1] + At[1]*t)
	                              * cos(Ax[2]*x[2] + At[2]*t)
	                              * sin(At[3]*t)
	        + Ck[0] * Ax[0]*Ax[0] + Ck[0] * Ax[1]*Ax[1] + Ck[0] * Ax[2]*Ax[2]
	        - 2.0 * Ck[2] * Ax[0]*Ax[0] * pow(cos(Ax[1]*x[1] + At[1]*t), 2.0)
	                                    * pow(cos(Ax[2]*x[2] + At[2]*t), 2.0)
	                                    * pow(cos(At[3]*t), 2.0)
	        - 2.0 * Ck[2] * Ax[1]*Ax[1] * pow(cos(Ax[0]*x[0] + At[0]*t), 2.0)
	                                    * pow(cos(Ax[2]*x[2] + At[2]*t), 2.0)
	                                    * pow(cos(At[3]*t), 2.0)
	        - 2.0 * Ck[2] * Ax[2]*Ax[2] * pow(cos(Ax[0]*x[0] + At[0]*t), 2.0)
	                                    * pow(cos(Ax[1]*x[1] + At[1]*t), 2.0)
	                                    * pow(cos(At[3]*t), 2.0)
	        ) * cos(Ax[0]*x[0] + At[0]*t)
	          * cos(Ax[1]*x[1] + At[1]*t)
	          * cos(Ax[2]*x[2] + At[2]*t)
	          * cos(At[3]*t)
	      + (
	        - rho * Cc[2] * At[0] * sin(Ax[0]*x[0] + At[0]*t)
	                              * cos(Ax[1]*x[1] + At[1]*t)
	                              * cos(Ax[2]*x[2] + At[2]*t)
	                              * cos(At[3]*t)
	        - rho * Cc[2] * At[1] * cos(Ax[0]*x[0] + At[0]*t)
	                              * sin(Ax[1]*x[1] + At[1]*t)
	                              * cos(Ax[2]*x[2] + At[2]*t)
	                              * cos(At[3]*t)
	        - rho * Cc[2] * At[2] * cos(Ax[0]*x[0] + At[0]*t)
	                              * cos(Ax[1]*x[1] + At[1]*t)
	                              * sin(Ax[2]*x[2] + At[2]*t)
	                              * cos(At[3]*t)
	        - rho * Cc[2] * At[3] * cos(Ax[0]*x[0] + At[0]*t)
	                              * cos(Ax[1]*x[1] + At[1]*t)
	                              * cos(Ax[2]*x[2] + At[2]*t)
	                              * sin(At[3]*t)
	        + 2.0 * Ck[1] * Ax[0]*Ax[0]
	        + 2.0 * Ck[1] * Ax[1]*Ax[1]
	        + 2.0 * Ck[1] * Ax[2]*Ax[2]
	        ) * pow(cos(Ax[0]*x[0] + At[0]*t), 2.0)
	          * pow(cos(Ax[1]*x[1] + At[1]*t), 2.0)
	          * pow(cos(Ax[2]*x[2] + At[2]*t), 2.0)
	          * pow(cos(At[3]*t), 2.0);

	return Q_T;
}

#endif

#include"MMSP.main.hpp"
