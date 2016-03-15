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
		AX(3,0.0), // amplitudes in space
		AT(4,0.0), // amplitudes in time
		Ck(3,0.0), // thermal conductivity polynomial coefficients
		Cc(3,0.0)  // heat capacity polynomial coefficients
{
		std::ifstream vfile("params.txt");

		// Set spatially-independent coefficients
		T0 = 1.0;
		dt = 0.0005;   // timestep
		noiseamp = 0.0;  // amplitude of random noise
		initscale = 1.0; // scaling factor for initial condition (should be "not too close to 1")
		rho = 1.0;
		AT[3] = 2.0*M_PI; // fourth dimension is time -- cannot be zero, unless you implement an iterative solver for steady-state
		Ck[0] = 3.0;    Ck[1] = 0.2;    Ck[2] = 0.001;
		Cc[0] = 1.0;    Cc[1] = 0.2;    Cc[2] = 0.003;

		// Set coeffients for each dimension. If a dimension does not exist, leave its value zero.
		if (dim==1) {
			int L=512;
			if (vfile)
				vfile >> dt >> L;
			N[0]=L;      h[0]=1.0/L;    AX[0] = 4.0*M_PI;    AT[0] = 2.0*M_PI;
		} else if (dim==2) {
			int L=256;
			if (vfile)
				vfile >> dt >> L;
			N[0]=2*L;    h[0]=1.0/L;    AX[0] = 4.0*M_PI;    AT[0] = 2.0*M_PI;
			N[1]=L;      h[1]=1.0/L;    AX[1] = 2.0*M_PI;    AT[1] = 2.5*M_PI;
		} else if (dim==3) {
			int L=64;
			if (vfile)
				vfile >> dt >> L;
			N[0]=2*L;    h[0]=1.0/L;    AX[0] = 4.0*M_PI;    AT[0] = 2.0*M_PI;
			N[1]=L;      h[1]=1.0/L;    AX[1] = 2.0*M_PI;    AT[1] = 2.5*M_PI;
			N[2]=L/2;    h[2]=1.0/L;    AX[2] = 1.0*M_PI;    AT[2] = 3.0*M_PI;
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
			double noise = vars.noiseamp*double(rand())/RAND_MAX;
			initGrid(n) = vars.initscale*MS_T(x, 0, vars.T0, vars.h, vars.AX, vars.AT) + noise;
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
			double noise = vars.noiseamp*double(rand())/RAND_MAX;
			initGrid(n) = vars.initscale*MS_T(x, 0, vars.T0, vars.h, vars.AX, vars.AT) + noise;
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
			double noise = vars.noiseamp*double(rand())/RAND_MAX;
			initGrid(n) = vars.initscale*MS_T(x, 0, vars.T0, vars.h, vars.AX, vars.AT) + noise;
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

	bool numerical=true; // set "false" to run analytical solution, only
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
				double source = MS_Q<dim>(x, elapsed, vars.T0, vars.rho, vars.h, vars.AX, vars.AT, vars.Ck, vars.Cc);
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
			newGrid(n) = MS_T(x, elapsed, vars.T0, vars.h, vars.AX, vars.AT);
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}

	// Compute global error w.r.t. manufactured solution
	double error = 0.0;
	for (int n=0; n<nodes(oldGrid); n++) {
		vector<int> x = position(oldGrid,n);
		double analytical = MS_T(x, elapsed, vars.T0, vars.h, vars.AX, vars.AT);
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
             const MMSP::vector<double>& h, const MMSP::vector<double>& AX, const MMSP::vector<double>& AT)
{
	// Analytical temperature
	MMSP::vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.length(); d++)
		x[d] = h[d]*xidx[d];
	assert(AX.length() == 3);
	assert(AT.length() == 4);

	return (
		T0 + cos(AX[0]*x[0] + AT[0]*t) * cos(AX[1]*x[1] + AT[1]*t) * cos(AX[2]*x[2] + AT[2]*t) * cos(AT[3]*t)
	);
}

template<>
double MS_Q<1>(const MMSP::vector<int>& xidx, const double& t, const double& T0, const double& rho, const MMSP::vector<double>& h,
             const MMSP::vector<double>& AX, const MMSP::vector<double>& AT,
             const MMSP::vector<double>& Ck, const MMSP::vector<double>& Cc)
{
	MMSP::vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.length(); d++)
		x[d] = h[d]*xidx[d];
	assert(AT.length() == 4);
	assert(AX.length() == 3);
	assert(Ck.length() == 3);
	assert(Cc.length() == 3);
	double Ax(AX[0]);
	double At(AT[0]), Dt(AT[3]);
	double k0(Ck[0]), k1(Ck[1]), k2(Ck[2]);
	double c0(Cc[0]), c1(Cc[1]), c2(Cc[2]);

	// Define the thermal source term. To change, edit the manufactured solution in manufactured_sympy.py
	// execute using Python, then paste the result in here.
	double Q_T = 0.0;
	Q_T = pow(Ax, 2)*(k0 + k1*(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)) + k2*pow(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t), 2))
          *cos(Dt*t)*cos(x[0]*Ax + At*t) + Ax*(-Ax*k1*sin(x[0]*Ax + At*t)*cos(Dt*t) - 2*Ax*k2*(T0 + cos(Dt*t)
          *cos(x[0]*Ax + At*t))*sin(x[0]*Ax + At*t)*cos(Dt*t))*sin(x[0]*Ax + At*t)*cos(Dt*t)
          + rho*(-At*sin(x[0]*Ax + At*t)*cos(Dt*t) - Dt*sin(Dt*t)*cos(x[0]*Ax + At*t))*(c0 + c1*(T0 + cos(Dt*t)
          *cos(x[0]*Ax + At*t)) + c2*pow(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t), 2));

	return Q_T;
}

template<>
double MS_Q<2>(const MMSP::vector<int>& xidx, const double& t, const double& T0, const double& rho, const MMSP::vector<double>& h,
             const MMSP::vector<double>& AX, const MMSP::vector<double>& AT,
             const MMSP::vector<double>& Ck, const MMSP::vector<double>& Cc)
{
	MMSP::vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.length(); d++)
		x[d] = h[d]*xidx[d];
	assert(AT.length() == 4);
	assert(AX.length() == 3);
	assert(Ck.length() == 3);
	assert(Cc.length() == 3);
	double Ax(AX[0]), By(AX[1]);
	double At(AT[0]), Bt(AT[1]), Dt(AT[3]);
	double k0(Ck[0]), k1(Ck[1]), k2(Ck[2]);
	double c0(Cc[0]), c1(Cc[1]), c2(Cc[2]);

	// Define the thermal source term. To change, edit the manufactured solution in manufactured_sympy.py
	// execute using Python, then paste the result in here.
	double Q_T = 0.0;
	Q_T = pow(Ax, 2)*(k0 + k1*(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t))
          + k2*pow(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t), 2))*cos(Dt*t)*cos(x[0]*Ax + At*t)
          *cos(x[1]*By + Bt*t) + Ax*(-Ax*k1*sin(x[0]*Ax + At*t)*cos(Dt*t)*cos(x[1]*By + Bt*t) - 2*Ax*k2*(T0 + cos(Dt*t)
          *cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t))*sin(x[0]*Ax + At*t)*cos(Dt*t)*cos(x[1]*By + Bt*t))*sin(x[0]*Ax + At*t)
          *cos(Dt*t)*cos(x[1]*By + Bt*t) + pow(By, 2)*(k0 + k1*(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t))
          + k2*pow(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t), 2))*cos(Dt*t)*cos(x[0]*Ax + At*t)
          *cos(x[1]*By + Bt*t) + By*(-By*k1*sin(x[1]*By + Bt*t)*cos(Dt*t)*cos(x[0]*Ax + At*t) - 2*By*k2*(T0 + cos(Dt*t)
          *cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t))*sin(x[1]*By + Bt*t)*cos(Dt*t)*cos(x[0]*Ax + At*t))*sin(x[1]*By + Bt*t)
          *cos(Dt*t)*cos(x[0]*Ax + At*t) + rho*(c0 + c1*(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t)) + c2
          *pow(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t), 2))*(-At*sin(x[0]*Ax + At*t)*cos(Dt*t)
          *cos(x[1]*By + Bt*t) - Bt*sin(x[1]*By + Bt*t)*cos(Dt*t)*cos(x[0]*Ax + At*t) - Dt*sin(Dt*t)*cos(x[0]*Ax + At*t)
          *cos(x[1]*By + Bt*t));

	return Q_T;
}

template<>
double MS_Q<3>(const MMSP::vector<int>& xidx, const double& t, const double& T0, const double& rho, const MMSP::vector<double>& h,
             const MMSP::vector<double>& AX, const MMSP::vector<double>& AT,
             const MMSP::vector<double>& Ck, const MMSP::vector<double>& Cc)
{
	MMSP::vector<double> x(3,0.0); // most general case is 3D, zero by default to handle 1D and 2D naturally
	for (int d=0; d<xidx.length(); d++)
		x[d] = h[d]*xidx[d];
	assert(AT.length() == 4);
	assert(AX.length() == 3);
	assert(Ck.length() == 3);
	assert(Cc.length() == 3);
	double Ax(AX[0]), By(AX[1]), Cz(AX[2]);
	double At(AT[0]), Bt(AT[1]), Ct(AT[2]), Dt(AT[3]);
	double k0(Ck[0]), k1(Ck[1]), k2(Ck[2]);
	double c0(Cc[0]), c1(Cc[1]), c2(Cc[2]);

	// Define the thermal source term. To change, edit the manufactured solution in manufactured_sympy.py
	// execute using Python, then paste the result in here.
	double Q_T = 0.0;
	Q_T = pow(Ax, 2)*(k0 + k1*(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t))
          + k2*pow(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t), 2))*cos(Dt*t)*cos(x[0]*Ax + At*t)
          *cos(x[1]*By + Bt*t) + Ax*(-Ax*k1*sin(x[0]*Ax + At*t)*cos(Dt*t)*cos(x[1]*By + Bt*t) - 2*Ax*k2*(T0 + cos(Dt*t)
          *cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t))*sin(x[0]*Ax + At*t)*cos(Dt*t)*cos(x[1]*By + Bt*t))*sin(x[0]*Ax + At*t)
          *cos(Dt*t)*cos(x[1]*By + Bt*t) + pow(By, 2)*(k0 + k1*(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t))
          + k2*pow(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t), 2))*cos(Dt*t)*cos(x[0]*Ax + At*t)
          *cos(x[1]*By + Bt*t) + By*(-By*k1*sin(x[1]*By + Bt*t)*cos(Dt*t)*cos(x[0]*Ax + At*t) - 2*By*k2*(T0 + cos(Dt*t)
          *cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t))*sin(x[1]*By + Bt*t)*cos(Dt*t)*cos(x[0]*Ax + At*t))*sin(x[1]*By + Bt*t)
          *cos(Dt*t)*cos(x[0]*Ax + At*t) + rho*(c0 + c1*(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t))
          + c2*pow(T0 + cos(Dt*t)*cos(x[0]*Ax + At*t)*cos(x[1]*By + Bt*t), 2))*(-At*sin(x[0]*Ax + At*t)*cos(Dt*t)
          *cos(x[1]*By + Bt*t) - Bt*sin(x[1]*By + Bt*t)*cos(Dt*t)*cos(x[0]*Ax + At*t) - Dt*sin(Dt*t)*cos(x[0]*Ax + At*t)
          *cos(x[1]*By + Bt*t));

	return Q_T;
}

#endif

#include"MMSP.main.hpp"
