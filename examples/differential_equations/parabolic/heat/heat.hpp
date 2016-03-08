// heat.hpp
// Example program for the heat model using MMSP and the Method of Manufactured Solutions
// Questions/comments to trevor.keller@gmail.com (Trevor Keller)

std::string PROGRAM = "heat";
std::string MESSAGE = "method of manufactured solutions using MMSP";

typedef MMSP::mmsgrid<1,double> GRID1D;
typedef MMSP::mmsgrid<2,double> GRID2D;
typedef MMSP::mmsgrid<3,double> GRID3D;

// Create a templated Manufactured Solution-specific derived grid class
template<int dim> class MMSP::coefficients {
	public:
		coefficients();    // constructor
		vector<int> N;     // number of grid points in each direction
		vector<double> h;  // grid spacing
		vector<double> Ax; // amplitudes in space
		vector<double> At; // amplitudes in time
		vector<double> Ck; // polynomial coefficients for thermal conductivity
		vector<double> Cc; // polynomial coefficients for heat capacity
		double rho;        // material density (constant)
};

// temperature-dependent thermal conductivity
double MS_k (const vector<double>& Ck, double temp);

// temperature-dependent heat capacity
double MS_Cp (const vector<double>& Cc, double temp);

// analytical expression for manufactured temperature profile
double MS_T (const vector<int>& xidx, double t,
             const vector<double>& h,  // grid spacing
             const vector<double>& Ax, // spatial amplitudes
             const vector<double>& At // temporal amplitudes
);

// analytical expression for manufactured thermal source
double MS_Q (const vector<int>& xidx, double t, double rho,
             const vector<double>& h,  // grid spacing
             const vector<double>& Ax, // spatial amplitudes
             const vector<double>& At, // temporal amplitudes
             const vector<double>& Ck, // thermal diffusivity polynomial coefficients
             const vector<double>& Cc  // heat capacity polynomial coefficients
);

// analytical expression for gradient of manufactured temperature profile
vector<double> MS_GradT (const vector<int>& xidx, double t,
                         const vector<double>& h,  // grid spacing
                         const vector<double>& Ax, // spatial amplitudes
                         const vector<double>& At // temporal amplitudes
);

