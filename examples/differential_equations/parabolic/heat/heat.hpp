// heat.hpp
// Example program for the heat model using MMSP and the Method of Manufactured Solutions
// Questions/comments to trevor.keller@gmail.com (Trevor Keller)

std::string PROGRAM = "heat";
std::string MESSAGE = "method of manufactured solutions using MMSP";

typedef MMSP::grid<1,double> GRID1D;
typedef MMSP::grid<2,double> GRID2D;
typedef MMSP::grid<3,double> GRID3D;

// Create a templated Manufactured Solution-specific derived grid class
namespace MMSP {
class coefficients {
	public:
		coefficients(int dim); // constructor -- takes only grid dimension as an argument
		vector<int> N;     // number of grid points in each direction
		double rho;        // material density (constant)
		double dt;         // timestep
		double noiseamp;   // amplitude of random initial noise
		double initscale;  // scaling factor for initial condition (should be "not too close to 1")
		vector<double> h;  // grid spacing
		vector<double> Ax; // amplitudes in space
		vector<double> At; // amplitudes in time
		vector<double> Ck; // polynomial coefficients for thermal conductivity
		vector<double> Cc; // polynomial coefficients for heat capacity
};
} // namespace

// temperature-dependent thermal conductivity and its temperature-derivative
double MS_k (const MMSP::vector<double>& Ck, const double& temp);
double MS_dkdT (const MMSP::vector<double>& Ck, const double& temp);


// temperature-dependent heat capacity
double MS_Cp (const MMSP::vector<double>& Cc, const double& temp);

// analytical expression for manufactured temperature profile
double MS_T (const MMSP::vector<int>& xidx, double t,
             const MMSP::vector<double>& h,  // grid spacing
             const MMSP::vector<double>& Ax, // spatial amplitudes
             const MMSP::vector<double>& At // temporal amplitudes
);

// analytical expression for manufactured thermal source
double MS_Q (const MMSP::vector<int>& xidx, double t, double rho,
             const MMSP::vector<double>& h,  // grid spacing
             const MMSP::vector<double>& Ax, // spatial amplitudes
             const MMSP::vector<double>& At, // temporal amplitudes
             const MMSP::vector<double>& Ck, // thermal diffusivity polynomial coefficients
             const MMSP::vector<double>& Cc  // heat capacity polynomial coefficients
);

// analytical expression for gradient of manufactured temperature profile
MMSP::vector<double> MS_GradT (const MMSP::vector<int>& xidx, double t,
                         const MMSP::vector<double>& h,  // grid spacing
                         const MMSP::vector<double>& Ax, // spatial amplitudes
                         const MMSP::vector<double>& At // temporal amplitudes
);

