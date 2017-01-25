#include<math.h>
#include<complex.h>
#include<fftw3.h>
//#include<omp.h>

#define WIDE_PI 3.1415926535897932384626433832795L

inline double cross_x(double a0, double a1, double a2, double b0, double b1, double b2) { return a1*b2 - a2*b1; }
inline double cross_y(double a0, double a1, double a2, double b0, double b1, double b2) { return a2*b0 - a0*b2; }
inline double cross_z(double a0, double a1, double a2, double b0, double b1, double b2) { return a0*b1 - a1*b0; }


enum Type_Nij {
	Tensor_xx, Tensor_yy, Tensor_zz, Tensor_xy, Tensor_xz, Tensor_yz
};

//==========================================
/* The following structs are not actually used by
FFTW themselves - the name is slightly misleading.
They basically hold several things that C functions
for calculating the demagnetising energy need.

(a)   i) System properties, e.g nx, ny, nz, scale
     ii) Quantities derived from system properties,
        e.g nxy, nxyz, lenx, leny, lenz

(b)  Pointers to arrays for
     * Demagnetising Tensor
     * Arrays to hold spin * Ms
     * Fourier transforms of these.
(c)  FFTW plans.

We may want to split up these things in future.

Also, we could probably delete the demagnetising tensor
after taking the fourier transform of it, as this is not
needed after that.
*/

typedef struct {
	int nx, ny, nz, nxy, nxyz;
	double dx, dy, dz;
	int lenx, leny, lenz, lenxy, lenxyz;
	int total_length;
        double scale;
	double *tensor_xx;
	double *tensor_yy;
	double *tensor_zz;
	double *tensor_xy;
	double *tensor_xz;
	double *tensor_yz;

	fftw_complex *Nxx;
	fftw_complex *Nyy;
	fftw_complex *Nzz;
	fftw_complex *Nxy;
	fftw_complex *Nxz;
	fftw_complex *Nyz;

	fftw_complex *Mx;
	fftw_complex *My;
	fftw_complex *Mz;
	fftw_complex *Hx;
	fftw_complex *Hy;
	fftw_complex *Hz;

	double *mx;
	double *my;
	double *mz;
	double *hx;
	double *hy;
	double *hz;

	//we need three plans
	fftw_plan tensor_plan;
	fftw_plan m_plan;
	fftw_plan h_plan;

} fft_demag_plan;

typedef struct {
	int nx, ny, nz, nxy, nxyz;
	double dx, dy, dz;
	int lenx, leny, lenz, lenxy, lenxyz;
	int total_length;
 	double scale;
	double *sx;
	double *sy;
	double *sz;

	fftw_complex *Sx;
	fftw_complex *Sy;
	fftw_complex *Sz;


	fftw_complex *Mx;
	fftw_complex *My;
	fftw_complex *Mz;
	fftw_complex *Phi;

	double *mx;
	double *my;
	double *mz;
	double *phi;

	//we need three plans
	fftw_plan tensor_plan;
	fftw_plan r2c_plan;
	fftw_plan c2r_plan;

} fft_demag_plan_scalar;

fft_demag_plan *create_plan(void);
void finalize_plan(fft_demag_plan *plan);
void init_plan(fft_demag_plan *plan, double dx, double dy, double dz, int nx, int ny, int nz);
void compute_dipolar_tensors(fft_demag_plan *plan);
void compute_demag_tensors(fft_demag_plan *plan);
void create_fftw_plan(fft_demag_plan *plan);
void compute_demag_tensors_2dpbc(fft_demag_plan *plan, double *tensors, double pbc_2d_error, int sample_repeat_nx, int sample_repeat_ny, double dipolar_radius);
void fill_demag_tensors_c(fft_demag_plan *plan, double *tensors);
void compute_fields(fft_demag_plan *plan, double *spin, double *mu_s, double *field);
void exact_compute(fft_demag_plan *plan, double *spin, double *mu_s, double *field);
double compute_demag_energy(fft_demag_plan *plan, double *spin, double *mu_s, double *field);


fft_demag_plan_scalar *create_plan_scalar(void);
void finalize_plan_scalar(fft_demag_plan_scalar *plan);
void init_plan_scalar(fft_demag_plan_scalar *plan, double dx, double dy, double dz, int nx, int ny, int nz);
void compute_dipolar_tensors_scalar(fft_demag_plan_scalar *plan);
void compute_demag_tensors_scalar(fft_demag_plan_scalar *plan);
void create_fftw_plan_scalar(fft_demag_plan_scalar *plan);
void compute_demag_tensors_2dpbc_scalar(fft_demag_plan_scalar *plan, double *tensors, double pbc_2d_error, int sample_repeat_nx, int sample_repeat_ny, double dipolar_radius);
void fill_demag_tensors_c_scalar(fft_demag_plan_scalar *plan, double *tensors);
void compute_fields_scalar(fft_demag_plan_scalar *plan, double *spin, double *mu_s, double *field);
void exact_compute_scalar(fft_demag_plan_scalar *plan, double *spin, double *mu_s, double *field);
double compute_demag_energy_scalar(fft_demag_plan_scalar *plan, double *spin, double *mu_s, double *field);



//=========================================================
//=========================================================
//used for sode
typedef struct {
	int nxyz;

	double dt;
	double T;
	double gamma;
	double *mu_s;
	double coeff;
	double Q;

	double theta;
	double theta1;
	double theta2;

	double *dm1;
	double *dm2;
	double *eta;

} ode_solver;

void init_solver(ode_solver *s, double k_B, double theta, int nxyz, double dt, double gamma);
ode_solver *create_ode_plan(void);
void finalize_ode_plan(ode_solver *plan);
void run_step1(ode_solver *s, double *m, double *h, double *m_pred, double *T,
		double *alpha, double *mu_s_inv, int *pins);
void run_step2(ode_solver *s, double *m_pred, double *h, double *m, double *T,
		double *alpha, double *mu_s_inv, int *pins);


double CalcSx(double x, double y, double z, double dx, double dy, double dz, double r);
double CalcSy(double x, double y, double z, double dx, double dy, double dz, double r);
double CalcSz(double x, double y, double z, double dx, double dy, double dz, double r);
