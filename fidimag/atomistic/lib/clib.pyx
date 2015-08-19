import numpy
cimport numpy as np
np.import_array()


cdef extern from "clib.h":
    
    void gauss_random_vec_with_init(double *x, int n)
    
    double skyrmion_number(double *spin, double *charge, int nx, int ny, int nz)
    void compute_guiding_center(double *spin, int nx, int ny, int nz, double *res)
    void compute_px_py_c(double *spin, int nx, int ny, int nz, double *px, double *py)

    void compute_exch_field(double *spin, double *field, double *energy,
                            double Jx, double Jy, double Jz,
                            int *ngbs, int nxyz)
    double compute_exch_energy(double *spin, double Jx, double Jy, double Jz, int nx, int ny, int nz, int xperiodic, int yperiodic)
    
    void dmi_field_bulk(double *spin, double *field, double *energy, double D, int *ngbs, int nxyz) 
    void dmi_field_interfacial_atomistic(double *spin, double *field, double *energy, double D, int *ngbs, int nxyz)
    double dmi_energy(double *spin, double D, int nx, int ny, int nz, int xperiodic, int yperiodic)

    void compute_anis(double *spin, double *field, double *energy,
                      double *Ku, double *axis, int nxyz)

    void llg_rhs(double * dm_dt, double * spin, double *h, double *alpha, int *pins, 
                 double gamma, int nxyz, int do_procession, double default_c)
    void llg_s_rhs(double * dm_dt, double * spin, double * h, double *alpha, double *chi, double gamma, int nxyz)
    void llg_rhs_jtimes(double *jtn, double *m, double *h, double *mp, double *hp, double *alpha, int *pins,
                double gamma, int nxyz, int do_procession, double default_c)

    void normalise(double *m, int nxyz)
    void compute_stt_field_c(double *spin, double *field, double *jx, double *jy,
        double dx, double dy, int nx, int ny, int nz, int xperiodic,
        int yperiodic)
    void llg_stt_rhs(double *dm_dt, double *m, double *h, double *h_stt, double *alpha,double beta, double u0, double gamma, int nxyz)


    # used for sllg
    ctypedef struct ode_solver:
        pass

    ode_solver *create_ode_plan()
    void init_solver(ode_solver *s, double k_B, double theta, int nxyz, double dt, double gamma)
    void finalize_ode_plan(ode_solver *plan)
    void run_step1(ode_solver *s, double *m, double *h, double *m_pred, double *T, double *alpha, double *mu_s_inv, int *pins)
    void run_step2(ode_solver *s, double *m_pred, double *h, double *m, double *T, double *alpha, double *mu_s_inv, int *pins)

def random_number(np.ndarray[double, ndim=1, mode="c"] v):
    cdef int n = len(v)
    
    
    print n
    
    gauss_random_vec_with_init(&v[0], n)
    


def compute_skymrion_number(np.ndarray[double, ndim=1, mode="c"] spin,
                            np.ndarray[double, ndim=1, mode="c"] charge,
                            nx, ny, nz):

    return skyrmion_number(&spin[0], &charge[0], nx, ny, nz)

def compute_RxRy(np.ndarray[double, ndim=1, mode="c"] spin,
                            nx, ny, nz):
    
    res = numpy.array([0.0,0.0])
    
    cdef np.ndarray[double, ndim=1, mode="c"] R = res

    compute_guiding_center(&spin[0], nx, ny, nz, &R[0])
    
    return res[0], res[1]

def compute_px_py(np.ndarray[double, ndim=1, mode="c"] spin,
                  nx,ny,nz,
                  np.ndarray[double, ndim=1, mode="c"] px,
                  np.ndarray[double, ndim=1, mode="c"] py):

    compute_px_py_c(&spin[0], nx, ny, nz, &px[0], &py[0])

def compute_exchange_field(np.ndarray[double, ndim=1, mode="c"] spin,
                            np.ndarray[double, ndim=1, mode="c"] field,
                            np.ndarray[double, ndim=1, mode="c"] energy,
                            Jx, Jy, Jz,
                            np.ndarray[int, ndim=2, mode="c"] ngbs,
                            nxyz
                            ):

    compute_exch_field(&spin[0], &field[0], &energy[0],
		       Jx, Jy, Jz,
                       &ngbs[0, 0], nxyz)
    
def compute_exchange_energy(np.ndarray[double, ndim=1, mode="c"] spin,
                            Jx, Jy, Jz, nx, ny, nz, xperiodic,yperiodic):

    return compute_exch_energy(&spin[0], Jx, Jy, Jz, nx, ny, nz, xperiodic, yperiodic)


def compute_anisotropy(np.ndarray[double, ndim=1, mode="c"] spin,
                        np.ndarray[double, ndim=1, mode="c"] field,
                        np.ndarray[double, ndim=1, mode="c"] energy,
                        np.ndarray[double, ndim=1, mode="c"] Ku,
                        np.ndarray[double, ndim=1, mode="c"] axis,
                        nxyz):
    compute_anis(&spin[0], &field[0], &energy[0], &Ku[0], 
                 &axis[0], nxyz)
    
    
def compute_dmi_field(np.ndarray[double, ndim=1, mode="c"] spin,
                      np.ndarray[double, ndim=1, mode="c"] field,
                      np.ndarray[double, ndim=1, mode="c"] energy,
                      D,
                      np.ndarray[int, ndim=2, mode="c"] ngbs,
                      nxyz):
    dmi_field_bulk(&spin[0], &field[0], &energy[0], D, &ngbs[0,0], nxyz)
    

def compute_dmi_field_interfacial(np.ndarray[double, ndim=1, mode="c"] spin,
                      np.ndarray[double, ndim=1, mode="c"] field,
                      np.ndarray[double, ndim=1, mode="c"] energy,
                      D,
                      np.ndarray[int, ndim=2, mode="c"] ngbs,
                      nxyz):
    dmi_field_interfacial_atomistic(&spin[0], &field[0], &energy[0],
                                    D, &ngbs[0, 0], nxyz)
    
def compute_dmi_energy(np.ndarray[double, ndim=1, mode="c"] spin,
                        D, nx, ny, nz,
                        xperiodic,yperiodic):
    return dmi_energy(&spin[0], D, nx, ny, nz, xperiodic,yperiodic)
    


def compute_llg_rhs(np.ndarray[double, ndim=1, mode="c"] dm_dt,
                np.ndarray[double, ndim=1, mode="c"] spin,
                np.ndarray[double, ndim=1, mode="c"] field,
                np.ndarray[double, ndim=1, mode="c"] alpha,
                np.ndarray[int, ndim=1, mode="c"] pins,
                gamma, nxyz, do_procession, default_c):
    llg_rhs(&dm_dt[0], &spin[0], &field[0], &alpha[0], &pins[0], gamma, nxyz, do_procession, default_c)

def compute_llg_jtimes(np.ndarray[double, ndim=1, mode="c"] jtn,
                np.ndarray[double, ndim=1, mode="c"] m,
                np.ndarray[double, ndim=1, mode="c"] field,
                np.ndarray[double, ndim=1, mode="c"] mp,
                np.ndarray[double, ndim=1, mode="c"] field_p,
                np.ndarray[double, ndim=1, mode="c"] alpha,
                np.ndarray[int, ndim=1, mode="c"] pins,
                gamma, nxyz, do_procession, default_c):
    llg_rhs_jtimes(&jtn[0], &m[0], &field[0], &mp[0], &field_p[0], &alpha[0], &pins[0], gamma, nxyz, do_procession, default_c)
    
def compute_llg_s_rhs(np.ndarray[double, ndim=1, mode="c"] dm_dt,
                np.ndarray[double, ndim=1, mode="c"] spin,
                np.ndarray[double, ndim=1, mode="c"] field,
                np.ndarray[double, ndim=1, mode="c"] alpha,
                np.ndarray[double, ndim=1, mode="c"] chi,
                gamma, nxyz):
    llg_s_rhs(&dm_dt[0], &spin[0], &field[0], &alpha[0], &chi[0], gamma, nxyz)


def compute_stt_field(np.ndarray[double, ndim=1, mode="c"] spin,
                np.ndarray[double, ndim=1, mode="c"] field,
                np.ndarray[double, ndim=1, mode="c"] jx,
                np.ndarray[double, ndim=1, mode="c"] jy,
                dx, dy, nx, ny, nz, xperiodic, yperiodic):
    compute_stt_field_c(&spin[0], &field[0], &jx[0], &jy[0], dx, dy, nx, ny, nz, xperiodic, yperiodic)

def compute_llg_stt_rhs(np.ndarray[double, ndim=1, mode="c"] dm_dt,
                np.ndarray[double, ndim=1, mode="c"] spin,
                np.ndarray[double, ndim=1, mode="c"] field,
                np.ndarray[double, ndim=1, mode="c"] field_stt,
                np.ndarray[double, ndim=1, mode="c"] alpha,
                beta, u0, gamma, nxyz):
    llg_stt_rhs(&dm_dt[0], &spin[0], &field[0], &field_stt[0] ,&alpha[0], beta, u0, gamma, nxyz)


def normalise_spin(np.ndarray[double, ndim=1, mode="c"] spin, nxyz):
    normalise(&spin[0], nxyz)

cdef class RK2S(object):
    cdef ode_solver * _c_plan
    cdef double dt
    cdef update_fun
    cdef np.ndarray pred_m
    cdef np.ndarray field
    cdef np.ndarray mu_s_inv
    cdef np.ndarray T
    cdef np.ndarray alpha
    cdef np.ndarray pins

    cdef public double t
    cdef public int step
    cdef public np.ndarray y

    def __cinit__(self,dt,nxyz,gamma,k_B,theta,
                            np.ndarray[double, ndim=1, mode="c"] mu_s_inv,
                            np.ndarray[double, ndim=1, mode="c"] alpha,
                            np.ndarray[double, ndim=1, mode="c"] spin,
                            np.ndarray[double, ndim=1, mode="c"] field,
                            np.ndarray[double, ndim=1, mode="c"] T,
                            np.ndarray[int, ndim=1, mode="c"] pins,
                            update_fun):

        self.t = 0
        self.step = 0
        self.dt = dt

        self.update_fun = update_fun
        self.mu_s_inv = mu_s_inv
        self.field = field
        self.T = T
        self.alpha = alpha
        self.pins = pins
        self.pred_m = numpy.zeros(3*nxyz,dtype=numpy.float)
        self.y = numpy.zeros(3*nxyz,dtype=numpy.float)
        

        self._c_plan = create_ode_plan()
        if self._c_plan is NULL:
            raise MemoryError()

        init_solver(self._c_plan,k_B,theta,nxyz,dt,gamma)

    def __dealloc__(self):
        if self._c_plan is not NULL:
            finalize_ode_plan(self._c_plan)
            self._c_plan = NULL


    def set_initial_value(self,np.ndarray[double, ndim=1, mode="c"] spin, t):
        self.t = t
        self.y[:] = spin[:]

    def successful(self):
        #print self.spin
        return True

    def run_step(self):
        cdef np.ndarray[double, ndim=1, mode="c"] y=self.y
        cdef np.ndarray[double, ndim=1, mode="c"] field=self.field
        cdef np.ndarray[double, ndim=1, mode="c"] pred_m=self.pred_m
        cdef np.ndarray[double, ndim=1, mode="c"] T=self.T
        cdef np.ndarray[double, ndim=1, mode="c"] alpha=self.alpha
        cdef np.ndarray[double, ndim=1, mode="c"] mu_s_inv=self.mu_s_inv
        cdef np.ndarray[int, ndim=1, mode="c"] pins = self.pins

        #print "from cython1", self.spin,self.field,self.pred_m
        self.update_fun(self.y, self.t)
        run_step1(self._c_plan,&y[0],&field[0],&pred_m[0],&T[0],&alpha[0], &mu_s_inv[0], &pins[0])
        
        self.step += 1
        self.t = self.step*self.dt
        
        self.update_fun(self.pred_m, self.t)
        run_step2(self._c_plan,&pred_m[0],&field[0],&y[0],&T[0],&alpha[0], &mu_s_inv[0], &pins[0])
        

    def run_until(self, t):
        while (self.t<t):
            self.run_step()
        return 0
