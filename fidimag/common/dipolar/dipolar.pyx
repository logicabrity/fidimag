cimport cython
import numpy
cimport numpy as np
np.import_array()


cdef extern from "dipolar.h":
    
    # used for demag
    ctypedef struct fft_demag_plan:
        pass

    fft_demag_plan * create_plan() nogil
    void finalize_plan(fft_demag_plan * plan) nogil
    int init_plan(fft_demag_plan * plan, double dx, double dy, double dz, int nx,int ny, int nz) nogil
    void compute_fields(fft_demag_plan * plan, double *spin, double *mu_s, double *field) nogil
    void exact_compute(fft_demag_plan * plan, double *spin, double *mu_s, double *field) nogil
    double compute_demag_energy(fft_demag_plan *plan, double *spin, double *mu_s, double *field) nogil
    void compute_dipolar_tensors(fft_demag_plan *plan) nogil
    void compute_demag_tensors(fft_demag_plan *plan) nogil
    void create_fftw_plan(fft_demag_plan *plan) nogil
    void compute_demag_tensors_2dpbc(fft_demag_plan *plan, double *tensors, double pbc_2d_error, int sample_repeat_nx, int sample_repeat_ny, double dipolar_radius) nogil
    void fill_demag_tensors_c(fft_demag_plan *plan, double *tensors) nogil

@cython.boundscheck(False)
cdef class FFTDemag(object):
    cdef fft_demag_plan *_c_plan
    
    
    #tensor_type could be 'dipolar', 'demag' or '2d_pbc'
    def __cinit__(self, double dx, double dy, double dz, int nx, int ny, int nz, tensor_type='dipolar'):
        with nogil:
            self._c_plan = create_plan()
        if self._c_plan is NULL:
            raise MemoryError()
        with nogil:
            flag = init_plan(self._c_plan, dx, dy, dz, nx, ny, nz)
        if flag == 0:
            raise Exception("FFTW Init failed")
        if tensor_type == 'dipolar':
            compute_dipolar_tensors(self._c_plan)
            create_fftw_plan(self._c_plan)
        elif tensor_type == 'demag':
            compute_demag_tensors(self._c_plan)
            create_fftw_plan(self._c_plan)
        elif tensor_type == '2d_pbc':
            pass
        else:
            raise Exception("Only support options 'dipolar', 'demag' and '2d_pbc'.")

    def free(self):
        self.__dealloc__()

    def __dealloc__(self):
        if self._c_plan is not NULL:
            finalize_plan(self._c_plan)
            self._c_plan = NULL

    cdef void compute_tensors_2dpbc(self, double [:] tensors, 
                            double pbc_2d_error, int sample_repeat_nx, int sample_repeat_ny, double dipolar_radius) nogil:
        compute_demag_tensors_2dpbc(self._c_plan, &tensors[0], pbc_2d_error, 
                            sample_repeat_nx, sample_repeat_ny, dipolar_radius)


    def fill_demag_tensors(self, double [:] tensors):
        with nogil:
            fill_demag_tensors_c(self._c_plan, &tensors[0])
            create_fftw_plan(self._c_plan)

    def compute_field(self,double [:] spin,
                        double [:] mu_s,
                        double [:] field):
        with nogil:
            compute_fields(self._c_plan, &spin[0], &mu_s[0], &field[0])

    def compute_exact(self,
                      double [:] spin,
                      double [:] mu_s,
                      double [:] field):
        with nogil:
            exact_compute(self._c_plan, &spin[0], &mu_s[0], &field[0])

    def compute_energy(self,
                      double [:] spin,
                      double [:] mu_s,
                      double [:] field):
        cdef double result
        with nogil:
            result = compute_demag_energy(self._c_plan, &spin[0], &mu_s[0], &field[0])
        return result

cdef extern from "demagcoef.h" nogil:
    double CalculateSDA00(double x, double y, double z, double dx,double dy,double dz)
    double DemagNxxAsymptotic(double x, double y, double z, double dx,double dy,double dz)
    
cdef double compute_Nxx(double x, double y, double z, double dx, double dy, double dz) nogil:
    cdef double result = CalculateSDA00(x,y,z,dx,dy,dz)
    return result

cdef double compute_Nxx_asy(double x, double y, double z, double dx, double dy, double dz) nogil:
    cdef double result = DemagNxxAsymptotic(x, y, z, dx, dy, dz)
    return result
