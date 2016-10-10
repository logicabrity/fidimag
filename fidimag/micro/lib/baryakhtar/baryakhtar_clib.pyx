import numpy
cimport numpy as np
np.import_array()

cdef extern from "baryakhtar_clib.h":
    void compute_laplace_m(double *m, double *field, double *Ms, double dx, double dy, double dz,
        int nx, int ny, int nz)

    void compute_relaxation_field_c(double *m, double *field, double *Ms, double chi_inv, int n)

    void compute_perp_field_c(double *m, double *field, double *field_p, int n)

    void llg_rhs_baryakhtar(double *dm_dt, double *m, double *h, double *delta_h,
        double *alpha, double beta, int *pins,
        double gamma, int nxyz, int do_precession)

    void llg_rhs_baryakhtar_reduced(double *dm_dt, double *m, double *hp, double *delta_hp,
                        double *alpha, double beta, int *pins,
                        double gamma, int nxyz, int do_precession, double default_c)

def compute_laplace_field(double [:] spin,
                            double [:] field,
                            double [:] Ms,
                            dx, dy, dz,
                            nx, ny, nz):

    compute_laplace_m(&spin[0], &field[0], &Ms[0], dx, dy, dz, nx, ny, nz)

def compute_relaxation_field(double [:] spin,
                            double [:] field,
                            double [:] Ms,
                            chi_inv,n):

    compute_relaxation_field_c(&spin[0], &field[0], &Ms[0], chi_inv, n)


def compute_perp_field(double [:] m,
                            double [:] field,
                            double [:] field_p,
                            n):

    compute_perp_field_c(&m[0], &field[0], &field_p[0], n)


def compute_llg_rhs_baryakhtar(double [:] dm_dt,
                               double [:] m,
                            double [:] h,
                            double [:] delta_h,
                            double [:] alpha,
                            beta,
                            int [:] pins,
                            gamma, nxyz,
                            do_precession):

    llg_rhs_baryakhtar(&dm_dt[0], &m[0], &h[0], &delta_h[0], &alpha[0], beta, &pins[0], gamma, nxyz, do_precession)


def compute_llg_rhs_baryakhtar_reduced(double [:] dm_dt,
                               double [:] m,
                            double [:] h,
                            double [:] delta_h,
                            double [:] alpha,
                            beta,
                            int [:] pins,
                            gamma, nxyz,
                            do_precession, default_c):

    llg_rhs_baryakhtar_reduced(&dm_dt[0], &m[0], &h[0], &delta_h[0], &alpha[0], beta, &pins[0], gamma, nxyz, do_precession, default_c)
