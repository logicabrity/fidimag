#include "clib.h"


void compute_anis(double *spin, double *field, double *energy,
	double *Ku, double *axis, int n) {

    /* Remember that the magnetisation order is
     *      mx1, my1, mz1, mx2, my2, mz2, mx3,...
     * so we get the corresponding components multiplying
     * by 3 in every iteration. The anisotropy *axis* has the
     * same ordering than the magnetisation (it can vary
     * in space) and the anisotropy constant can change on
     * every lattice site
     * Neighbouring sites are not relevant here
     *
     */
    #pragma omp parallel for
	for (int i = 0; i < n; i++) {

        double m_u = (spin[3 * i] * axis[3 * i] +
                      spin[3 * i + 1] * axis[3 * i + 1] +
                      spin[3 * i + 2] * axis[3 * i + 2]);


		field[3 * i]     = 2 * Ku[i] * m_u * axis[3 * i]    ;
		field[3 * i + 1] = 2 * Ku[i] * m_u * axis[3 * i + 1];
		field[3 * i + 2] = 2 * Ku[i] * m_u * axis[3 * i + 2];

		energy[i] = -Ku[i] * (m_u * m_u);

	}

}


void compute_anis_cubic(double *spin, double *field, double *energy,
	double *Kc, int n) {

    /* Remember that the magnetisation order is
     *      mx1, my1, mz1, mx2, my2, mz2, mx3,...
     * so we get the corresponding components multiplying
     * by 3 in every iteration.
     *
     */
      #pragma omp parallel for
	    for (int i = 0; i < n; i++) {
				int j = 3*i;
				field[j] = - 4*Kc[i]*spin[j]*spin[j]*spin[j];
				field[j+1] = - 4*Kc[i]*spin[j+1]*spin[j+1]*spin[j+1];
				field[j+2] = - 4*Kc[i]*spin[j+2]*spin[j+2]*spin[j+2];

	      energy[i] = -0.25*(field[j]*spin[j]+field[j+1]*spin[j+1]+field[j+2]*spin[j+2]) ;

	    }

}
