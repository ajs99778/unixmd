#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

// windows C compiler is stupid about complex numbers - work around it later
//#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
//
//// Routine to calculate dot product from two temporary arrays
//static double dot(int nst, _Dcomplex *u, _Dcomplex *v){
//
//    double norm, real_sum, imag_sum;
//    int ist;
//
//    for(ist = 0; ist < nst; ist++){
//        real_sum += creal(conj(u[ist])) * creal(v[ist]);
//        imag_sum = cimag(conj(u[ist])) * cimag(v[ist]);
//    }
//
//    _Dcomplex sum = {real_sum, imag_sum};
//
//    norm = creal(sum);
//    return norm;
//
//}
//
//// Routine to calculate cdot contribution originated from Ehrenfest term
//static void cdot(int nst, double *e, double **dv, _Dcomplex *c, _Dcomplex *c_dot){
//
//    int ist, jst;
//    double egs, **na_term = malloc(nst * sizeof(double*));
//    for(ist = 0; ist < nst; ist++){
//        na_term[ist] = malloc(2 * sizeof(double));
//    }
//    
//
//    for(ist = 0; ist < nst; ist++){
//        na_term[ist][0] = 0.0;
//        na_term[ist][1] = 0.0;
//        for(jst = 0; jst < nst; jst++){
//            if(ist != jst){
//                na_term[ist][0] -= creal(dv[ist][jst]) * creal(c[jst]);
//                na_term[ist][1] -= cimag(dv[ist][jst]) * cimag(c[jst]);
//            }
//        }
//    }
//
//    egs = e[0];
//    for(ist = 0; ist < nst; ist++){
//        c_dot[ist] = - 1.0 * I * c[ist] * (e[ist] - egs) + na_term[ist];
//    }
//
//    free(na_term);
//
//}
//
//// Routine to calculate rhodot contribution originated from Ehrenfest term
//static void rhodot(int nst, double *e, double **dv, _Dcomplex **rho, _Dcomplex **rho_dot){
//
//    int ist, jst, kst;
//
//    for(ist = 0; ist < nst; ist++){
//        for(jst = 0; jst < nst; jst++){
//            rho_dot[ist][jst] = 0.0 + 0.0 * I;
//        }
//    }
//
//    for(ist = 0; ist < nst; ist++){
//        for(jst = 0; jst < nst; jst++){
//            if(ist != jst){
//                rho_dot[ist][ist] -= dv[ist][jst] * 2.0 * creal(rho[ist][jst]);
//            }
//        }
//    }
//
//    for(ist = 0; ist < nst; ist++){
//        for(jst = ist + 1; jst < nst; jst++){
//            rho_dot[ist][jst] -=  1.0 * I * (e[jst] - e[ist]) * rho[ist][jst];
//            for(kst = 0; kst < nst; kst++){
//                rho_dot[ist][jst] -= dv[ist][kst] * rho[kst][jst] + dv[jst][kst] * rho[ist][kst];
//            }
//            rho_dot[jst][ist] = conj(rho_dot[ist][jst]);
//        }
//    }
//
//}
//
//
//
//#else
    
// Routine to calculate dot product from two temporary arrays
static double dot(int nst, double complex *u, double complex *v){

    double complex sum;
    double norm;
    int ist;

    sum = 0.0 + 0.0 * I;
    for(ist = 0; ist < nst; ist++){
        sum += conj(u[ist]) * v[ist];
    }

    norm = creal(sum);
    return norm;

}

// Routine to calculate cdot contribution originated from Ehrenfest term
static void cdot(int nst, double *e, double **dv, double **h, double complex *c, double complex *c_dot){

    double complex *na_term = malloc(nst * sizeof(double complex));
    double complex *so_term = malloc(nst * sizeof(double complex));

    int ist, jst;
    double egs;

    for(ist = 0; ist < nst; ist++){
        na_term[ist] = 0.0 + 0.0 * I;
        so_term[ist] = 0.0 + 0.0 * I;
        for(jst = 0; jst < nst; jst++){
            if(ist != jst){
                na_term[ist] -= dv[ist][jst] * c[jst];
                so_term[ist] += h[ist][jst] * c[jst];
            }
        }
    }

    egs = e[0];
    for(ist = 0; ist < nst; ist++){
        //           |           energy term             |     NAC      |          SOC         |
        c_dot[ist] = - 1.0 * I * c[ist] * (e[ist] - egs) + na_term[ist] - 1.0 * I * so_term[ist];
    }
    
    free(na_term);
    free(so_term);


}

// Routine to calculate rhodot contribution originated from Ehrenfest term
static void rhodot(int nst, double *e, double **dv, double **h, double complex **rho, double complex **rho_dot){

    int ist, jst, kst;

    for(ist = 0; ist < nst; ist++){
        for(jst = 0; jst < nst; jst++){
            rho_dot[ist][jst] = 0.0 + 0.0 * I;
        }
    }

    for(ist = 0; ist < nst; ist++){
        for(jst = 0; jst < nst; jst++){
            if(ist != jst){
                rho_dot[ist][ist] -= dv[ist][jst] * 2.0 * creal(rho[ist][jst]);
            }
        }
    }

    for(ist = 0; ist < nst; ist++){
        for(jst = ist + 1; jst < nst; jst++){
            rho_dot[ist][jst] -=  1.0 * I * (e[jst] - e[ist]) * rho[ist][jst];
            for(kst = 0; kst < nst; kst++){
                rho_dot[ist][jst] -= dv[ist][kst] * rho[kst][jst] + dv[jst][kst] * rho[ist][kst];
            }
            rho_dot[jst][ist] = conj(rho_dot[ist][jst]);
        }
    }

}


//#endif
