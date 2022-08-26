# cython: language_level=3
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.complex cimport complex
import numpy as np
cimport numpy as np

cdef extern from "rk4.c":
    void rk4(int nst, int nesteps, double dt, char *elec_object, double *energy, \
        double *energy_old, double **nacme, double **nacme_old, double complex *coef, \
        double complex **rho)
def el_run(
    number_of_states,
    number_of_elec_steps,
    time_step,
    current_energies,
    previous_energies,
    current_nacme,
    previous_nacme,
    state_coeff,
    mol_rho,
    elec_object,
):
    cdef:
        char *elec_object_c
        double *energy
        double *energy_old
        double **nacme
        double **nacme_old
        double complex *coef
        double complex **rho

        bytes py_bytes
        int ist, jst, nst, nesteps, verbosity
        double dt

    # Assign size variables
    nst = number_of_states
    nesteps = number_of_elec_steps
    dt = time_step

    # Allocate variables
    energy = <double*> PyMem_Malloc(nst * sizeof(double))
    energy_old = <double*> PyMem_Malloc(nst * sizeof(double))

    nacme = <double**> PyMem_Malloc(nst * sizeof(double*))
    nacme_old = <double**> PyMem_Malloc(nst * sizeof(double*))

    for ist in range(nst):
        nacme[ist] = <double*> PyMem_Malloc(nst * sizeof(double))
        nacme_old[ist] = <double*> PyMem_Malloc(nst * sizeof(double))

    # Assign variables from python to C
    for ist in range(nst):
        energy[ist] = current_energies[ist]
        energy_old[ist] = previous_energies[ist]

    for ist in range(nst):
        for jst in range(nst):
            nacme[ist][jst] = current_nacme[ist, jst]
            nacme_old[ist][jst] = previous_nacme[ist, jst]

    # Debug related
    verbosity = 0

    # # Assign coef or rho with respect to propagation scheme
    # if (elec_object == "coefficient"):
    # 
    #     coef = <double complex*> PyMem_Malloc(nst * sizeof(double complex))
    # 
    #     for ist in range(nst):
    #         coef[ist] = md.mol.states[ist].coef

    # elif (elec_object == "density"):

    rho = <double complex**> PyMem_Malloc(nst * sizeof(double complex*))
    for ist in range(nst):
        rho[ist] = <double complex*> PyMem_Malloc(nst * sizeof(double complex))
    
    for ist in range(nst):
        for jst in range(nst):
            rho[ist][jst] = mol_rho[ist, jst]

    py_bytes = elec_object.encode()
    elec_object_c = py_bytes

    # Propagate electrons
    rk4(nst, nesteps, dt, elec_object_c, energy, energy_old, nacme, nacme_old, coef, rho)

    # # Assign variables from C to python
    # if (md.elec_object == "coefficient"):
    # 
    #     for ist in range(nst):
    #         md.mol.states[ist].coef = coef[ist]
    # 
    #     for ist in range(nst):
    #         for jst in range(nst):
    #             md.mol.rho[ist, jst] = np.conj(md.mol.states[ist].coef) * md.mol.states[jst].coef
    # 
    #     PyMem_Free(coef)
    # 
    # elif (md.elec_object == "density"):

    # Deallocate variables
    for ist in range(nst):
        PyMem_Free(nacme[ist])
        PyMem_Free(nacme_old[ist])

    PyMem_Free(energy)
    PyMem_Free(energy_old)

    PyMem_Free(nacme)
    PyMem_Free(nacme_old)
    for ist in range(nst):
        for jst in range(nst):
            mol_rho[ist, jst] = rho[ist][jst]
    
    for ist in range(nst):
        PyMem_Free(rho[ist])
    PyMem_Free(rho)

    

