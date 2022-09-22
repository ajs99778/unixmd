# cython: language_level=3
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.complex cimport complex
import numpy as np
cimport numpy as np

cdef extern from "rk4.c":
    void rk4(int nst, int nesteps, double dt, char *elec_object, double *energy, \
        double *energy_old, double **nacme, double **nacme_old, double **socme, \
        double **socme_old, double complex *coef, double complex **rho)
def el_run(
    program_state,
    elec_object,
):
    cdef:
        char *elec_object_c
        double *energy
        double *energy_old
        double **nacme
        double **nacme_old
        double **socme
        double **socme_old
        double complex *coef
        double complex **rho

        bytes py_bytes
        int ist, jst, nst, nesteps, verbosity
        double dt

    # Assign size variables
    nst = program_state.number_of_electronic_states
    nesteps = program_state.electronic_propogation_steps
    dt = program_state.step_size.as_atomic()

    # Allocate variables
    energy = <double*> PyMem_Malloc(nst * sizeof(double))
    energy_old = <double*> PyMem_Malloc(nst * sizeof(double))

    nacme = <double**> PyMem_Malloc(nst * sizeof(double*))
    nacme_old = <double**> PyMem_Malloc(nst * sizeof(double*))

    socme = <double**> PyMem_Malloc(nst * sizeof(double*))
    socme_old = <double**> PyMem_Malloc(nst * sizeof(double*))

    for ist in range(nst):
        nacme[ist] = <double*> PyMem_Malloc(nst * sizeof(double))
        nacme_old[ist] = <double*> PyMem_Malloc(nst * sizeof(double))
        socme[ist] = <double*> PyMem_Malloc(nst * sizeof(double))
        socme_old[ist] = <double*> PyMem_Malloc(nst * sizeof(double))

    # Assign variables from python to C
    for ist in range(nst):
        energy[ist] = program_state.state_energies[-1].as_hartree()[ist]
        energy_old[ist] = program_state.state_energies[-2].as_hartree()[ist]

    for ist in range(nst):
        for jst in range(nst):
            nacme[ist][jst] = program_state.nacmes[-1][ist, jst]
            nacme_old[ist][jst] = program_state.nacmes[-2][ist, jst]
            if program_state.intersystem_crossing:
                socme[ist][jst] = program_state.socmes[-1][ist, jst]
                socme_old[ist][jst] = program_state.socmes[-2][ist, jst]
            else:
                socme[ist][jst] = 0.
                socme_old[ist][jst] = 0.

    # Debug related
    verbosity = 1

    # # Assign coef or rho with respect to propagation scheme
    if (elec_object == "coefficient"):
        coef = <double complex*> PyMem_Malloc(nst * sizeof(double complex))
    
        for ist in range(nst):
            coef[ist] = program_state.state_coefficients[ist]

    elif (elec_object == "density"):
        rho = <double complex**> PyMem_Malloc(nst * sizeof(double complex*))
        for ist in range(nst):
            rho[ist] = <double complex*> PyMem_Malloc(nst * sizeof(double complex))
    
        for ist in range(nst):
            for jst in range(nst):
                rho[ist][jst] = program_state.rho[ist, jst]

    py_bytes = elec_object.encode()
    elec_object_c = py_bytes

    # Propagate electrons
    rk4(
        nst, nesteps, dt, elec_object_c, energy, energy_old,
        nacme, nacme_old, socme, socme_old, coef, rho
    )

    # Assign variables from C to python
    if (elec_object == "coefficient"):
        for ist in range(nst):
            program_state.state_coefficients[ist] = coef[ist]
    
        for ist in range(nst):
            for jst in range(nst):
                program_state.rho[ist, jst] = np.conj(program_state.state_coefficients[ist]) * program_state.state_coefficients[jst]
    
        PyMem_Free(coef)
    
    elif (elec_object == "density"):
        for ist in range(nst):
            for jst in range(nst):
                program_state.rho[ist, jst] = rho[ist][jst]
    
        for ist in range(nst):
            PyMem_Free(rho[ist])
        PyMem_Free(rho)
    
    # Deallocate variables
    for ist in range(nst):
        PyMem_Free(nacme[ist])
        PyMem_Free(nacme_old[ist])
        PyMem_Free(socme[ist])
        PyMem_Free(socme_old[ist])

    PyMem_Free(energy)
    PyMem_Free(energy_old)

    PyMem_Free(nacme)
    PyMem_Free(nacme_old)
    PyMem_Free(socme)
    PyMem_Free(socme_old)

    

