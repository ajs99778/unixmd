# cython: language_level=3
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from "tdnac.c":
    void TD_NAC(
        int istep,
        int nst,
        int nbasis,
        int norb,
        int nocc,
        int nvirt,
        double dt,
        int *orb_ini,
        int *orb_final,
        double **nacme,
        double **ao_overlap,
        double **mo_coef_old,
        double **mo_coef_new,
        double ***ci_coef_old,
        double ***ci_coef_new
    )

def wf_overlap(program_state, dt_py, nacme_out):
    cdef:
        int *orb_ini
        int *orb_final
        double **nacme
        double **ao_overlap
        double **mo_coef_old
        double **mo_coef_new
        double ***ci_coef_old
        double ***ci_coef_new

        int istep, ist, nst, ibasis, jbasis, iorb, jorb, nbasis, norb, nocc, nvirt
        double dt

    # Assign size variables
    dt = dt_py
    istep = program_state.current_step
    nst = program_state.number_of_electronic_states
    nbasis = program_state.number_of_basis_functions
    nocc = program_state.number_of_alpha_occupied
    nvirt = program_state.number_of_alpha_virtual
    norb = nvirt + nocc

    previous_mo_coeff = program_state.previous_mo_coefficients
    current_mo_coeff = program_state.current_mo_coefficients
    previous_ci_coeff = program_state.previous_ci_coefficients
    current_ci_coeff = program_state.current_ci_coefficients

    if program_state.intersystem_crossing:
        nst = 1 + int((nst - 1) / 2)
    
    # Allocate NACME variables
    orb_ini = <int*> PyMem_Malloc(1 * sizeof(int))
    orb_final = <int*> PyMem_Malloc(1 * sizeof(int))

    nacme = <double**> PyMem_Malloc(nst * sizeof(double*))

    ao_overlap = <double**> PyMem_Malloc(nbasis * sizeof(double*))
    mo_coef_old = <double**> PyMem_Malloc(norb * sizeof(double*))
    mo_coef_new = <double**> PyMem_Malloc(norb * sizeof(double*))

    for ist in range(nst):
        nacme[ist] = <double*> PyMem_Malloc(nst * sizeof(double))

    for ibasis in range(nbasis):
        ao_overlap[ibasis] = <double*> PyMem_Malloc(nbasis * sizeof(double))

    for iorb in range(norb):
        mo_coef_old[iorb] = <double*> PyMem_Malloc(nbasis * sizeof(double))
        mo_coef_new[iorb] = <double*> PyMem_Malloc(nbasis * sizeof(double))

    ci_coef_old = <double***> PyMem_Malloc(nst * sizeof(double**))
    ci_coef_new = <double***> PyMem_Malloc(nst * sizeof(double**))

    for ist in range(nst):
        ci_coef_old[ist] = <double**> PyMem_Malloc(nocc * sizeof(double*))
        ci_coef_new[ist] = <double**> PyMem_Malloc(nocc * sizeof(double*))

    for ist in range(nst):
        for iorb in range(nocc):
            ci_coef_old[ist][iorb] = <double*> PyMem_Malloc(nvirt * sizeof(double))
            ci_coef_new[ist][iorb] = <double*> PyMem_Malloc(nvirt * sizeof(double))

    # Assign NACME variables from python to C
    orb_ini[0] = program_state.orb_ini[0]
    orb_final[0] = program_state.orb_final[0]

    for ist in range(nst):
        for jst in range(nst):
            nacme[ist][jst] = 0.

    for ibasis in range(nbasis):
        for jbasis in range(nbasis):
            ao_overlap[ibasis][jbasis] = program_state.atomic_orbital_overlap[ibasis, jbasis]
    
    for iorb in range(norb):
        for ibasis in range(nbasis):
            mo_coef_old[iorb][ibasis] = previous_mo_coeff[iorb, ibasis]
            mo_coef_new[iorb][ibasis] = current_mo_coeff[iorb, ibasis]

    for ist in range(nst):
        for iorb in range(nocc):
            for jorb in range(nvirt):
                ci_coef_old[ist][iorb][jorb] = previous_ci_coeff[ist, iorb, jorb]
                ci_coef_new[ist][iorb][jorb] = current_ci_coeff[ist, iorb, jorb]

    # Calculate TDNAC term for CIoverlap
    TD_NAC(istep, nst, nbasis, norb, nocc, nvirt, dt, orb_ini, orb_final, nacme, \
        ao_overlap, mo_coef_old, mo_coef_new, ci_coef_old, ci_coef_new)
    
    # Assign NACME variables from C to python
    for ist in range(nst):
        for jst in range(nst):
            nacme_out[ist, jst] = nacme[ist][jst]

    # Deallocate NACME variables
    PyMem_Free(orb_ini)
    PyMem_Free(orb_final)

    for ist in range(nst):
        PyMem_Free(nacme[ist])
    PyMem_Free(nacme)
    
    for ibasis in range(nbasis):
        PyMem_Free(ao_overlap[ibasis])

    for iorb in range(norb):
        PyMem_Free(mo_coef_old[iorb])
        PyMem_Free(mo_coef_new[iorb])

    PyMem_Free(ao_overlap)
    PyMem_Free(mo_coef_old)
    PyMem_Free(mo_coef_new)

    for ist in range(nst):
        for iorb in range(nocc):
            PyMem_Free(ci_coef_old[ist][iorb])
            PyMem_Free(ci_coef_new[ist][iorb])

    for ist in range(nst):
        PyMem_Free(ci_coef_old[ist])
        PyMem_Free(ci_coef_new[ist])

    PyMem_Free(ci_coef_old)
    PyMem_Free(ci_coef_new)

