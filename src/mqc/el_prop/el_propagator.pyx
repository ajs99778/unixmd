# cython: language_level=3
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.complex cimport complex
import numpy as np
cimport numpy as np


def cdot(nst, e, dv, h, c):
    na_term = np.zeros(nst, dtype=np.cdouble)
    so_term = np.zeros(nst, dtype=np.cdouble)

    print("cdot c", c, flush=True)

    for i in range(0, nst):
        for j in range(0, nst):
            if i == j:
                continue
            na_term[i] -= dv[i, j] * c[j]
            so_term[i] += h[i, j] * c[j]

    print("nrg", -1.0j * c * (e - e[0]))
    print("nac", na_term)
    print("soc", -1.0j * so_term)

    return -1.0j * c * (e - e[0]) + na_term - 1.0j * so_term


def rhodot(nst, e, dv, h, rho, rho_dot):
    rho_dot = np.zeros((nst, nst), dtype=np.cdouble)

    for i in range(0, nst):
        for j in range(0, nst):
            if i == j:
                continue
            rho_dot[i, j] -= dv[i, j] * 2 * np.real(rho[i, j])
    
    for i in range(0, nst):
        for j in range(i + 1, nst):
            rho_dot[i, j] -= 1.0j * (e[j] - e[i]) * rho[i, j]
            for k in range(0, nst):
                rho_dot[i, j] -= dv[i, k] * rho[k, j] + dv[j, k] * rho[i, k]
            rho_dot[j, i] = np.conjugate(rho_dot[i, j])




def rk4_coef(nst, nesteps, dt, elec_object, energy, energy_old, nacme, nacme_old, socme, socme_old, coef):
    k1 = np.zeros(nst, dtype=np.cdouble)
    k2 = np.zeros(nst, dtype=np.cdouble)
    k3 = np.zeros(nst, dtype=np.cdouble)
    k4 = np.zeros(nst, dtype=np.cdouble)
    kfunction = np.zeros(nst, dtype=np.cdouble)
    variation = np.zeros(nst, dtype=np.cdouble)
    c_dot = np.zeros(nst, dtype=np.cdouble)
    coef_new = np.zeros(nst, dtype=np.cdouble)
    eenergy = np.zeros(nst, dtype=np.double)
    dv = np.zeros((nst, nst), dtype=np.double)
    h = np.zeros((nst, nst), dtype=np.double)
    # frac = 1. / (nesteps - 1)
    frac = 1. / nesteps
    edt = dt * frac

    for iestep in range(0, nesteps):
        for ist in range(0, nst):
            eenergy[ist] = energy_old[ist] + (energy[ist] - energy_old[ist]) * iestep * frac
            for jst in range(0, nst):
                dv[ist, jst] = nacme_old[ist, jst] + (nacme[ist][jst] - nacme_old[ist, jst]) * iestep * frac
                h[ist, jst] = socme_old[ist, jst] + (socme[ist][jst] - socme_old[ist, jst]) * iestep * frac

        c_dot = cdot(nst, eenergy, dv, h, coef)

        k1 = edt * c_dot
        kfunction = 0.5 * k1
        coef_new = coef + kfunction

        print(iestep, "k1", coef_new, flush=True)

        c_dot = cdot(nst, eenergy, dv, h, coef_new)

        k2 = edt * c_dot
        kfunction = (np.sqrt(2.) - 1) * k1 / 2 + (1 - np.sqrt(2.) / 2) * k2
        coef_new = coef + kfunction

        print(iestep, "k2", coef_new, flush=True)

        c_dot = cdot(nst, eenergy, dv, h, coef_new)

        k3 = edt * c_dot
        kfunction = -np.sqrt(2.) * k2 / 2 + (1 + np.sqrt(2.) / 2) * k3
        coef_new = coef + kfunction

        print(iestep, "k3", coef_new, flush=True)

        c_dot = cdot(nst, eenergy, dv, h, coef_new)

        k4 = edt * c_dot
        variation = (k1 + (2 - np.sqrt(2.)) * k2 + (2 + np.sqrt(2.)) * k3 + k4) / 6
        coef_new = coef + variation

        print(iestep, "k4", coef_new, flush=True)

        norm = np.real(np.vdot(coef_new, coef_new))
        print("norm", norm, flush=True)
        
        coef = coef_new / np.sqrt(norm)
    
    return coef_new

def rk4_rho(nst, nesteps, dt, elec_object, energy, energy_old, nacme, nacme_old, socme, socme_old, rho):
    k1 = np.zeros((nst, nst), dtype=np.cdouble)
    k2 = np.zeros((nst, nst), dtype=np.cdouble)
    k3 = np.zeros((nst, nst), dtype=np.cdouble)
    k4 = np.zeros((nst, nst), dtype=np.cdouble)
    kfunction = np.zeros((nst, nst), dtype=np.cdouble)
    variation = np.zeros((nst, nst), dtype=np.cdouble)
    rho_dot = np.zeros((nst, nst), dtype=np.cdouble)
    rho_new = np.zeros((nst, nst), dtype=np.cdouble)
    eenergy = np.zeros(nst, dtype=np.double)
    dv = np.zeros((nst, nst), dtype=np.double)
    h = np.zeros((nst, nst), dtype=np.double)
    frac = 1. / (nesteps - 1)
    edt = dt * frac

    for iestep in range(0, nesteps):
        for ist in range(0, nst):
            eenergy[ist] = energy_old[ist] + (energy[ist] - energy_old[ist]) * iestep * frac
            for jst in range(0, nst):
                dv[ist, jst] = nacme_old[ist, jst] + (nacme[ist][jst] - nacme_old[ist, jst]) * iestep * frac
                h[ist, jst] = socme_old[ist, jst] + (socme[ist][jst] - socme_old[ist, jst]) * iestep * frac

        rhodot(nst, eenergy, dv, h, rho, rho_dot)
        
        k1 = edt * rho_dot
        kfunction = 0.5 * k1
        rho_new = rho + kfunction

        rhodot(nst, eenergy, dv, h, rho_new, rho_dot)
        k2 = edt * rho_dot
        kfunction = (np.sqrt(2) - 1) * k1 / 2 + (1 - np.sqrt(2) / 2) * k2
        rho_new = rho + kfunction

        rhodot(nst, eenergy, dv, h, rho_new, rho_dot)

        k3 = edt * rho_dot
        kfunction = -np.sqrt(2) * k2 / 2 + (1 + np.sqrt(2) / 2) * k3
        rho_new = rho + kfunction

        rhodot(nst, eenergy, dv, h, rho_new, rho_dot)

        k4 = edt * rho_dot
        variation = (k1 + (2 - np.sqrt(2)) * k2 + (2 + np.sqrt(2)) * k3 + k4) / 6
        rho += variation

    return rho


def el_run(
    program_state,
    elec_object,
):
    # Assign size variables
    nst = program_state.number_of_electronic_states
    nesteps = program_state.electronic_propogation_steps
    dt = program_state.step_size.as_atomic()

    nst = program_state.number_of_electronic_states
    if program_state.intersystem_crossing:
        socme_old = program_state.socmes[-2]
        socme = program_state.socmes[-1]
    else:
        socme_old = np.zeros((nst, nst), dtype=np.cdouble)
        socme = np.zeros((nst, nst), dtype=np.cdouble)

    # Propagate electrons
    if elec_object == "coefficient":
        program_state.state_coefficients = rk4_coef(
            nst,
            program_state.electronic_propogation_steps,
            dt,
            elec_object,
            program_state.state_energies[-1].as_hartree(),
            program_state.state_energies[-2].as_hartree(),
            program_state.nacmes[-1],
            program_state.nacmes[-2],
            socme,
            socme_old,
            program_state.state_coefficients,
        )

    elif elec_object == "density":
        program_state.rho = rk4_rho(
            nst,
            program_state.electronic_propogation_steps,
            dt,
            elec_object,
            program_state.state_energies[-1].as_hartree(),
            program_state.state_energies[-2].as_hartree(),
            program_state.nacmes[-1],
            program_state.nacmes[-2],
            socme,
            socme_old,
            program_state.rho,
        )
        
    

