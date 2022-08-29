import cupy as cp
from pygpe.spintwo.wavefunction import Wavefunction


def step_wavefunction(wfn: Wavefunction, params: dict) -> None:
    """Propagates the wavefunction forward one time step.

    :param wfn: The wavefunction of the system.
    :param params: The parameters of the system.
    """
    _kinetic_step(wfn, params)
    wfn.ifft()
    _interaction_step(wfn, params)
    wfn.fft()
    _kinetic_step(wfn, params)
    if isinstance(params["dt"], complex):
        _renormalise_wavefunction(wfn)


def _kinetic_step(wfn: Wavefunction, pm: dict) -> None:
    """Computes the kinetic energy subsystem for half a time step.

    :param wfn: The wavefunction of the system.
    :param pm:  The parameters dictionary.
    """
    wfn.fourier_plus2_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)
    wfn.fourier_plus1_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)
    wfn.fourier_zero_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)
    wfn.fourier_minus1_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)
    wfn.fourier_minus2_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)


def _interaction_step(wfn: Wavefunction, pm: dict) -> None:
    # Calculate density and singlets
    n = _density(wfn)
    a20 = _singlet_duo(wfn)

    # Perform singlet step
    temp_wfn = _evolve_spin_singlet(wfn, n, a20, pm)

    # Calculate spin
    fp, fz = _calculate_spin_vectors(temp_wfn)
    mod_f = cp.sqrt(fz ** 2 + abs(fp) ** 2)

    # Evolve spin term c2 * F^2
    qfactor, q2factor, q3factor, q4factor = _calc_q_factors(mod_f, pm)

    fzq = cp.nan_to_num(fz / mod_f)
    fpq = cp.nan_to_num(fp / mod_f)

    qpsi = _calc_qpsi(fzq, fpq, temp_wfn)
    q2psi = _calc_qpsi(fzq, fpq, qpsi)
    q3psi = _calc_qpsi(fzq, fpq, q2psi)
    q4psi = _calc_qpsi(fzq, fpq, q3psi)

    for ii in range(len(temp_wfn)):
        temp_wfn[ii] += qfactor * qpsi[ii] + q2factor * q2psi[ii] + q3factor * q3psi[ii] + q4factor * q4psi[ii]

    # Evolve (c0+c4)*n + (V + pm + qm^2):
    for ii in range(len(temp_wfn)):
        m_f = 2 - ii  # Current spin component
        temp_wfn[ii] *= cp.exp(-1j * pm["dt"] * ((pm["c0"] + pm["c4"]) * n + pm["trap"] - pm["p"] * m_f
                                                 + pm["q"] * m_f ** 2))

    # Update wavefunction arrays
    wfn.plus2_component = temp_wfn[0]
    wfn.plus1_component = temp_wfn[1]
    wfn.zero_component = temp_wfn[2]
    wfn.minus1_component = temp_wfn[3]
    wfn.minus2_component = temp_wfn[4]


def _density(wfn: Wavefunction) -> cp.ndarray:
    return abs(wfn.plus2_component) ** 2 + abs(wfn.plus1_component) ** 2 + abs(wfn.zero_component) ** 2 \
           + abs(wfn.minus1_component) ** 2 + abs(wfn.minus2_component) ** 2


def _singlet_duo(wfn: Wavefunction) -> cp.ndarray:
    return 1 / cp.sqrt(5) * (wfn.zero_component ** 2 - 2 * wfn.plus1_component * wfn.minus1_component
                             + 2 * wfn.plus2_component * wfn.minus2_component)


def _evolve_spin_singlet(wfn: Wavefunction, dens: cp.ndarray, singlet: cp.ndarray, pm: dict) -> list[cp.ndarray]:
    s = cp.nan_to_num(cp.sqrt(dens ** 2 - abs(singlet) ** 2))
    cos_term = cp.cos(pm["c4"] * s * pm["dt"])
    sin_term = cp.sin(pm["c4"] * s * pm["dt"]) / s
    sin_term[s == 0] = 0  # Corrects division by 0

    psi_p2 = wfn.plus2_component * cos_term + 1j * (dens * wfn.plus2_component
                                                    - singlet * cp.conj(wfn.minus2_component)) * sin_term
    psi_p1 = wfn.plus1_component * cos_term + 1j * (dens * wfn.plus1_component
                                                    + singlet * cp.conj(wfn.minus1_component)) * sin_term
    psi_0 = wfn.zero_component * cos_term + 1j * (dens * wfn.zero_component
                                                  - singlet * cp.conj(wfn.zero_component)) * sin_term
    psi_m1 = wfn.minus1_component * cos_term + 1j * (dens * wfn.minus1_component
                                                     + singlet * cp.conj(wfn.plus1_component)) * sin_term
    psi_m2 = wfn.minus2_component * cos_term + 1j * (dens * wfn.minus2_component
                                                     - singlet * cp.conj(wfn.plus2_component)) * sin_term

    return [psi_p2, psi_p1, psi_0, psi_m1, psi_m2]


def _calculate_spin_vectors(wfn: list[cp.ndarray]):
    fp = cp.sqrt(6) * (wfn[1] * cp.conj(wfn[2]) + wfn[2] * cp.conj(wfn[3])) + 2 * (wfn[3] * cp.conj(wfn[4]) + wfn[0]
                                                                                   * cp.conj(wfn[1]))
    fz = 2 * (abs(wfn[0]) ** 2 - abs(wfn[4]) ** 2) + abs(wfn[1]) ** 2 - abs(wfn[3]) ** 2
    return fp, fz


def _calc_q_factors(mod_f: cp.ndarray, pm: dict):
    cos1, sin1 = cp.cos(pm["c2"] * mod_f * pm["dt"]), cp.sin(pm["c2"] * mod_f * pm["dt"])
    cos2, sin2 = cp.cos(2 * pm["c2"] * mod_f * pm["dt"]), cp.sin(2 * pm["c2"] * mod_f * pm["dt"])
    qfactor = 1j * (-4 / 3 * sin1 + 1 / 6 * sin2)
    q2factor = (-5 / 4 + 4 / 3 * cos1 - 1 / 12 * cos2)
    q3factor = 1j * (1 / 3 * sin1 - 1 / 6 * sin2)
    q4factor = (1 / 4 - 1 / 3 * cos1 + 1 / 12 * cos2)

    return qfactor, q2factor, q3factor, q4factor


def _calc_qpsi(fz, fp, wfn):
    qpsi = [2 * fz * wfn[0] + fp * wfn[1],
            cp.conj(fp) * wfn[0] + fz * wfn[1] + cp.sqrt(3 / 2) * fp * wfn[2],
            cp.sqrt(3 / 2) * (cp.conj(fp) * wfn[1] + fp * wfn[3]),
            cp.sqrt(3 / 2) * cp.conj(fp) * wfn[2] - fz * wfn[3] + fp * wfn[4],
            cp.conj(fp) * wfn[3] - 2 * fz * wfn[4]]

    return qpsi


def _renormalise_wavefunction(wfn: Wavefunction) -> None:
    """Re-normalises the wavefunction to the correct atom number.

    :param wfn: The wavefunction of the system.
    """
    wfn.ifft()
    correct_atom_plus2 = wfn.atom_num_plus2
    correct_atom_plus1 = wfn.atom_num_plus1
    correct_atom_zero = wfn.atom_num_zero
    correct_atom_minus1 = wfn.atom_num_minus1
    correct_atom_minus2 = wfn.atom_num_minus2

    current_atom_plus2, current_atom_plus1, current_atom_zero, current_atom_minus1, current_atom_minus2 \
        = _calculate_atom_num(wfn)

    wfn.plus2_component *= cp.sqrt(correct_atom_plus2 / current_atom_plus2)
    wfn.plus1_component *= cp.sqrt(correct_atom_plus1 / current_atom_plus1)
    wfn.zero_component *= cp.sqrt(correct_atom_zero / current_atom_zero)
    wfn.minus1_component *= cp.sqrt(correct_atom_minus1 / current_atom_minus1)
    wfn.minus2_component *= cp.sqrt(correct_atom_minus2 / current_atom_minus2)
    wfn.fft()


def _calculate_atom_num(wfn: Wavefunction) -> tuple[int, int, int, int, int]:
    """Calculates the atom number of each wavefunction component.

    :param wfn: The wavefunction of the system.
    :return: The atom numbers of the plus two, plus one, zero, minus one, and minus two
    components, respectively.
    """
    atom_num_plus2 = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.plus2_component) ** 2)
    atom_num_plus1 = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.plus1_component) ** 2)
    atom_num_zero = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.zero_component) ** 2)
    atom_num_minus1 = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.minus1_component) ** 2)
    atom_num_minus2 = wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.minus2_component) ** 2)

    return atom_num_plus2, atom_num_plus1, atom_num_zero, atom_num_minus1, atom_num_minus2
