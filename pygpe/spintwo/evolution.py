try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp
from pygpe.spintwo.wavefunction import SpinTwoWavefunction

pauliX = cp.array([[0,1,0,0,0],[1,0,cp.sqrt(3/2),0,0],[0,cp.sqrt(3/2),0,cp.sqrt(3/2),0],[0,0,cp.sqrt(3/2),0,1],[0,0,0,1,0]], dtype='complex128')
pauliY = 1j * cp.array([[0,-1,0,0,0],[1,0,-cp.sqrt(3/2),0,0],[0,cp.sqrt(3/2),0,-cp.sqrt(3/2),0],[0,0,cp.sqrt(3/2),0,-1],[0,0,0,1,0]], dtype='complex128')
pauliZ = cp.array([[2,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,-1,0],[0,0,0,0,-2]], dtype='complex128')
paulis = [pauliX,pauliY,pauliZ]

def step_wavefunction(wfn: SpinTwoWavefunction, params: dict) -> None:
    """Propagates the wavefunction forward one time step.

    :param wfn: The wavefunction of the system.
    :type wfn: :class:`Wavefunction`
    :param params: The parameters of the system.
    :type params: dict
    """
    _kinetic_step(wfn, params)
    wfn.ifft()
    _interaction_step(wfn, params)
    wfn.fft()
    _kinetic_step(wfn, params)
    if isinstance(params["dt"], complex):
        _renormalise_wavefunction(wfn)


def _kinetic_step(wfn: SpinTwoWavefunction, pm: dict) -> None:
    """Computes the kinetic energy subsystem for half a time step.

    :param wfn: The wavefunction of the system.
    :param pm:  The parameters' dictionary.
    """
    wfn.fourier_plus2_component *= cp.exp(-0.25 * 1j * pm["dt"] * (wfn.grid.wave_number+8*pm['q']))
    wfn.fourier_plus1_component *= cp.exp(-0.25 * 1j * pm["dt"] * (wfn.grid.wave_number+2*pm['q']))
    wfn.fourier_zero_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)
    wfn.fourier_minus1_component *= cp.exp(-0.25 * 1j * pm["dt"] * (wfn.grid.wave_number+2*pm['q']))
    wfn.fourier_minus2_component *= cp.exp(-0.25 * 1j * pm["dt"] * (wfn.grid.wave_number+8*pm['q']))


def _interaction_step(wfn: SpinTwoWavefunction, pm: dict) -> None:
    wfnArray = cp.array([wfn.plus2_component,wfn.plus1_component,wfn.zero_component,wfn.minus1_component,wfn.minus2_component])
    # Calculate density and singlets
    n = _density(wfn)
    a20 = _singlet_duo(wfn)
    # fx, fy, fz = _calculate_spin_vectors(wfnArray)

    # Perform singlet step
    # temp_wfn = wfnArray
    temp_wfn = _evolve_spin_singlet(wfn, n, a20, pm)

    # Calculate spin
    
    fz = 2 * ( abs(wfnArray[0])**2 - abs(wfnArray[4])**2 ) + abs(wfnArray[1])**2 - abs(wfnArray[3])**2
    fp =  2*( cp.conj(wfnArray[1])*wfnArray[0]  + cp.conj(wfnArray[4])*wfnArray[3] ) + cp.sqrt(6) * ( cp.conj(wfnArray[2])*wfnArray[1] + cp.conj(wfnArray[3])*wfnArray[2] )
    mod_f = cp.sqrt( fz**2 + abs(fp)**2 )
    # Evolve spin term c2 * F^2

    q1factor, q2factor, q3factor, q4factor = _calc_q_factors(mod_f, pm)

    # fDotF = pauliX[...,None,None] * normalFs[0] + pauliY[...,None,None] * normalFs[1] + pauliZ[...,None,None] * normalFs[2]
    # fDotF2 = cp.einsum('ij...,jk...->ik...', fDotF, fDotF)
    # fDotF3 = cp.einsum('ij...,jk...->ik...', fDotF2, fDotF)
    # fDotF4 = cp.einsum('ij...,jk...->ik...', fDotF3, fDotF)

    # spinTerm = cp.identity(5)[...,None,None] + fDotF*q1factor + fDotF2*q2factor + fDotF3*q3factor + fDotF4*q4factor
    # temp_wfn = cp.einsum( 'ij...,j...->i...', spinTerm, temp_wfn )

    fzNorm = cp.nan_to_num(fz / mod_f)
    fPerpNorm = cp.nan_to_num(fp / mod_f)

    qpsi = _calc_qpsi(fzNorm, fPerpNorm, temp_wfn)
    q2psi = _calc_qpsi(fzNorm, fPerpNorm, qpsi)
    q3psi = _calc_qpsi(fzNorm, fPerpNorm, q2psi)
    q4psi = _calc_qpsi(fzNorm, fPerpNorm, q3psi)

    for ii in range(len(temp_wfn)):
        temp_wfn[ii] += (
            q1factor * qpsi[ii]
            + q2factor * q2psi[ii]
            + q3factor * q3psi[ii]
            + q4factor * q4psi[ii]
        )




    # Evolve (c0+c4)*n + (V + pm + qm^2):
    for ii in range(len(temp_wfn)):
        m_f = 2 - ii  # Current spin component
        temp_wfn[ii] *= cp.exp(
            -1j
            * pm["dt"]
            * (
                ( pm["c0"] + pm['c4']/5 ) * n
                + pm["trap"]
                - pm["p"] * m_f
            )
        )

    # Update wavefunction arrays
    wfn.plus2_component = temp_wfn[0]
    wfn.plus1_component = temp_wfn[1]
    wfn.zero_component = temp_wfn[2]
    wfn.minus1_component = temp_wfn[3]
    wfn.minus2_component = temp_wfn[4]


def _density(wfn: SpinTwoWavefunction) -> cp.ndarray:
    return (
        abs(wfn.plus2_component) ** 2
        + abs(wfn.plus1_component) ** 2
        + abs(wfn.zero_component) ** 2
        + abs(wfn.minus1_component) ** 2
        + abs(wfn.minus2_component) ** 2
    )


def _singlet_duo(wfn: SpinTwoWavefunction) -> cp.ndarray:
    # Missing a sqrt(5) since its wrapped into a redefinition of pm['c4']
    # This parameter gets divided by 5 everytime it shows up due to this.
    return (
            wfn.zero_component**2
            - 2 * wfn.plus1_component * wfn.minus1_component
            + 2 * wfn.plus2_component * wfn.minus2_component
    )


def _evolve_spin_singlet(
    wfn: SpinTwoWavefunction, dens: cp.ndarray, singlet: cp.ndarray, pm: dict
) -> list[cp.ndarray]:
    s = cp.nan_to_num( cp.sqrt(dens**2 - (abs(singlet) ** 2) ) )
    cos_term = cp.cos(pm["c4"]/5 * s * pm["dt"])
    sin_term = cp.sin(pm["c4"]/5 * s * pm["dt"]) / s
    sin_term[s == 0] = 0  # Corrects division by 0

    psi_p2 = (
        wfn.plus2_component * cos_term
        + 1j
        * (dens * wfn.plus2_component - singlet * cp.conj(wfn.minus2_component))
        * sin_term
    )
    psi_p1 = (
        wfn.plus1_component * cos_term
        + 1j
        * (dens * wfn.plus1_component + singlet * cp.conj(wfn.minus1_component))
        * sin_term
    )
    psi_0 = (
        wfn.zero_component * cos_term
        + 1j
        * (dens * wfn.zero_component - singlet * cp.conj(wfn.zero_component))
        * sin_term
    )
    psi_m1 = (
        wfn.minus1_component * cos_term
        + 1j
        * (dens * wfn.minus1_component + singlet * cp.conj(wfn.plus1_component))
        * sin_term
    )
    psi_m2 = (
        wfn.minus2_component * cos_term
        + 1j
        * (dens * wfn.minus2_component - singlet * cp.conj(wfn.plus2_component))
        * sin_term
    )

    return [psi_p2, psi_p1, psi_0, psi_m1, psi_m2]


def _calculate_spin_vectors(wfn: cp.ndarray):
    conjWfn = cp.conj(wfn)
    fx = cp.einsum('kij,kl,lij->ij',conjWfn,pauliX,wfn )
    fy = cp.einsum('kij,kl,lij->ij',conjWfn,pauliY,wfn )
    fz = 2 * ( abs(wfn[0])**2 - abs(wfn[4])**2 ) + abs(wfn[1])**2 - abs(wfn[3])**2
    return fx, fy, fz


def _calc_q_factors(mod_f: cp.ndarray, pm: dict):
    cos1, sin1 = cp.cos(pm["c2"] * mod_f * pm["dt"]), cp.sin(
        pm["c2"] * mod_f * pm["dt"]
    )
    cos2, sin2 = cp.cos(2 * pm["c2"] * mod_f * pm["dt"]), cp.sin(
        2 * pm["c2"] * mod_f * pm["dt"]
    )
    qfactor = 1j * (-4 / 3 * sin1 + 1 / 6 * sin2)
    q2factor = -5 / 4 + 4 / 3 * cos1 - 1 / 12 * cos2
    q3factor = 1j * (1 / 3 * sin1 - 1 / 6 * sin2)
    q4factor = 1 / 4 - 1 / 3 * cos1 + 1 / 12 * cos2

    return qfactor, q2factor, q3factor, q4factor


def _calc_qpsi(fz, fp, wfn):
    # fp = fx - i fy
    qpsi = [
        2 * fz * wfn[0] + fp * wfn[1],
        cp.conj(fp) * wfn[0] + fz * wfn[1] + cp.sqrt(3 / 2) * fp * wfn[2],
        cp.sqrt(3 / 2) * (cp.conj(fp) * wfn[1] + fp * wfn[3]),
        cp.sqrt(3 / 2) * cp.conj(fp) * wfn[2] - fz * wfn[3] + fp * wfn[4],
        cp.conj(fp) * wfn[3] - 2 * fz * wfn[4],
    ]

    return qpsi


def _renormalise_wavefunction(wfn: SpinTwoWavefunction) -> None:
    """Re-normalises the wavefunction to the correct atom number and magnetisation.
    This is based off a projection onto a polyhedron.
    :param wfn: The wavefunction of the system.
    """
    wfn.ifft()
    correctAtomNum = wfn.atom_num_plus2 + wfn.atom_num_plus1+ wfn.atom_num_zero + wfn.atom_num_minus1 + wfn.atom_num_minus2
    correctMagNum = 2 * (wfn.atom_num_plus2 - wfn.atom_num_minus2) + (wfn.atom_num_plus1 - wfn.atom_num_minus1)
    currentPlus2, currentPlus1, currentZero, currentMinus1, currentMinus2 = _calculate_atom_num( wfn )
    currentAtomNum = currentPlus2 + currentPlus1 + currentZero + currentMinus1 + currentMinus2
    currentMagNum = 2 * ( currentPlus2 - currentMinus2 ) + (currentPlus1 - currentMinus1 )

    if correctAtomNum == currentAtomNum and correctMagNum == currentMagNum:
        wfn.fft()
        return
    
    p1 = ( 4 * correctAtomNum + 2 * correctMagNum - 8 * currentPlus2 + 14 * currentPlus1 - 4 * currentZero - 2 * currentMinus1  ) / 40
    p2 = ( 8 * correctAtomNum - 8 * currentPlus2 + 2 * currentPlus1 + 12 * currentZero + 2 * currentMinus1 - 8 * currentMinus2  ) / 40
    p3 = ( 4 * correctAtomNum - 2 * correctMagNum - 2 * currentPlus1 - 4 * currentZero + 14 * currentMinus1 - 8 * currentMinus2 ) / 40


    if [p1,p2,p3] not in wfn.polytope.polyhedron: # This deals with if componets become empty
        p1,p2,p3 = wfn.polytope.project([p1,p2,p3])

    correctPlus2 = (2*correctAtomNum + correctMagNum)/4 - p1 - p2
    correctPlus1 = 2 * p1
    correctZero = 2 * p2 - p1 - p3
    correctMinus1 = 2 * p3
    correctMinus2 = (2*correctAtomNum - correctMagNum)/4 - p2 - p3

    wfn.plus2_component *= cp.sqrt(correctPlus2/currentPlus2)
    wfn.plus1_component *= cp.sqrt(correctPlus1/currentPlus1)
    wfn.zero_component *= cp.sqrt(correctZero/currentZero)
    wfn.minus1_component *= cp.sqrt(correctMinus1/currentMinus1)
    wfn.minus2_component *= cp.sqrt(correctMinus2/currentMinus2)

    wfn.fft()


def _calculate_atom_num(
    wfn: SpinTwoWavefunction,
) -> float:
    """Calculates the total atom number of the system.

    :param wfn: The wavefunction of the system.
    :return: The total atom number.
    """
    atom_num_plus2 = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.plus2_component) ** 2
    )
    atom_num_plus1 = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.plus1_component) ** 2
    )
    atom_num_zero = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.zero_component) ** 2
    )
    atom_num_minus1 = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.minus1_component) ** 2
    )
    atom_num_minus2 = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.minus2_component) ** 2
    )

    return atom_num_plus2, atom_num_plus1, atom_num_zero, atom_num_minus1, atom_num_minus2
