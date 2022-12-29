import cupy as cp
from pygpe.scalar.wavefunction import Wavefunction


def step_wavefunction(wfn: Wavefunction, params: dict) -> None:
    """Propagates the wavefunction forward one time step.

    :param wfn: The wavefunction of the system.
    :type wfn: :class:`Wavefunction`
    :param params: The parameters of the system.
    :type params: dict
    """
    _kinetic_step(wfn, params)
    wfn.ifft()
    _potential_step(wfn, params)
    wfn.fft()
    _kinetic_step(wfn, params)
    if isinstance(params["dt"], complex):
        _renormalise_wavefunction(wfn)


def _kinetic_step(wfn: Wavefunction, pm: dict) -> None:
    """Computes the kinetic energy subsystem for half a time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameters' dictionary.
    """
    wfn.fourier_component *= cp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)


def _potential_step(wfn: Wavefunction, pm: dict) -> None:
    """Computes the potential subsystem for a full time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameters' dictionary.
    """
    wfn.component *= cp.exp(
        -1j * pm["dt"] * (pm["trap"] + pm["g"] * cp.abs(wfn.component) ** 2)
    )


def _renormalise_wavefunction(wfn: Wavefunction) -> None:
    """Re-normalises the wavefunction to the correct atom number.

    :param wfn: The wavefunction of the system.
    """
    wfn.ifft()
    correct_atom_num = wfn.atom_num
    current_atom_num = _calculate_atom_num(wfn)
    wfn.component *= cp.sqrt(correct_atom_num / current_atom_num)
    wfn.fft()


def _calculate_atom_num(wfn: Wavefunction) -> float:
    """Calculates the current atom number of the wavefunction.

    :param wfn: The wavefunction of the system.
    :return: The atom number.
    """
    return wfn.grid.grid_spacing_product * cp.sum(cp.abs(wfn.component) ** 2)
