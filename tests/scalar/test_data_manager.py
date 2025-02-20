import h5py
import numpy as np
from pathlib import Path

import pygpe.shared.data_manager_paths as dmp
from pygpe.scalar.data_manager import DataManager
from pygpe.scalar.wavefunction import ScalarWavefunction
from pygpe.shared.grid import Grid

FILENAME = "scalar_test.hdf5"
FILE_PATH = "."


def generate_wavefunction(
    points: tuple[int, int] = (64, 64),
    grid_spacing: tuple[float, float] = (0.5, 0.5),
) -> ScalarWavefunction:
    """Generates a 2D `Wavefunction` object for use in testing.

    :param points: Number of grid points in each dimension,
        defaults to (64, 64).
    :type points: tuple[int, int], optional
    :param grid_spacing: Grid spacing in each dimension, defaults to
        (0.5, 0.5).
    :type grid_spacing: tuple[float, float], optional.
    :return: `ScalarWavefunction` object.
    :rtype: ScalarWavefunction.
    """
    return ScalarWavefunction(Grid(points, grid_spacing))


def generate_parameters() -> dict:
    """Generates the scalar BEC parameters dictionary for use in testing.

    :return: The generated dictionary.
    :rtype: dict.
    """
    scalar_parameter_types = ["g", "trap", "nt", "dt", "t"]
    params = {}
    for key in scalar_parameter_types:
        params[key] = hash(key)

    return params


def test_data_manager_creation():
    """Tests whether the DataManager gets constructed without errors."""

    wavefunction = generate_wavefunction()
    params = generate_parameters()
    DataManager(FILENAME, FILE_PATH, wavefunction, params)

    Path.unlink(Path(f"{FILE_PATH}/{FILENAME}"))


def test_correct_parameters():
    """Tests whether condensate parameters are correctly saved to file."""

    wavefunction = generate_wavefunction()
    params = generate_parameters()
    DataManager(FILENAME, FILE_PATH, wavefunction, params)

    with h5py.File(f"{FILE_PATH}/{FILENAME}", "r") as file:
        for key, value in params.items():
            assert value == file[f"{dmp.PARAMETERS}/{key}"][...]

    Path.unlink(Path(f"{FILE_PATH}/{FILENAME}"))


def test_correct_wavefunction():
    """Tests whether the condensate wavefunction is correctly saved to file."""
    wavefunction = generate_wavefunction()
    params = generate_parameters()
    DataManager(FILENAME, FILE_PATH, wavefunction, params)

    with h5py.File(f"{FILE_PATH}/{FILENAME}", "r") as file:
        saved_wavefunction = file[f"{dmp.SCALAR_WAVEFUNCTION}"][:, :, 0]
        np.testing.assert_array_almost_equal(wavefunction.component, saved_wavefunction)

    Path.unlink(Path(f"{FILE_PATH}/{FILENAME}"))
