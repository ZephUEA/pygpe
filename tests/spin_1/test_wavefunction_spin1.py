import unittest
import cupy as cp
from typing import Tuple
from pygpe.shared.grid import Grid2D
from pygpe.spin_1.wavefunction import Wavefunction2D


def generate_wavefunction2d(points: Tuple[int, int], grid_spacing: Tuple[float, float]) -> Wavefunction2D:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension, respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension, respectively.
    :return: The Wavefunction2D object.
    """
    return Wavefunction2D(Grid2D(points, grid_spacing))


class TestWavefunction2D(unittest.TestCase):

    def test_polar_initial_state(self):
        """Tests whether the polar initial state is set correctly."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.set_initial_state("Polar")

        self.assertEqual(wavefunction.plus_component.all(), 0.)
        self.assertEqual(wavefunction.zero_component.all(), 1.)
        self.assertEqual(wavefunction.minus_component.all(), 0.)

    def test_empty_initial_state(self):
        """Tests whether the empty initial state correctly sets
        all wavefunction components to zero.
        """
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.set_initial_state("empty")

        self.assertEqual(wavefunction.plus_component.all(), 0.)
        self.assertEqual(wavefunction.zero_component.all(), 0.)
        self.assertEqual(wavefunction.minus_component.all(), 0.)

    def test_set_initial_state_raises_error(self):
        """Tests that an unsupported/invalid initial state returns an error."""
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        with self.assertRaises(ValueError):
            wavefunction.set_initial_state("garbage")

    def test_fft_normalised(self):
        """Tests whether performing a forward followed by a backwards
        fast Fourier transform on the wavefunction retains the same input.
        """
        wavefunction_1 = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction_1.set_initial_state("polar")
        wavefunction_2 = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction_2.set_initial_state("polar")

        wavefunction_2.fft()
        wavefunction_2.ifft()

        # assert_array_equal returns None if arrays are equal
        self.assertIsNone(cp.testing.assert_array_equal(wavefunction_1.plus_component, wavefunction_2.plus_component))
        self.assertIsNone(cp.testing.assert_array_equal(wavefunction_1.zero_component, wavefunction_2.zero_component))
        self.assertIsNone(cp.testing.assert_array_equal(wavefunction_1.minus_component, wavefunction_2.minus_component))

    def test_adding_noise_outer(self):
        """Tests whether adding noise to empty outer components correctly
        makes those components non-zero.
        """
        wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
        wavefunction.set_initial_state("empty")
        wavefunction.add_noise_to_components("outer", 0, 1e-2)

        self.assertNotEqual(wavefunction.plus_component.all(), 0.)
        self.assertEqual(wavefunction.zero_component.all(), 0.)
        self.assertNotEqual(wavefunction.minus_component.all(), 0.)
