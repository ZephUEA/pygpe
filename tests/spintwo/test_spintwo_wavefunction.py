import cupy as cp
import pytest
from pygpe.shared.grid import Grid
from pygpe.spintwo.wavefunction import Wavefunction


def generate_wavefunction2d(
        points: tuple[int, int], grid_spacing: tuple[float, float]
) -> Wavefunction:
    """Generates and returns a Wavefunction2D object specified

    :param points: The number of grid points in the x and y dimension, respectively.
    :param grid_spacing: The spacing of grid points in the x and y dimension, respectively.
    :return: The Wavefunction2D object.
    """
    return Wavefunction(Grid(points, grid_spacing))


def test_uniaxial_initial_state():
    """Tests whether the uniaxial nematic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("UN", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.ones((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.zeros((64, 64), dtype="complex128")
    )


def test_biaxial_initial_state():
    """Tests whether the biaxial nematic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("BN", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component,
        1 / cp.sqrt(2.0) * cp.ones((64, 64), dtype="complex128"),
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component,
        1 / cp.sqrt(2.0) * cp.ones((64, 64), dtype="complex128"),
    )


def test_f2p_initial_state():
    """Tests whether the ferromagnetic-2 (with spin up) initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F2p", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component, cp.ones((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.zeros((64, 64), dtype="complex128")
    )


def test_f2m_initial_state():
    """Tests whether the ferromagnetic-2 (with spin down) initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F2m", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.ones((64, 64), dtype="complex128")
    )


def test_f1p_initial_state():
    """Tests whether the ferromagnetic-1 (with spin up) initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F1p", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.ones((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.zeros((64, 64), dtype="complex128")
    )


def test_f1m_initial_state():
    """Tests whether the ferromagnetic-1 (with spin down) initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1}
    wavefunction.set_ground_state("F1m", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component, cp.ones((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.zeros((64, 64), dtype="complex128")
    )


def test_cyclic_initial_state():
    """Tests whether the cyclic initial state is set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    params = {"n0": 1, "c2": 0.5, "p": 0.0, "q": 0}
    wavefunction.set_ground_state("cyclic", params)

    cp.testing.assert_array_equal(
        wavefunction.plus2_component,
        cp.sqrt(1 / 3) * cp.ones((64, 64), dtype="complex128"),
    )
    cp.testing.assert_array_equal(
        wavefunction.plus1_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.zero_component, cp.zeros((64, 64), dtype="complex128")
    )
    cp.testing.assert_array_equal(
        wavefunction.minus1_component,
        cp.sqrt(2 / 3) * cp.ones((64, 64), dtype="complex128"),
    )
    cp.testing.assert_array_equal(
        wavefunction.minus2_component, cp.zeros((64, 64), dtype="complex128")
    )


def test_custom_components():
    """Tests whether custom wavefunction components are set correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    plus2_comp = cp.ones(wavefunction.grid.shape, dtype="complex128")
    plus1_comp = cp.zeros(wavefunction.grid.shape, dtype="complex128")
    zero_comp = cp.zeros(wavefunction.grid.shape, dtype="complex128")
    minus1_comp = cp.sqrt(1 / 3) * cp.ones(wavefunction.grid.shape, dtype="complex128")
    minus2_comp = 5 * cp.ones(wavefunction.grid.shape, dtype="complex128")
    wavefunction.set_custom_components(
        plus2_comp, plus1_comp, zero_comp, minus1_comp, minus2_comp
    )

    cp.testing.assert_array_equal(wavefunction.plus2_component, plus2_comp)
    cp.testing.assert_array_equal(wavefunction.plus1_component, plus1_comp)
    cp.testing.assert_array_equal(wavefunction.zero_component, zero_comp)
    cp.testing.assert_array_equal(wavefunction.minus1_component, minus1_comp)
    cp.testing.assert_array_equal(wavefunction.minus2_component, minus2_comp)


def test_adding_noise_list():
    """Tests whether adding noise to specified empty components correctly
    makes those components non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    zeros = cp.zeros(wavefunction.grid.shape, dtype='complex128')
    wavefunction.set_custom_components(cp.zeros_like(zeros), cp.zeros_like(zeros), cp.zeros_like(zeros),
                                       cp.zeros_like(zeros), cp.zeros_like(zeros))
    wavefunction.add_noise_to_components(["plus2", "plus1"], 0, 1e-2)

    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.plus2_component, cp.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.plus1_component, cp.zeros(wavefunction.grid.shape)
        )


def test_adding_noise_all():
    """Tests whether adding noise to all empty components correctly
    makes those components non-zero.
    """
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    zeros = cp.zeros(wavefunction.grid.shape, dtype='complex128')
    wavefunction.set_custom_components(cp.zeros_like(zeros), cp.zeros_like(zeros), cp.zeros_like(zeros),
                                       cp.zeros_like(zeros), cp.zeros_like(zeros))
    wavefunction.add_noise_to_components("all", 0, 1e-2)

    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.plus2_component, cp.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.plus1_component, cp.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.zero_component, cp.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.minus1_component, cp.zeros(wavefunction.grid.shape)
        )
    with pytest.raises(AssertionError):
        cp.testing.assert_array_equal(
            wavefunction.minus2_component, cp.zeros(wavefunction.grid.shape)
        )


def test_phase_all():
    """Tests that a phase applied to all components is applied correctly."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_custom_components(
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
    )

    phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
    wavefunction.apply_phase(phase, "all")

    cp.testing.assert_allclose(cp.angle(wavefunction.plus2_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.plus1_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.zero_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.minus1_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.minus2_component), phase)


def test_phase_multiple_components():
    """Tests that a phase is applied correctly to multiple components."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_custom_components(
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
    )

    phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
    wavefunction.apply_phase(phase, ["plus1", "minus1"])

    cp.testing.assert_allclose(cp.angle(wavefunction.plus1_component), phase)
    cp.testing.assert_allclose(cp.angle(wavefunction.minus1_component), phase)


def test_phase_single():
    """Tests that a phase is applied correctly to a single component."""
    wavefunction = generate_wavefunction2d((64, 64), (0.5, 0.5))
    wavefunction.set_custom_components(
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
        cp.ones((64, 64), dtype="complex128"),
    )

    phase = cp.random.uniform(size=(64, 64), dtype=cp.float64)
    wavefunction.apply_phase(phase, "zero")

    cp.testing.assert_allclose(cp.angle(wavefunction.zero_component), phase)
