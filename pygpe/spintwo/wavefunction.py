from pygpe.shared.grid import Grid
import cupy as cp


class Wavefunction:

    def __init__(self, grid: Grid):
        self.grid = grid

        self.plus2_component = cp.empty(grid.shape, dtype='complex128')
        self.plus1_component = cp.empty(grid.shape, dtype='complex128')
        self.zero_component = cp.empty(grid.shape, dtype='complex128')
        self.minus1_component = cp.empty(grid.shape, dtype='complex128')
        self.minus2_component = cp.empty(grid.shape, dtype='complex128')
        self.fourier_plus2_component = cp.empty(grid.shape, dtype='complex128')
        self.fourier_plus1_component = cp.empty(grid.shape, dtype='complex128')
        self.fourier_zero_component = cp.empty(grid.shape, dtype='complex128')
        self.fourier_minus1_component = cp.empty(grid.shape, dtype='complex128')
        self.fourier_minus2_component = cp.empty(grid.shape, dtype='complex128')

        self.atom_num_plus2 = 0
        self.atom_num_plus1 = 0
        self.atom_num_zero = 0
        self.atom_num_minus1 = 0
        self.atom_num_minus2 = 0

    def set_ground_state(self, ground_state: str, params: dict) -> None:
        """Sets the components of the wavefunction according to
        the ground state we wish to be in.

        :param ground_state: The ground state of the wavefunction.
        :param params: Dictionary containing condensate parameters.
        """
        ground_states = {
            "UN": _uniaxial_initial_state,
            "BN": _biaxial_initial_state,
            "F2p": _ferromagnetic2p_initial_state,
            "F2m": _ferromagnetic2m_initial_state,
            "F1p": _ferromagnetic1p_initial_state,
            "F1m": _ferromagnetic1m_initial_state,
            "cyclic": _cyclic_initial_state
        }

        ground_states[ground_state.lower()](self, params)

        self._update_atom_numbers()

    def set_custom_components(self, plus2_component: cp.ndarray = None, plus1_component: cp.ndarray = None,
                              zero_component: cp.ndarray = None, minus1_component: cp.ndarray = None,
                              minus2_component: cp.ndarray = None) -> None:
        """Sets the wavefunction components to the specified arrays.

        :param plus2_component: Plus two component of the wavefunction.
        :param plus1_component: Plus one component of the wavefunction.
        :param zero_component: Zero component of the wavefunction.
        :param minus1_component: Minus one component of the wavefunction.
        :param minus2_component: Minus two component of the wavefunction.
        """
        if plus2_component is not None:
            self.plus2_component = plus2_component
        if plus1_component is not None:
            self.plus1_component = plus1_component
        if zero_component is not None:
            self.zero_component = zero_component
        if minus1_component is not None:
            self.minus1_component = minus1_component
        if minus2_component is not None:
            self.minus2_component = minus2_component

    def add_noise_to_components(self, components: str, mean: float, std_dev: float) -> None:
        """Adds noise to the specified wavefunction components
        using a normal distribution.

        :param components: "outer", "inner", "outer+inner", "middle", or "all": The components to add noise to.
        :param mean: The mean of the normal distribution.
        :param std_dev: The standard deviation of the normal distribution.
        """
        if components.lower() == "outer":
            self.plus2_component += self._generate_complex_normal_dist(mean, std_dev)
            self.minus2_component += self._generate_complex_normal_dist(mean, std_dev)
        elif components.lower() == "inner":
            self.plus1_component += self._generate_complex_normal_dist(mean, std_dev)
            self.minus1_component += self._generate_complex_normal_dist(mean, std_dev)
        elif components.lower() == "outer+inner":
            self.plus2_component += self._generate_complex_normal_dist(mean, std_dev)
            self.plus1_component += self._generate_complex_normal_dist(mean, std_dev)
            self.minus1_component += self._generate_complex_normal_dist(mean, std_dev)
            self.minus2_component += self._generate_complex_normal_dist(mean, std_dev)
        elif components.lower() == "middle":
            self.zero_component += self._generate_complex_normal_dist(mean, std_dev)
        elif components.lower() == "all":
            self.plus2_component += self._generate_complex_normal_dist(mean, std_dev)
            self.plus1_component += self._generate_complex_normal_dist(mean, std_dev)
            self.zero_component += self._generate_complex_normal_dist(mean, std_dev)
            self.minus1_component += self._generate_complex_normal_dist(mean, std_dev)
            self.minus2_component += self._generate_complex_normal_dist(mean, std_dev)
        else:
            raise ValueError(f"{components} is not a supported configuration")

        self._update_atom_numbers()

    def _generate_complex_normal_dist(self, mean: float, std_dev: float) -> cp.ndarray:
        """Returns a ndarray of complex values containing results from
        a normal distribution.
        """
        return cp.random.normal(mean, std_dev, size=self.grid.shape) + 1j * cp.random.normal(mean, std_dev,
                                                                                             size=self.grid.shape)

    def apply_phase(self, phase: cp.ndarray, components: str | list[str] = 'all') -> None:
        """Applies a phase to specified components.

        :param phase: The phase to be applied.
        :param components: The components of which to apply the phase to.
        """
        if isinstance(components, list):
            for component in components:
                self._apply_phase_to_component(phase, component)
        elif components == 'all':
            for component in ['plus2', 'plus1', 'zero', 'minus1', 'minus2']:
                self._apply_phase_to_component(phase, component)
        elif isinstance(components, str):
            self._apply_phase_to_component(phase, components)
        else:
            raise ValueError(f'Components type {components} is unsupported')

    def _apply_phase_to_component(self, phase: cp.ndarray, component: str):
        if component == 'plus2':
            self.plus2_component *= cp.exp(1j * phase)
        elif component == 'plus1':
            self.plus1_component *= cp.exp(1j * phase)
        elif component == 'zero':
            self.zero_component *= cp.exp(1j * phase)
        elif component == 'minus1':
            self.minus1_component *= cp.exp(1j * phase)
        elif component == 'minus2':
            self.minus1_component *= cp.exp(1j * phase)
        else:
            raise ValueError(f'Component type {component} is unsupported')

    def _update_atom_numbers(self):
        self.atom_num_plus2 = self.grid.grid_spacing_product * cp.sum(cp.abs(self.plus2_component) ** 2)
        self.atom_num_plus1 = self.grid.grid_spacing_product * cp.sum(cp.abs(self.plus1_component) ** 2)
        self.atom_num_zero = self.grid.grid_spacing_product * cp.sum(cp.abs(self.zero_component) ** 2)
        self.atom_num_minus1 = self.grid.grid_spacing_product * cp.sum(cp.abs(self.minus1_component) ** 2)
        self.atom_num_minus2 = self.grid.grid_spacing_product * cp.sum(cp.abs(self.minus2_component) ** 2)

    def fft(self):
        """Fourier transforms real-space components and updates Fourier-space components."""
        self.fourier_plus2_component = cp.fft.fftn(self.plus2_component)
        self.fourier_plus1_component = cp.fft.fftn(self.plus1_component)
        self.fourier_zero_component = cp.fft.fftn(self.zero_component)
        self.fourier_minus1_component = cp.fft.fftn(self.minus1_component)
        self.fourier_minus2_component = cp.fft.fftn(self.minus2_component)

    def ifft(self):
        """Inverse Fourier transforms Fourier-space components and updates real-space components."""
        self.plus2_component = cp.fft.ifftn(self.fourier_plus2_component)
        self.plus1_component = cp.fft.ifftn(self.fourier_plus1_component)
        self.zero_component = cp.fft.ifftn(self.fourier_zero_component)
        self.minus1_component = cp.fft.ifftn(self.fourier_minus1_component)
        self.minus2_component = cp.fft.ifftn(self.fourier_minus2_component)


def _uniaxial_initial_state(wfn: Wavefunction, params: dict):
    """Sets wavefunction components to uniaxial nematic state."""
    wfn.plus2_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.plus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.zero_component = cp.sqrt(params["n0"]) * cp.ones(wfn.grid.shape, dtype='complex128')
    wfn.minus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus2_component = cp.zeros(wfn.grid.shape, dtype='complex128')


def _biaxial_initial_state(wfn: Wavefunction, params: dict):
    """Sets wavefunction components to biaxial nematic polar state."""
    wfn.plus2_component = cp.sqrt(params["n0"]) / cp.sqrt(2.) * cp.ones(wfn.grid.shape, dtype='complex128')
    wfn.plus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.zero_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus2_component = cp.sqrt(params["n0"]) / cp.sqrt(2.) * cp.ones(wfn.grid.shape, dtype='complex128')


def _ferromagnetic2p_initial_state(wfn: Wavefunction, params: dict):
    """Sets wavefunction components to ferromagnetic (F=2) state, with atoms in the plus two component."""
    wfn.plus2_component = cp.sqrt(params["n0"]) * cp.ones(wfn.grid.shape, dtype='complex128')
    wfn.plus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.zero_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus2_component = cp.zeros(wfn.grid.shape, dtype='complex128')


def _ferromagnetic2m_initial_state(wfn: Wavefunction, params: dict):
    """Sets wavefunction components to ferromagnetic (F=2) state, with atoms in the minus two component."""
    wfn.plus2_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.plus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.zero_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus2_component = cp.sqrt(params["n0"]) * cp.ones(wfn.grid.shape, dtype='complex128')


def _ferromagnetic1p_initial_state(wfn: Wavefunction, params: dict):
    """Sets wavefunction components to ferromagnetic (F=1) state, with atoms in the plus one component."""
    wfn.plus2_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.plus1_component = cp.sqrt(params["n0"]) * cp.ones(wfn.grid.shape, dtype='complex128')
    wfn.zero_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus2_component = cp.zeros(wfn.grid.shape, dtype='complex128')


def _ferromagnetic1m_initial_state(wfn: Wavefunction, params: dict):
    """Sets wavefunction components to ferromagnetic (F=1) state, with atoms in the minus one component."""
    wfn.plus2_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.plus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.zero_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus1_component = cp.sqrt(params["n0"]) * cp.ones(wfn.grid.shape, dtype='complex128')
    wfn.minus2_component = cp.zeros(wfn.grid.shape, dtype='complex128')


def _cyclic_initial_state(wfn: Wavefunction, params: dict):
    """Sets wavefunction components to (two-component) cyclic state,
    with atoms in the plus two and minus one components."""
    fz = params["p"] + params["q"] / (params["c2"] * params["n0"])

    wfn.plus2_component = cp.sqrt(params["n0"]) * cp.sqrt((1 + fz) / 3) * cp.ones(wfn.grid.shape, dtype='complex128')
    wfn.plus1_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.zero_component = cp.zeros(wfn.grid.shape, dtype='complex128')
    wfn.minus1_component = cp.sqrt(params["n0"]) * cp.sqrt((2 - fz) / 3) * cp.ones(wfn.grid.shape, dtype='complex128')
    wfn.minus2_component = cp.zeros(wfn.grid.shape, dtype='complex128')