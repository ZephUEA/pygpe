import h5py
from pygpe.shared.grid import Grid
from pygpe.spin_1.wavefunction import Wavefunction
from pygpe.spin_1.parameters import Parameters


class DataManager:
    def __init__(self, filename: str, data_path: str):
        self.filename = filename
        self.data_path = data_path

        # Create file
        h5py.File(f'{self.data_path}/{self.filename}', 'w')

    def save_initial_parameters(self, grid: Grid, wfn: Wavefunction, parameters: Parameters) -> None:
        """Saves the initial grid, wavefunction and parameters to a HDF5 file.

        :param grid: The grid object of the system.
        :param wfn: The wavefunction of the system.
        :param parameters: The parameter object of the system.
        """

        self._save_initial_grid_params(grid)
        self._save_initial_wfn(wfn)
        self._save_params(parameters)

    def _save_initial_grid_params(self, grid: Grid) -> None:
        """Creates new datasets in file for grid-related parameters and saves
        initial values.
        """
        if grid.ndim == 1:
            with h5py.File(f'{self.data_path}/{self.filename}', 'w') as file:
                file.create_dataset('grid/nx', data=grid.num_points_x)
                file.create_dataset('grid/dx', data=grid.grid_spacing_x)
        elif grid.ndim == 2:
            with h5py.File(f'{self.data_path}/{self.filename}', 'w') as file:
                file.create_dataset('grid/nx', data=grid.num_points_x)
                file.create_dataset('grid/ny', data=grid.num_points_y)
                file.create_dataset('grid/dx', data=grid.grid_spacing_x)
                file.create_dataset('grid/dy', data=grid.grid_spacing_y)
        elif grid.ndim == 3:
            raise NotImplementedError

    def _save_initial_wfn(self, wfn: Wavefunction) -> None:
        """Creates new datasets in file for the wavefunction and saves
        initial values.
        """
        if wfn.grid.ndim == 1:
            with h5py.File(f'{self.data_path}/{self.filename}', 'w') as file:
                file.create_dataset('wavefunction/psi_plus', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, None), dtype='complex128')
                file.create_dataset('wavefunction/psi_0', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, None), dtype='complex128')
                file.create_dataset('wavefunction/psi_minus', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, None), dtype='complex128')
        elif wfn.grid.ndim == 2:
            with h5py.File(f'{self.data_path}/{self.filename}', 'w') as file:
                file.create_dataset('wavefunction/psi_plus', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, wfn.grid.num_points_y), dtype='complex128')
                file.create_dataset('wavefunction/psi_0', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, wfn.grid.num_points_y), dtype='complex128')
                file.create_dataset('wavefunction/psi_minus', (wfn.grid.num_points_x, 1),
                                    maxshape=(wfn.grid.num_points_x, wfn.grid.num_points_y), dtype='complex128')
        elif wfn.grid.ndim == 3:
            raise NotImplementedError

    def _save_params(self, params: Parameters) -> None:
        """Creates new datasets in file for condensate & time parameters
        and saves initial values.
        """
        with h5py.File(f'{self.data_path}/{self.filename}', 'r+') as file:
            # Condensate and trap parameters
            file.create_dataset('parameters/c0', data=params.c0)
            file.create_dataset('parameters/c2', data=params.c2)
            file.create_dataset('parameters/p', data=params.p)
            file.create_dataset('parameters/q', data=params.q)
            file.create_dataset('parameters/trap', data=params.trap)

            # Time-related parameters
            file.create_dataset('parameters/dt', data=params.dt)
            file.create_dataset('parameters/nt', data=params.nt)
