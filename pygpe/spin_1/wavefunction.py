from pygpe.shared.grid import Grid2D
import cupy as cp


class Wavefunction2D:

    def __init__(self, grid: Grid2D):
        self.grid = grid

        self.plus_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')
        self.zero_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')
        self.minus_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')
        self.fourier_plus_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')
        self.fourier_zero_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')
        self.fourier_minus_component = cp.empty((grid.num_points_x, grid.num_points_y), dtype='complex64')

    def set_initial_state(self, ground_state: str):
        if ground_state.lower() == "polar":
            self.plus_component = complex(0., 0.)
            self.zero_component = complex(1., 0.)
            self.minus_component = complex(0., 0.)
        else:
            raise ValueError(f"Argument \"{ground_state}\" is not a supported type")
