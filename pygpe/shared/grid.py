import numpy as np
import cupy as cp


class Grid:
    """An object representing the numerical grid.
    It contains information on the number of grid points, the shape, the dimensionality, and lengths of the grid.

    :param points: Number of points in each spatial dimension.
    :type points: int or tuple of ints
    :param grid_spacings: Numerical spacing between grid points in each spatial dimension.
    :type grid_spacings: float or tuple of floats

    :ivar shape: Shape of the grid.
    :ivar ndim: Dimensionality of the grid.
    :ivar total_num_points: Total number of grid points across all dimensions.

    :ivar num_points_x: Number of points in the x-direction.
    :ivar num_points_y: (2D and 3D only) Number of points in the y-direction.
    :ivar num_points_z: (3D only) Number of points in the z-direction.
    :ivar length_x: Length of the grid in the x-direction.
    :ivar length_y: (2D and 3D only) Length of the grid in the y-direction.
    :ivar length_z: (3D only) Length of the grid in the z-direction.
    :ivar x_mesh: The x meshgrid. The dimensionality matches that of `ndim`.
    :ivar y_mesh: (2D and 3D only) The y meshgrid. The dimensionality matches that of `ndim`.
    :ivar z_mesh: (3D only) The z meshgrid. The dimensionality matches that of `ndim`.
    :ivar grid_spacing_x: Grid spacing in the x-direction.
    :ivar grid_spacing_y: (2D and 3D only) Grid spacing in the y-direction.
    :ivar grid_spacing_z: (3D only) Grid spacing in the z-direction.
    :ivar grid_spacing_product: The product of the grid spacing for each dimension.
    :ivar fourier_x_mesh: The Fourier-space x meshgrid. The dimensionality matches that of `ndim`.
    :ivar fourier_y_mesh: (2D and 3D only) The Fourier-space y meshgrid. The dimensionality matches that of `ndim`.
    :ivar fourier_z_mesh: (3D only) The Fourier-space z meshgrid. The dimensionality matches that of `ndim`.
    :ivar fourier_spacing_x: Fourier grid spacing in the x-direction.
    :ivar fourier_spacing_y: (2D and 3D only) Fourier grid spacing in the y-direction.
    :ivar fourier_spacing_z: (3D only) Fourier grid spacing in the z-direction.
    """

    def __init__(
        self, points: int | tuple[int, ...], grid_spacings: float | tuple[float, ...]
    ):
        """Constructs the grid object."""

        self.shape = points
        if isinstance(points, tuple):
            if len(points) != len(grid_spacings):
                raise ValueError(f"{points} and {grid_spacings} are not of same length")
            if len(points) > 3:
                raise ValueError(f"{points} is not a valid dimensionality")
            self.ndim = len(points)
            self.total_num_points = 1
            for point in points:
                self.total_num_points *= point
        else:
            self.ndim = 1
            self.total_num_points = points

        if self.ndim == 1:
            self._generate_1d_grids(points, grid_spacings)
        elif self.ndim == 2:
            self._generate_2d_grids(points, grid_spacings)
        elif self.ndim == 3:
            self._generate_3d_grids(points, grid_spacings)

    def _generate_1d_grids(self, points: int, grid_spacing: float):
        """Generates meshgrid for a 1D grid."""
        self.num_points_x = points
        self.grid_spacing_x = grid_spacing
        self.grid_spacing_product = self.grid_spacing_x

        self.length_x = self.num_points_x * self.grid_spacing_x
        self.x_mesh = (
            np.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.grid_spacing_x
        )

        self.fourier_spacing_x = np.pi / (self.num_points_x // 2 * self.grid_spacing_x)
        self.fourier_x_mesh = np.fft.fftshift(
            np.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.fourier_spacing_x
        )

        # Defined on device for use in evolution
        self.wave_number = cp.asarray(self.fourier_x_mesh**2)

    def _generate_2d_grids(
        self, points: tuple[int, ...], grid_spacings: tuple[float, ...]
    ):
        """Generates meshgrid for a 2D grid."""
        self.num_points_x, self.num_points_y = points
        self.grid_spacing_x, self.grid_spacing_y = grid_spacings
        self.grid_spacing_product = self.grid_spacing_x * self.grid_spacing_y

        self.length_x = self.num_points_x * self.grid_spacing_x
        self.length_y = self.num_points_y * self.grid_spacing_y

        x = (
            np.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.grid_spacing_x
        )
        y = (
            np.arange(-self.num_points_y // 2, self.num_points_y // 2)
            * self.grid_spacing_y
        )
        self.x_mesh, self.y_mesh = np.meshgrid(x, y)

        # Generate Fourier space variables
        self.fourier_spacing_x = np.pi / (self.num_points_x // 2 * self.grid_spacing_x)
        self.fourier_spacing_y = np.pi / (self.num_points_y // 2 * self.grid_spacing_y)

        fourier_x = (
            np.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.fourier_spacing_x
        )
        fourier_y = (
            np.arange(-self.num_points_y // 2, self.num_points_y // 2)
            * self.fourier_spacing_y
        )

        self.fourier_x_mesh, self.fourier_y_mesh = np.meshgrid(fourier_x, fourier_y)
        self.fourier_x_mesh = np.fft.fftshift(self.fourier_x_mesh)
        self.fourier_y_mesh = np.fft.fftshift(self.fourier_y_mesh)

        # Defined on device for use in evolution
        self.wave_number = cp.asarray(
            self.fourier_x_mesh**2 + self.fourier_y_mesh**2
        )

    def _generate_3d_grids(
        self, points: tuple[int, ...], grid_spacings: tuple[float, ...]
    ):
        """Generates meshgrid for a 3D grid."""
        self.num_points_x, self.num_points_y, self.num_points_z = points
        self.grid_spacing_x, self.grid_spacing_y, self.grid_spacing_z = grid_spacings
        self.grid_spacing_product = (
            self.grid_spacing_x * self.grid_spacing_y * self.grid_spacing_z
        )

        self.length_x = self.num_points_x * self.grid_spacing_x
        self.length_y = self.num_points_y * self.grid_spacing_y
        self.length_z = self.num_points_z * self.grid_spacing_z

        x = (
            np.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.grid_spacing_x
        )
        y = (
            np.arange(-self.num_points_y // 2, self.num_points_y // 2)
            * self.grid_spacing_y
        )
        z = (
            np.arange(-self.num_points_z // 2, self.num_points_z // 2)
            * self.grid_spacing_z
        )
        self.x_mesh, self.y_mesh, self.z_mesh = np.meshgrid(x, y, z)

        # Generate Fourier space variables
        self.fourier_spacing_x = np.pi / (self.num_points_x // 2 * self.grid_spacing_x)
        self.fourier_spacing_y = np.pi / (self.num_points_y // 2 * self.grid_spacing_y)
        self.fourier_spacing_z = np.pi / (self.num_points_z // 2 * self.grid_spacing_z)

        fourier_x = (
            np.arange(-self.num_points_x // 2, self.num_points_x // 2)
            * self.fourier_spacing_x
        )
        fourier_y = (
            np.arange(-self.num_points_y // 2, self.num_points_y // 2)
            * self.fourier_spacing_y
        )
        fourier_z = (
            np.arange(-self.num_points_z // 2, self.num_points_z // 2)
            * self.fourier_spacing_z
        )

        self.fourier_x_mesh, self.fourier_y_mesh, self.fourier_z_mesh = np.meshgrid(
            fourier_x, fourier_y, fourier_z
        )
        self.fourier_x_mesh = np.fft.fftshift(self.fourier_x_mesh)
        self.fourier_y_mesh = np.fft.fftshift(self.fourier_y_mesh)
        self.fourier_z_mesh = np.fft.fftshift(self.fourier_z_mesh)

        # Defined on device for use in evolution
        self.wave_number = cp.asarray(
            self.fourier_x_mesh**2
            + self.fourier_y_mesh**2
            + self.fourier_z_mesh**2
        )
