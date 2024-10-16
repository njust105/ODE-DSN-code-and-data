import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd
import os


class RandomPath:
    def __init__(self, dimensions: int, num_control_points: int, dt: float, end_time: float) -> None:
        self.dimensions = dimensions
        self.dt = dt
        self.end_time = end_time
        self.n_step = int(self.end_time / self.dt)
        self.num_control_points = num_control_points
        self.control_points = np.zeros(
            (self.num_control_points, self.dimensions))
        self.times = np.zeros(1)
        self.path = np.zeros((self.dimensions, 1))

    def reset(self):
        self.control_points = np.zeros(
            (self.num_control_points, self.dimensions))
        self.times = np.zeros(1)
        self.path = np.zeros((self.dimensions, 1))

    def generate_paths(self):
        self.times = np.linspace(0, self.n_step * self.dt, self.n_step + 1)
        curves = []
        for i in range(self.dimensions):
            curves.append(CubicSpline(range(self.num_control_points),
                          self.control_points[:, i], bc_type='not-a-knot'))

        curves_interpolated = []
        for i in range(self.dimensions):
            curves_interpolated.append(curves[i](
                range(self.num_control_points)[-1] * np.linspace(0, 1, self.n_step + 1)))

        self.path = np.array(curves_interpolated)

    def generate_control_points(self, space_ranges: list = [-1, 1]):
        space_ranges = np.array(space_ranges)
        # repeat to other space if one range is provided
        if space_ranges.ndim == 1:
            space_ranges = np.tile(space_ranges, (self.dimensions, 1))
        elif len(space_ranges) < self.dimensions:
            print("Number of ranges is less than dimensions")
            exit(1)

        self.control_points = np.zeros(
            (self.num_control_points, self.dimensions))
        self.control_points[0] = np.zeros(self.dimensions)
        for i in range(1, self.num_control_points):
            new_point = np.zeros(self.dimensions)
            for j in range(self.dimensions):
                new_point[j] = np.random.uniform(
                    space_ranges[j][0], space_ranges[j][1])
            self.control_points[i] = new_point

    def plot_path(self, figsize: tuple | None = None):
        num_3d_plots = self.dimensions // 3
        if self.dimensions % 3 == 1:
            num_3d_plots -= 1
        num_2d_plots = (self.dimensions - 3 * num_3d_plots) // 2

        fig = plt.figure(figsize=figsize)

        # 3d
        for i in range(num_3d_plots):
            ax = fig.add_subplot(num_3d_plots + num_2d_plots,
                                 1, i + 1, projection='3d')
            ax.plot(self.path[3*i], self.path[3*i+1],
                    self.path[3*i+2], label='$x_{'+str(3*i)+'} - x_{' + str(3*i+2) + '}$')
            ax.set_xlabel('$x_{'+str(3*i)+'}$')
            ax.set_ylabel('$x_{'+str(3*i+1)+'}$')
            ax.set_zlabel('$x_{'+str(3*i+2)+'}$')
            # ax.set_title('Random Motion Path')
            ax.legend()

        # 2d
        for i in range(num_2d_plots):
            ax = fig.add_subplot(
                num_3d_plots + num_2d_plots, 1, num_3d_plots + i + 1)
            ax.plot(self.path[3*num_3d_plots+2*i],
                    self.path[3*num_3d_plots+2*i+1], label='$x_{'+str(3*num_3d_plots+2*i)+'} - x_{'+str(3*num_3d_plots+2*i+1)+'}$')
            ax.set_xlabel('$x_{'+str(3*num_3d_plots+2*i)+'}$')
            ax.set_ylabel('$x_{'+str(3*num_3d_plots+2*i+1)+'}$')
            # ax.set_title('Random Motion Path')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def save(self, save_path: str):
        df = pd.DataFrame(
            np.vstack([self.times, self.path]).T, columns=[
                'time'] + [f'x{i}' for i in range(len(self.path))]
        )

        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(save_path, index=False)
        print(f"Random path has been saved to {save_path}.")


