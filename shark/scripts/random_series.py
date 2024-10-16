from random_path import *
from fastdtw import fastdtw
from time import time


class RandomSeries:
    def __init__(self, dimensions: int, batch_size: int, num_control_points: int, dt: float, end_time: float, ranges: list, distance_method="cp") -> None:
        self.series = []
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.num_control_points = num_control_points
        self.dt = dt
        self.end_time = end_time
        self.ranges = ranges
        self.distance_method = distance_method
        self.saving = False
        self.save_dir = "."

    def is_saving(self):
        self.saving = True

    def set_save_dir(self, dir):
        self.save_dir = dir

    def init(self):
        a = RandomPath(self.dimensions, self.num_control_points,
                       self.dt, self.end_time)
        a.generate_control_points([0, 0.0001])
        a.generate_paths()
        self.series = [a]
        if self.saving:
            a.save(f"{self.save_dir}/0.csv")

    def generate_continuous_multivariate_series(self):
        series = []
        for _ in range(self.batch_size):
            seq = RandomPath(
                self.dimensions, self.num_control_points, self.dt, self.end_time)
            seq.generate_control_points(self.ranges)
            seq.generate_paths()
            series.append(seq)
        return series

    def calculate_distance(self, series1: RandomPath, series2: RandomPath,):
        if self.distance_method == "cp":
            distance, _ = fastdtw(series1.control_points.transpose(),
                                  series2.control_points.transpose())
        elif self.distance_method == "full":
            distance, _ = fastdtw(series1.path.transpose(),
                                  series2.path.transpose())
        else:
            return 0

        return distance

    def add_new_serie(self):
        candidate_series = self.generate_continuous_multivariate_series()
        farthest_candidate = None
        farthest_dist = -np.inf
        for candidate in candidate_series:
            min_dist_to_selected = min(self.calculate_distance(
                candidate, present_serie) for present_serie in self.series)
            if min_dist_to_selected > farthest_dist:
                farthest_dist = min_dist_to_selected
                farthest_candidate = candidate
        self.series.append(farthest_candidate)

    def generate_series(self, n: int):
        time_start = time()
        self.init()
        while len(self.series) < n:
            # self.plot_series(0)
            time_1 = time()
            self.add_new_serie()
            time_2 = time()
            time_used = time_2 - time_start
            time_left = self.predict_time(n, time_used)
            print(f"{len(self.series)}/{n}",
                  f"({time_2 - time_1:.2f} s)",
                  f"\ttotal used: {self.format_duration(time_used)}",
                  f"\tleft: {self.format_duration(time_left)}")
            if self.saving:
                self.series[-1].save(f"{self.save_dir}/{len(self.series)}.csv")

    def predict_time(self, n,  time_used):
        n_ = len(self.series)
        time_per_ = time_used/((1 + n_) * n_ / 2)
        time_left = (n_+1 + n) * (n - n_) / 2 * time_per_
        return time_left

    def format_duration(self, seconds):
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        time_components = []
        if days > 0:
            time_components.append(f"{days:.0f}d")
        if hours > 0:
            time_components.append(f"{hours:.0f}h")
        if minutes > 0:
            time_components.append(f"{minutes:.0f}m")
        time_components.append(f"{seconds:.1f}s")
        return " ".join(time_components)

    def save_paths(self, dir):
        for i, seq in enumerate(self.series):
            seq.save(f"{dir}/{i}.csv")

    def plot_series(self, ):
        if self.dimensions == 1:
            plt.figure()
            for idx, seq in enumerate(self.series):
                plt.plot(seq.times, seq.path[0, :]
                         #  , label=f'serie {idx}'
                         )
            plt.xlabel('t')
            plt.ylabel(f'y')
            plt.legend()
            plt.show()
            return

        num_3d_plots = self.dimensions // 3
        if self.dimensions % 3 == 1:
            num_3d_plots -= 1
        num_2d_plots = (self.dimensions - 3 * num_3d_plots) // 2

        fig = plt.figure()

        for i in range(num_3d_plots):
            ax = fig.add_subplot(num_3d_plots + num_2d_plots,
                                 1, i + 1, projection='3d')
            for idx, seq in enumerate(self.series):
                ax.plot(seq.path[3*i], seq.path[3*i+1],
                        seq.path[3*i+2]
                        # , label='$x_{'+str(3*i)+'} - x_{' + str(3*i+2) + '}$'
                        )
            ax.set_xlabel('$x_{'+str(3*i)+'}$')
            ax.set_ylabel('$x_{'+str(3*i+1)+'}$')
            ax.set_zlabel('$x_{'+str(3*i+2)+'}$')
            # ax.set_title('Random Motion Path')
            ax.legend()

        for i in range(num_2d_plots):
            ax = fig.add_subplot(
                num_3d_plots + num_2d_plots, 1, num_3d_plots + i + 1)
            for idx, seq in enumerate(self.series):
                ax.plot(seq.path[3*num_3d_plots+2*i],
                        seq.path[3*num_3d_plots+2*i+1]
                        # , label='$x_{'+str(3*num_3d_plots+2*i)+'} - x_{'+str(3*num_3d_plots+2*i+1)+'}$'
                        )
            ax.set_xlabel('$x_{'+str(3*num_3d_plots+2*i)+'}$')
            ax.set_ylabel('$x_{'+str(3*num_3d_plots+2*i+1)+'}$')
            # ax.set_title('Random Motion Path')
            ax.legend()

        plt.tight_layout()
        plt.show()


save_dir = "paths"
saving = False

all_series = []

dim = 6
batch_size = 100

# RO-FRP
strain_ranges = [-0.4, 0.4]
time_range = [1.3e-4, 400]
time_step_num = 1000

# # UD_FRP
# strain_ranges = [-0.02, 0.02]
# time_range = [6.66e-6, 20]
# time_step_num = 1000

series_num_per_time_range_3 = 30
series_num_per_time_range_2 = 10
time_range_step_num = 20

time_ends = np.logspace(np.log10(time_range[0]), np.log10(
    time_range[1]), time_range_step_num)


print(time_ends)

number = 0
for i, time_end in enumerate(time_ends):
    print("time: ", time_end, f"{i+1}/{time_range_step_num}")
    cp3 = RandomSeries(dim, batch_size, 3, time_end/time_step_num, time_end,
                       strain_ranges, distance_method="cp")
    cp3.generate_series(series_num_per_time_range_3)

    cp2 = RandomSeries(dim, batch_size, 2, time_end/time_step_num, time_end,
                       strain_ranges, distance_method="cp")
    cp2.generate_series(series_num_per_time_range_2)

    for c in cp2.series:
        if saving:
            c.save(f"{save_dir}/{number}.csv")
            number += 1
        else:
            all_series.append(c)

    for c in cp3.series:
        if saving:
            c.save(f"{save_dir}/{number}.csv")
            number += 1
        else:
            all_series.append(c)


if not saving:
    import random
    random.shuffle(all_series)
    for i, c in enumerate(all_series):
        c.save(f"{save_dir}/{i}.csv")
