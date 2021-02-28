from datetime import datetime
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import os
import random as rnd
from hyperparameters import *

class Cells:
    def __init__(self, energy_sources) -> None:
        self.energy_sources = energy_sources
        self.cells = cp.zeros((WIDTH, HEIGHT))
        self.start_time = datetime.now()
        self.counter = 0

        self._create_img_dir()

    def save_as_img(self) -> None:
        plt.imshow(cp.asnumpy(self.cells), interpolation=IMG_INTERPOLATION)
        plt.title(TITLE_FORMAT.format(self.counter))

        plt.savefig("{}/{}.png".format(self._get_img_dir(), self.counter))

    def _get_img_dir(self) -> str:
        start_time = self.start_time.strftime("%Y-%m-%d-%H-%M-%S")
        dir_name = SAVE_DIRECTORY_FORMAT.format(start_time)
        return dir_name

    def _create_img_dir(self) -> None:
        if not os.path.isdir(self._get_img_dir()):
            os.mkdir(self._get_img_dir())

    def simulate(self):
        print("Simulate energy...")
        self._simulate_energy()
        print("Simulate death...")
        self._simulate_death()
        print("Simulate birth...")
        self._simulate_birth()
        self.counter += 1

    def _simulate_energy(self) -> None:
        next_cells_alive = cp.zeros((WIDTH, HEIGHT))

        for source in self.energy_sources:
            print("Simulate energy from {}...".format(source))
            partial_cells_alive = self._simulate_each_source(source)
            next_cells_alive += partial_cells_alive
            next_cells_alive[next_cells_alive > ENERGY_SOURCE_POWER] = ENERGY_SOURCE_POWER

        self.cells = next_cells_alive

    def _get_max_distance(self) -> int:
        ret = 1
        mul = ENERGY_SOURCE_POWER
        while mul >= ENERGY_MIN:
            mul *= ENERGY_LOSS_RATE_MEAN
            ret += 1
        return ret - 1

    def _breadth_first_search(self, energy_pos):
        (energy_pos_x, energy_pos_y) = energy_pos
        energy_pos_x = int(energy_pos_x)
        energy_pos_y = int(energy_pos_y)

        binary_cells = cp.copy(self.cells)
        binary_cells[binary_cells > 0] = 1
        binary_cells[energy_pos_x][energy_pos_y] = 1

        bfs_cells = np.full((WIDTH, HEIGHT), INFINITY)
        bfs_cells[energy_pos_x][energy_pos_y] = 0

        current_stack = []
        current_stack.append((energy_pos_x, energy_pos_y))

        dx = [0, 0, -1, -1, -1, 1, 1, 1]
        dy = [-1, 1, 0, -1, 1, 0, -1, 1]

        max_distance = self._get_max_distance()

        while len(current_stack) != 0:
            (x, y) = current_stack.pop()
            x = int(x)
            y = int(y)

            if bfs_cells[x][y] == max_distance:
                continue

            for i in range(8):
                next_x = int(x + dx[i])
                next_y = int(y + dy[i])

                if (next_x < 0) or (WIDTH <= next_x):
                    continue

                if (next_y < 0) or (HEIGHT <= next_y):
                    continue

                if binary_cells[next_x][next_y] == 0:
                    continue

                if bfs_cells[next_x][next_y] > bfs_cells[x][y] + 1:
                    bfs_cells[next_x][next_y] = bfs_cells[x][y] + 1
                    current_stack.append((next_x, next_y))

        related_cells = cp.copy(cp.asarray(bfs_cells))
        related_cells[related_cells != INFINITY] = 1
        related_cells[related_cells == INFINITY] = 0

        bfs_cells[bfs_cells == INFINITY] = 0
        bfs_cells = cp.asarray(bfs_cells)

        return related_cells, bfs_cells

    def _simulate_each_source(self, energy_pos):
        related_cells, bfs_cells = self._breadth_first_search(energy_pos)
        filled_power = cp.full((WIDTH, HEIGHT), ENERGY_SOURCE_POWER)
        filled_decrease = cp.full((WIDTH, HEIGHT), ENERGY_LOSS_RATE_MEAN)
        normalized_random = cp.random.normal(1, ENERGY_LOSS_DEVIATION, (WIDTH, HEIGHT))
        normalized_random[normalized_random < ENERGY_RANDOM_MIN] = ENERGY_RANDOM_MIN

        cells_from_source = cp.multiply(cp.multiply(filled_power, related_cells), cp.power(filled_decrease, bfs_cells))
        randomized_cells_from_source = cp.multiply(cells_from_source, normalized_random)

        return randomized_cells_from_source

    def _simulate_death(self) -> None:
        randoms = cp.random.random((WIDTH, HEIGHT))
        self.cells -= randoms
        self.cells[self.cells < 0] = -1
        self.cells += randoms
        self.cells[self.cells < 0] = 0

    def _simulate_birth(self):
        dx = [0, 0, -1, -1, -1, 1, 1, 1]
        dy = [-1, 1, 0, -1, 1, 0, -1, 1]

        next_cells_alive = cp.zeros((WIDTH, HEIGHT))

        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.cells[x][y] != 0:
                    next_cells_alive[x][y] = self.cells[x][y]
                    continue

                for i in range(8):
                    next_x = x + dx[i]
                    next_y = y + dy[i]

                    if (next_x < 0) or (WIDTH <= next_x):
                        continue

                    if (next_y < 0) or (HEIGHT <= next_y):
                        continue

                    if rnd.random() <= self.cells[next_x][next_y] * BIRTH_RATE:
                        next_cells_alive[x][y] = self.cells[next_x][next_y] * ENERGY_LOSS_RATE_MEAN
                        break

        self.cells = next_cells_alive

def main():
    sources = cp.array([(7, 5), (20, 8), (14, 20), (40, 20), (40, 35), (25, 35)])

    cells = Cells(sources)

    for count in range(EPOCH):
        print("-----Count: {}-----".format(count))
        cells.simulate()
        cells.save_as_img()

if __name__ == "__main__":
    main()

