#!coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

class ClassMap():

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.generate_map(row, col)

    def generate_map(self, row, col):
        return np.zeros([self.row, self.col])

    def generate_obstacle(self, n=2):
        for i in range(n):
            obstc_x = np.random.randint(0, self.col)
            obstc_y = np.random.randint(0, self.row)
            obstc_length = np.random.randint(1, self.col // 2)
            obstc_width = np.random.randint(1, self.row // 2)
            # value 1 means this zone have obstacle 
            self.map[obstc_x:obstc_x + obstc_length + 1, obstc_y:obstc_y + obstc_width + 1] = 1

    def draw_map(self, path=None, edge=None):
        plt.imshow(self.map)
        # plot(col, row)

        if path:
            for point in path:
                plt.plot(point[1], point[0], 'bo')

            # plt.plot(edge[:, point[0], 'bo')
        plt.plot(self.begin[1], self.begin[0], 'ro')
        plt.plot(self.goal[1], self.goal[0], 'go')
        plt.show()

    def in_free_space(self, row, col):
        if row >= self.row or row < 0 \
           or col >= self.col or col < 0 \
           or self.map[row, col] == 1:
            return False
        else:
            return True

    def random_gen_point(self):
        while True:
            row = np.random.randint(0, self.row)
            col = np.random.randint(0, self.col)
            if self.in_free_space(row, col):
                break

        return np.array([row, col])

    def init_goal_and_begin_point(self):
        self.goal = self.random_gen_point()
        self.begin = self.random_gen_point()

    def print_gold_and_init(self):
        print(self.goal_row, self.goal_col)
        print(self.map[self.goal_row, self.goal_col])
        print(self.init_row, self.init_col)
        print(self.map[self.init_row, self.init_col])

# m = ClassMap(100,100)
# m.generate_obstacle()
# m.init_goal_and_begin_point()
# m.draw_map()