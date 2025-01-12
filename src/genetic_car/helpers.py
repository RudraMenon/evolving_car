import numpy as np


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __iter__(self):
        yield self.x
        yield self.y

    def to_tuple(self):
        return self.x, self.y

    def to_int_tuple(self):
        return int(self.x), int(self.y)

    def copy(self):
        return Point(self.x, self.y)


def calc_distance(pt1, pt2):
    return np.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)
