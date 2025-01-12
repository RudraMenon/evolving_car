import copy
import time
from typing import Optional, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.genetic_car.helpers import Point
from src.genetic_car.plots import draw_car
from src.genetic_car.track import Track


def bresenhams_line_algorithm(start: tuple, direction: float, image, max_length):
    """
    Implements Bresenham's line algorithm to find the closest point that is a '1' in a given direction in a 2D image.

    Parameters:
    - start: Tuple (x, y) representing the starting coordinates.
    - direction: Tuple (dx, dy) representing the direction vector.
    - image: 2D numpy array representing the image.
    - max_length: Maximum length to trace the line.

    Returns:
    - (x, y): Coordinates of the closest point that is '1' or the point at max_length if no intersection is found.
    """
    x0, y0 = start
    dx, dy = np.cos(np.radians(direction)), np.sin(np.radians(direction))

    # Initialize variables
    x, y = x0, y0
    rows, cols = image.shape

    # Bresenham's line drawing loop with max length constraint
    for _ in range(max_length):
        # Move in the direction (dx, dy)
        x += dx
        y += dy

        # Stop if out of bounds
        if not (0 <= x < cols and 0 <= y < rows):
            break

        if image[int(y), int(x)] != 0:
            return (x, y)

    # Return the point at max length if no intersection found
    return (x0 + dx * max_length, y0 + dy * max_length)


class Car:
    def __init__(self,
                 track: Track,
                 starting_point: Point,
                 position: Optional[Point] = None,
                 direction: float = 0,
                 speed: float = 10,
                 max_turn_angle: float = 20,
                 view_angles: Optional[List[int]] = None,
                 view_length: int = 100) -> None:
        self.track = track
        self.starting_point = starting_point
        self.position = position or starting_point.copy()
        self.initial_speed = speed
        self.speed = speed
        self.direction = direction
        self.max_turn_angle = max_turn_angle
        self.initial_direction = direction
        self.view_angles = view_angles or [-10, 0, 10]
        self.view_length = view_length
        self.crash_points = self.calc_crash_points()

    def turn(self, angle: float):
        angle = np.clip(angle, -self.max_turn_angle, self.max_turn_angle)
        self.direction += angle

    def update_crash_points(self):
        self.crash_points = self.calc_crash_points()

    def calc_crash_points(self):
        crash_points = []
        for angle in self.view_angles:
            intersection = self.find_intersection(direction_offset=angle)
            crash_points.append(intersection)

        return crash_points

    def find_intersection(self, direction_offset: float = 0):
        direction = self.direction + direction_offset
        intersection = bresenhams_line_algorithm(self.position.to_tuple(),
                                                 direction,
                                                 self.track.image,
                                                 self.view_length)
        return Point(*intersection)

    def move(self, time_step: float = 1):
        dx = float(self.speed * np.cos(np.radians(self.direction)))
        dy = float(self.speed * np.sin(np.radians(self.direction)))
        self.x += dx * time_step
        self.y += dy * time_step
        self.crash_points = self.calc_crash_points()

    def is_crashed(self):
        img_h, img_w = self.track.image.shape
        if not (0 <= self.x < img_w and 0 <= self.y < img_h):
            return True
        if self.track.image[int(self.y), int(self.x)] != 0:
            return True
        return False

    def progress_on_track(self):
        return self.track.get_progress(self.position.to_tuple())

    def to_dict(self):
        return {
            "position": self.position.to_tuple(),
            "direction": self.direction,
            "speed": self.speed,
            "max_turn_angle": self.max_turn_angle,
            "view_angles": self.view_angles,
            "view_length": self.view_length
        }

    @property
    def x(self):
        return self.position.x

    @x.setter
    def x(self, value):
        self.position.x = value

    @property
    def y(self):
        return self.position.y

    @y.setter
    def y(self, value):
        self.position.y = value

    def reset(self):
        self.position = self.starting_point.copy()
        self.direction = self.initial_direction
        self.speed = self.initial_speed
        self.crash_points = self.calc_crash_points()

    def copy(self):
        return copy.deepcopy(self)


if __name__ == "__main__":
    from pathlib import Path

    image = cv2.imread(str(Path(__file__).parent / "tracks" / "race_track_1.png"), cv2.IMREAD_GRAYSCALE)
    track = Track(image)
    st = time.time()
    car = Car(track, Point(222, 96), view_angles=[30, 15, 0, -15, -30], direction=0)
    print("Time taken:", time.time() - st)
    image = draw_car(car)
    plt.imshow(image)
    plt.show()
