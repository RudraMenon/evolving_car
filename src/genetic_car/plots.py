from typing import List, Tuple
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.genetic_car.helpers import Point

if TYPE_CHECKING:
    from src.genetic_car.car import Car
    from src.genetic_car.track import Track


# Track plotting functions
def draw_track(track: "Track") -> np.ndarray:
    img_cpy = track.image.copy()
    img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_GRAY2RGB)
    for point in track.center_path:
        img_cpy = cv2.circle(img_cpy, tuple(point), 1, (0, 0, 255), -1)
    # draw start_point
    img_cpy = cv2.circle(img_cpy, track.start_point, 5, (0, 255, 0), -1)
    return img_cpy


def show_track(track):
    plt.figure()
    plt.imshow(draw_track(track))


# Car plotting functions
def draw_car(car: "Car") -> np.ndarray:
    image = draw_track(car.track)
    for crash_point in car.crash_points:
        image = cv2.circle(image, crash_point.to_int_tuple(), 5, (255, 0, 0), -1)
        image = cv2.line(
            image,
            car.position.to_int_tuple(),
            crash_point.to_int_tuple(),
            (255, 0, 0),
            1,
        )
    # draw arrow indicating the direction of the car
    arrow_length = 20
    arrow_end = (
        car.position.x + arrow_length * np.cos(np.radians(car.direction)),
        car.position.y + arrow_length * np.sin(np.radians(car.direction)),
    )
    image = cv2.arrowedLine(
        image,
        car.position.to_int_tuple(),
        (int(arrow_end[0]), int(arrow_end[1])),
        (0, 0, 255),
        2,
    )

    car_pos = car.position.to_int_tuple()
    closest_point_index = car.track.get_closest_point_index(car_pos)
    closest_point = car.track.center_path[closest_point_index]
    # image = cv2.circle(image, closest_point, 5, (255, 255, 0), -1)
    return image


def replay_run(car, run: List[dict]):
    """ interactively replay a run of the car on the track with opencv"""
    car = car.copy()
    car.reset()
    cv2.imshow("Replay", draw_car(car))
    delay = 0
    i = 0
    while i < len(run):
        position = run[i]['position']
        direction = run[i]['direction']
        progress = run[i]['progress']

        if "action" in run[i]:
            action = run[i]['action']
        else:
            action = None

        if "reward" in run[i]:
            reward = run[i]['reward']
        else:
            reward = None

        if "direction_offset" in run[i]:
            direction_offset = f"{run[i]['direction_offset']:0.2f}"
        else:
            direction_offset = "None"

        car.position = Point(*position)
        car.direction = direction
        car.update_crash_points()

        img = draw_car(car)
        # draw reward on the image
        texts = []
        texts += [f"Step: {i}"]
        texts += [f"Position: {car.position.to_int_tuple()}"]
        texts += [f"Direction: {direction:.2f}"]
        texts += [f"Direction Offset: {direction_offset }"]
        texts += [f"Action: {action}"]
        texts += [f"Reward: {reward:.2f}"]
        texts += [f"Progress: {progress:.2f}"]
        texts += [f"Is Crashed: {car.is_crashed()}"]

        font_scale = 0.5
        for j, text in enumerate(texts):
            cv2.putText(img, text, (10, 20 + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
        cv2.imshow("Replay", img)
        key_press = cv2.waitKey(delay)
        if key_press == ord("q"):
            break
        if key_press == ord("f"):
            delay = 1 if delay == 0 else 0
        if key_press == "a":
            i -= 1
        if key_press == "d":
            i += 1
        i += 1

    cv2.destroyAllWindows()
    return img
