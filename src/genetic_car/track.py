import cv2
import numpy as np
from scipy.spatial import KDTree
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from genetic_car.plots import draw_track
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def find_center_path(track):
    """
    Finds the ordered center path of a track represented as a 2D binary array.

    Parameters:
    - track (np.ndarray): 2D array with 0s as the track and 1s (or others) as off-track.

    Returns:
    - list of tuple: An ordered list of (row, col) coordinates representing the center path.
    """
    # Ensure binary array
    binary_track = (track == 0).astype(np.uint8)

    # Skeletonize the track to get the centerline
    skeleton = skeletonize(binary_track)

    # Label connected components in the skeleton
    labeled_skeleton = label(skeleton)

    # Keep the largest connected component
    regions = regionprops(labeled_skeleton)
    if not regions:
        return []

    largest_region = max(regions, key=lambda r: r.area)
    center_path_binary = (labeled_skeleton == largest_region.label)

    # Extract coordinates of the skeleton
    skeleton_coords = np.column_stack(np.nonzero(center_path_binary))

    # switch row and col
    skeleton_coords = np.flip(skeleton_coords, axis=1)

    # Sort the coordinates to produce an ordered path
    ordered_path = order_skeleton_points(skeleton_coords)

    return np.asarray(ordered_path)


def order_skeleton_points(points, start_index=0) -> np.ndarray:
    """
    Orders skeleton points into a sequential path.

    Parameters:
    - points (np.ndarray): Array of shape (N, 2) with (row, col) coordinates.

    Returns:
    - list of tuple: Ordered list of (row, col) coordinates.
    """
    if len(points) == 0:
        return []

    # Use a KDTree to find the nearest neighbors
    tree = KDTree(points)
    ordered = []

    # Start with an arbitrary point (e.g., the first point)
    current = points[start_index]
    visited = set()

    while len(visited) < len(points):
        # Add current point to the ordered list
        ordered.append(tuple(current))
        visited.add(tuple(current))

        # Find the nearest neighbor that hasn't been visited
        distances, indices = tree.query(current, k=len(points))
        for idx in indices:
            neighbor = tuple(points[idx])
            if neighbor not in visited:
                current = points[idx]
                break

    return np.asarray(ordered)


class Track:
    def __init__(self, image: np.ndarray) -> None:
        log.info("loading track...")
        self.image = image
        self.image = cv2.dilate(self.image, np.ones((15, 15), np.uint8), iterations=1)
        self.center_path = find_center_path(self.image)
        log.info("track loaded")

    def get_progress(self, point: tuple) -> float:
        closest_point_index = self.get_closest_point_index(point)
        progress = closest_point_index / (len(self.center_path) - 1)
        return progress

    def set_start_point(self, point: tuple) -> None:
        log.info("setting start point to %s", point)
        closest_point_index = self.get_closest_point_index(point) - 1
        if closest_point_index < 0:
            closest_point_index = len(self.center_path) - 1
        self.center_path = order_skeleton_points(self.center_path, start_index=closest_point_index)

    def get_closest_point_index(self, point: tuple) -> int:
        tree = KDTree(self.center_path)
        _, idx = tree.query(point)
        return idx

    def direction_at_point(self, point: tuple) -> float:
        idx = self.get_closest_point_index(point)
        if idx == 0:
            idx += 1
        point_range = 5
        nearby_point_indexes = np.arange(idx - point_range, idx + point_range)
        nearby_point_indexes[nearby_point_indexes < 0] += len(self.center_path)

        # Fit a line to the nearby points
        nearby_points = self.center_path[nearby_point_indexes]
        x = nearby_points[:, 0]
        y = nearby_points[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return np.degrees(np.arctan(m))

    @property
    def start_point(self):
        return tuple(self.center_path[0])


if __name__ == "__main__":
    from pathlib import Path

    image = cv2.imread(str(Path(__file__).parent / "tracks" / "race_track_1.png"), cv2.IMREAD_GRAYSCALE)
    track = Track(image)
    center_path = track.center_path

    # Create a window to display the track
    track_image = draw_track(track)


    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # Draw the track and mouse position
            display_image = track_image.copy()
            cv2.circle(display_image, (x, y), 5, (255, 0, 0), -1)
            closest_point_idx = track.get_closest_point_index((x, y))
            closest_point = center_path[closest_point_idx]
            cv2.circle(display_image, (closest_point[0], closest_point[1]), 5, (0, 255, 0), -1)

            cv2.putText(display_image, f"Progress: {track.get_progress((x,y)):.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Show the updated image
            cv2.imshow("Track", display_image)


    cv2.namedWindow("Track")
    cv2.setMouseCallback("Track", mouse_callback)

    # Display the image and wait for user interaction
    cv2.imshow("Track", track_image)
    while True:
        key_press = cv2.waitKey(10)
        if key_press == ord("q"):
            break

    cv2.destroyAllWindows()
