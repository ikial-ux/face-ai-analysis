import math

def compute_angle_between_coords(coord_a, coord_b):
    """
    Calculate the angle between two sets of coordinates in degrees.

    Args:
        coord_a (tuple): Tuple representing the first set of coordinates (x, y).
        coord_b (tuple): Tuple representing the second set of coordinates (x, y).

    Returns:
        float: Angle between the coordinates in degrees.
    """
    return 180 / math.pi * math.atan((float(coord_b[1] - coord_a[1]) / (coord_b[0] - coord_a[0])))


def find_enclosing_box(coords):
    """
    Find the enclosing box coordinates (x, y, width, height) from a list of coordinates.

    Args:
        coords (numpy.ndarray): Array of coordinates.

    Returns:
        tuple: Enclosing box coordinates (x, y, width, height).
    """
    return min(coords[:, 0]), min(coords[:, 1]), max(coords[:, 0]) - min(coords[:, 0]), max(coords[:, 1]) - min(coords[:, 1])


def determine_feature_boundary(coordinates, feature_index):
    """
    Determine the boundary coordinates for a specific facial feature.

    Args:
        coordinates (numpy.ndarray): Array of facial landmark coordinates.
        feature_index (int): Index representing a particular facial feature.

    Returns:
        tuple: Boundary coordinates (x, y, width, height) for the specified facial feature.
    """
    feature_ranges = {
        1: (17, 22),
        2: (22, 27),
        3: (36, 42),
        4: (42, 48),
        5: (29, 36),
        6: (48, 68),
        7: (48, 68)
    }

    start_idx, end_idx = feature_ranges.get(feature_index, (0, 0))
    if start_idx and end_idx:
        x_val, y_val, width_val, height_val = find_enclosing_box(coordinates[start_idx:end_idx])
        return x_val, y_val, width_val, height_val
    else:
        return None