from typing import Union
import math
import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        Union[float, np.ndarray]: The distance between the arrays.
    """
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


def angle_distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        Union[float, np.ndarray]: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    dist = 1 - np.inner(a, b) ** 2
    return dist


def normalizeVector(data: np.ndarray, min: float=0, max: float=1) -> Union[float, np.ndarray]:
    _range = np.max(data) - np.min(data)
    if _range != 0.0:
        scaled_data = (data - np.min(data)) / (max - min)
        normalized_data = scaled_data / _range
    else: normalized_data = data

    return normalized_data

def euler_to_quaternion(roll: float, pitch: float, yaw:float) -> Union[float, np.ndarray]:
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
    return [qw, qx, qy, qz]


def area_proportion(a: float, b: float) -> Union[float, np.ndarray]:
    # calculate the proportion of the area bettween obj and hole (triangle)
    # a: obj length, b: gap

    return 1 - ((3 * a ** 2) / (a * math.sqrt(3) + 2 * b) ** 2)


# print(area_proportion(3.8, 0.4))