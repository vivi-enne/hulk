import cv2
import numpy as np


def get_homography_matrix(
    matches_left: np.ndarray, matches_right: np.ndarray
) -> np.ndarray:
    H, mask = cv2.findHomography(matches_left, matches_right, cv2.RANSAC, 5.0)

    cv2.triangulatePoints()
    return H
