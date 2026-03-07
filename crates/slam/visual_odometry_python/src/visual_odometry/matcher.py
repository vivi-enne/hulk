from dataclasses import dataclass

import cv2
import numpy as np
from visual_odometry_rust import Isometry, XFeatOutput

from .candidate_tracker import LandmarkCandidateTracker


class DimensionMismatchError(Exception):
    def __init__(
        self, actual: tuple[int, ...], expected: tuple[int, ...]
    ) -> None:
        super().__init__(f"expected shape {expected}, but got {actual}")

    @staticmethod
    def raise_if_different(
        actual: tuple[int, ...], expected: tuple[int, ...]
    ) -> None:
        if actual != expected:
            raise DimensionMismatchError(actual, expected)


@dataclass(frozen=True)
class ExtractedFeatures:
    keypoints: np.ndarray
    scores: np.ndarray
    descriptors: np.ndarray

    def __post_init__(self) -> None:
        n = self.keypoints.shape[0]
        DimensionMismatchError.raise_if_different(self.keypoints.shape, (n, 2))
        DimensionMismatchError.raise_if_different(self.scores.shape, (n,))
        DimensionMismatchError.raise_if_different(
            self.descriptors.shape, (n, 64)
        )

    def __getitem__(self, index: np.ndarray) -> "ExtractedFeatures":
        return ExtractedFeatures(
            self.keypoints[index],
            self.scores[index],
            self.descriptors[index],
        )


class VisualOdometryMatcher:
    def __init__(
        self, left_calibration: np.ndarray, right_calibration: np.ndarray
    ) -> None:
        self.left_calibration = left_calibration
        self.right_calibration = right_calibration

        self.previous_features = None
        self.previous_points = None

        self.candidate_tracker = LandmarkCandidateTracker()

    def step(
        self, features: XFeatOutput
    ) -> tuple[Isometry, np.ndarray, np.ndarray]:
        left, right = xfeat_to_features(features)
        left_matches, right_matches = match(left.descriptors, right.descriptors)
        left_filtered = left[left_matches]
        right_filtered = right[right_matches]

        triangulated_points, non_singular_feature_mask = triangulate_points(
            left_filtered,
            right_filtered,
            self.left_calibration,
            self.right_calibration,
        )

        left_filtered_non_singular = left_filtered[non_singular_feature_mask]

        if self.previous_features is None or self.previous_points is None:
            self.previous_features = left_filtered_non_singular
            self.previous_points = triangulated_points
            return (
                Isometry.zero(),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 32), dtype=np.float32),
            )

        previous_matches, current_matches = match(
            self.previous_features.descriptors,
            left_filtered_non_singular.descriptors,
        )

        matched_points = self.previous_points[previous_matches]
        image_positions = left_filtered_non_singular[current_matches].keypoints

        odometry = solve_pose_transform(
            matched_points, image_positions, self.left_calibration
        )

        self.previous_points = triangulated_points
        self.previous_features = left_filtered_non_singular

        proposed_points, proposed_descriptors = self.candidate_tracker.update(
            number_of_current_features=len(triangulated_points),
            current_three_dimensional_points=triangulated_points,
            current_descriptors=left_filtered_non_singular.descriptors,
            previous_match_indices=previous_matches,
            current_match_indices=current_matches,
        )

        return (
            odometry if odometry is not None else Isometry.zero(),
            proposed_points,
            proposed_descriptors,
        )


def xfeat_to_features(
    features: XFeatOutput,
) -> tuple[ExtractedFeatures, ExtractedFeatures]:
    keypoints = features.keypoints.astype(np.float32)
    scores = features.scores
    descriptors = features.descriptors

    a = ExtractedFeatures(keypoints[0], scores[0], descriptors[0])
    b = ExtractedFeatures(keypoints[1], scores[1], descriptors[1])
    return (a, b)


def match(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    minimum_cosine_similarity: float = 0.82,
) -> tuple[np.ndarray, np.ndarray]:
    from .warp_matcher import match as wp_match

    return wp_match(
        descriptors_a,
        descriptors_b,
        minimum_cosine_similarity=minimum_cosine_similarity,
    )

    similarity_matrix = np.inner(descriptors_a, descriptors_b)

    match_indices_12 = np.argmax(similarity_matrix, axis=1)

    # Extracting values via indexing avoids a redundant full-matrix reduction
    index0 = np.arange(len(match_indices_12))
    maximum_similarity = similarity_matrix[index0, match_indices_12]

    match_indices_21 = np.argmax(similarity_matrix, axis=0)

    is_mutual = match_indices_21[match_indices_12] == index0

    if minimum_cosine_similarity > 0:
        is_good = maximum_similarity > minimum_cosine_similarity
        valid_mask = is_mutual & is_good
    else:
        valid_mask = is_mutual

    index1 = match_indices_12[valid_mask]
    index0 = index0[valid_mask]

    return index0, index1


def triangulate_points(
    left: ExtractedFeatures,
    right: ExtractedFeatures,
    left_calibration: np.ndarray,
    right_calibration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(left.keypoints) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)

    homogeneous_triangulation = np.asarray(
        cv2.triangulatePoints(
            left_calibration,
            right_calibration,
            left.keypoints.T,
            right.keypoints.T,
        )
    ).T
    is_non_singular = np.abs(homogeneous_triangulation[..., -1]) > 1e-6

    valid_points = homogeneous_triangulation[is_non_singular]
    if len(valid_points) == 0:
        return np.empty((0, 3), dtype=np.float32), is_non_singular

    converted_points = cv2.convertPointsFromHomogeneous(valid_points)

    return converted_points.reshape(-1, 3), is_non_singular


def solve_pose_transform(
    previous_points: np.ndarray,
    current_image_points: np.ndarray,
    calibration: np.ndarray,
) -> Isometry | None:
    if len(current_image_points) < 4 or len(previous_points) < 4:
        return None

    success, rotation, translation, _ = cv2.solvePnPRansac(
        previous_points, current_image_points, calibration[:3, :3], None
    )
    if not success:
        return None

    return Isometry(
        translation.squeeze().astype(np.float32),
        rotation.squeeze().astype(np.float32),
    )
