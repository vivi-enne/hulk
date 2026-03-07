import numpy as np


class LandmarkCandidateTracker:
    def __init__(self, minimum_consecutive_frames: int = 3) -> None:
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.feature_track_counts: np.ndarray | None = None

    def update(
        self,
        *,
        number_of_current_features: int,
        current_three_dimensional_points: np.ndarray,
        current_descriptors: np.ndarray,
        previous_match_indices: np.ndarray,
        current_match_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_track_counts is None:
            self.feature_track_counts = np.ones(
                number_of_current_features, dtype=np.int32
            )
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, current_descriptors.shape[1]), dtype=np.float32),
            )

        current_track_counts = np.ones(
            number_of_current_features, dtype=np.int32
        )

        if len(previous_match_indices) > 0:
            current_track_counts[current_match_indices] = (
                self.feature_track_counts[previous_match_indices] + 1
            )

        mature_feature_mask = (
            current_track_counts == self.minimum_consecutive_frames
        )

        proposed_three_dimensional_points = current_three_dimensional_points[
            mature_feature_mask
        ]
        proposed_descriptors = current_descriptors[mature_feature_mask]

        self.feature_track_counts = current_track_counts

        return proposed_three_dimensional_points, proposed_descriptors
