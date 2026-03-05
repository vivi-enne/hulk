from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
import torch

# from .landmark_map import (
#     LocalLandmarkView,
#     MapLandmarkView,
#     filter_landmarks_in_view,
# )


class XFeatModel:
    dev: torch.device

    def to(self, device: torch.device) -> None: ...

    def detectAndCompute(
        self, images: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]: ...

    def match(
        self, left_descriptors: torch.Tensor, right_descriptor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True, kw_only=True)
class ExtractedFeatures:
    keypoints: torch.Tensor
    scores: torch.Tensor
    descriptors: torch.Tensor

    def __getitem__(
        self, index: int | np.ndarray | range | torch.Tensor
    ) -> "ExtractedFeatures":
        return ExtractedFeatures(
            keypoints=self.keypoints[index],
            scores=self.scores[index],
            descriptors=self.descriptors[index],
        )


class LandmarkCandidateTracker:
    minimum_consecutive_frames: int
    feature_track_counts: None | np.ndarray

    def __init__(self, minimum_consecutive_frames: int = 3) -> None:
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.feature_track_counts = None

    def update(
        self,
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


class VisualOdometry:
    left_calibration: np.ndarray
    right_calibration: np.ndarray
    device: torch.device
    xfeat_model: XFeatModel

    previous_left_features: None | ExtractedFeatures
    previous_points_3d: None | np.ndarray
    candidate_tracker: LandmarkCandidateTracker

    # fovx_tan: torch.Tensor
    # fovy_tan: torch.Tensor

    def __init__(
        self,
        left_calibration: np.ndarray,
        right_calibration: np.ndarray,
        # fovx: float,
        # fovy: float,
        minimum_consecutive_frames_to_spawn_landmark: int = 5,
        device: str | int | torch.device = "cpu",
    ) -> None:
        self.left_calibration = left_calibration
        self.right_calibration = right_calibration
        self.device = torch.device(device)
        self.xfeat_model = torch.hub.load(  # pyright: ignore[reportAttributeAccessIssue]
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=512
        )
        self.xfeat_model.dev = self.device
        self.xfeat_model.to(self.device)

        self.previous_left_features = None
        self.previous_points_3d = None
        self.candidate_tracker = LandmarkCandidateTracker(
            minimum_consecutive_frames_to_spawn_landmark
        )

        # self.fovx_tan = torch.scalar_tensor(fovx / 2.0, device=self.device)
        # self.fovy_tan = torch.scalar_tensor(fovy / 2.0, device=self.device)

    def setup_on_first_step(
        self, left_features: ExtractedFeatures, points_3d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.previous_left_features = left_features
        self.previous_points_3d = points_3d
        proposed_points, proposed_descriptors = self.candidate_tracker.update(
            number_of_current_features=len(points_3d),
            current_three_dimensional_points=points_3d,
            current_descriptors=left_features.descriptors.cpu().numpy(),
            previous_match_indices=np.empty(0, dtype=np.int32),
            current_match_indices=np.empty(0, dtype=np.int32),
        )
        return np.zeros(3), np.zeros(3), proposed_points, proposed_descriptors

    def step(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        # latest_map_to_camera: RigidTransform,
        # landmarks: MapLandmarkView,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # view = filter_landmarks_in_view(
        #     self.fovx_tan, self.fovy_tan, latest_map_to_camera, landmarks
        # )
        left_features, right_features = self.extract_features(
            [left_image, right_image]
        )
        left_match_indices, right_match_indices = self.match(
            left_features.descriptors, right_features.descriptors
        )
        left_features = left_features[left_match_indices]
        right_features = right_features[right_match_indices]

        (points_3d, left_features, right_features) = self.triangulate_points(
            left_features, right_features
        )
        if (
            self.previous_left_features is None
            or self.previous_points_3d is None
        ):
            return self.setup_on_first_step(left_features, points_3d)

        landmark_match_indices, current_match_indices = self.match(
            self.previous_left_features.descriptors, left_features.descriptors
        )

        matched_point_cloud = self.previous_points_3d[landmark_match_indices]
        point_cloud_image_points = left_features.keypoints[
            current_match_indices
        ]
        pose_update = self.solve_pose_update(
            matched_point_cloud, point_cloud_image_points.cpu().numpy()
        )
        translation, rotation = (
            pose_update
            if pose_update is not None
            else (np.zeros(3), np.zeros(3))
        )

        proposed_points, proposed_descriptors = self.candidate_tracker.update(
            number_of_current_features=len(points_3d),
            current_three_dimensional_points=points_3d,
            current_descriptors=left_features.descriptors.cpu().numpy(),
            previous_match_indices=landmark_match_indices,
            current_match_indices=current_match_indices,
        )

        self.previous_points_3d = points_3d
        self.previous_left_features = left_features

        return (
            translation.astype(np.float32),
            rotation.astype(np.float32),
            proposed_points,
            proposed_descriptors,
        )

    def extract_features(
        self, images: Iterable[np.ndarray]
    ) -> list[ExtractedFeatures]:
        # image is in [H, W, C] format
        # image_tensor: [N, 1, H, W]
        image_tensor = (
            torch.stack([torch.from_numpy(image[..., 0]) for image in images])
            .unsqueeze(1)
            .to(self.device)
        )
        outputs = self.xfeat_model.detectAndCompute(image_tensor)
        return [ExtractedFeatures(**output) for output in outputs]

    @torch.inference_mode()
    def match(
        self,
        descriptors_a: torch.Tensor,
        descriptors_b: torch.Tensor,
        minimum_cosine_similarity: float = 0.82,
    ) -> tuple[np.ndarray, np.ndarray]:
        similarity_matrix = torch.inner(descriptors_a, descriptors_b)

        maximum_similarity, match_indices_12 = similarity_matrix.max(dim=1)
        _, match_indices_21 = similarity_matrix.max(dim=0)

        index0 = torch.arange(
            len(match_indices_12), device=match_indices_12.device
        )
        is_mutual = match_indices_21[match_indices_12] == index0

        if minimum_cosine_similarity > 0:
            is_good = maximum_similarity > minimum_cosine_similarity
            valid_mask = is_mutual & is_good
        else:
            valid_mask = is_mutual

        index1 = match_indices_12[valid_mask]
        index0 = index0[valid_mask]

        return index0.cpu().numpy(), index1.cpu().numpy()

    def triangulate_points(
        self,
        left_features: ExtractedFeatures,
        right_features: ExtractedFeatures,
    ) -> tuple[np.ndarray, ExtractedFeatures, ExtractedFeatures]:
        points_4d = np.asarray(
            cv2.triangulatePoints(
                self.left_calibration,
                self.right_calibration,
                left_features.keypoints.cpu().numpy().T,
                right_features.keypoints.cpu().numpy().T,
            )
        ).T
        non_singular_mask = np.abs(points_4d[:, -1]) > 1e-6
        points_3d = cv2.convertPointsFromHomogeneous(
            points_4d[non_singular_mask]
        ).squeeze()

        return (
            points_3d,
            left_features[non_singular_mask],
            right_features[non_singular_mask],
        )

    def solve_pose_update(
        self, previous_points_3d: np.ndarray, current_points_2d: np.ndarray
    ) -> None | tuple[np.ndarray, np.ndarray]:
        if len(current_points_2d) < 4:
            return None

        success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(
            previous_points_3d,
            current_points_2d,
            self.left_calibration[:3, :3],
            None,
        )
        if not success:
            return None

        return (
            translation_vector.squeeze(1),
            rotation_vector.squeeze(1),
        )
