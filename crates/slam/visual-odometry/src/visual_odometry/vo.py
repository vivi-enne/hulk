from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.spatial.transform import RigidTransform, Rotation


class XFeatModel:
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


class VisualOdometry:
    left_calibration: np.ndarray
    right_calibration: np.ndarray
    device: torch.device
    xfeat_model: XFeatModel

    previous_left_features: None | ExtractedFeatures
    previous_points_3d: None | np.ndarray

    def __init__(
        self,
        left_calibration: np.ndarray,
        right_calibration: np.ndarray,
        device: str | int | torch.device = "cpu",
    ) -> None:
        self.left_calibration = left_calibration
        self.right_calibration = right_calibration
        self.device = torch.device(device)
        self.xfeat_model = torch.hub.load(  # pyright: ignore[reportAttributeAccessIssue]
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=128
        )
        self.xfeat_model.to(self.device)

        self.previous_left_features = None
        self.previous_points_3d = None

    def step(
        self, left_image: np.ndarray, right_image: np.ndarray
    ) -> RigidTransform:
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
            self.previous_left_features = left_features
            self.previous_points_3d = points_3d
            return RigidTransform.identity()

        landmark_match_indices, current_match_indices = self.match(
            self.previous_left_features.descriptors, left_features.descriptors
        )

        matched_point_cloud = self.previous_points_3d[landmark_match_indices]
        point_cloud_image_points = left_features.keypoints[
            current_match_indices
        ]
        pose_update = self.solve_pose_update(
            matched_point_cloud, point_cloud_image_points.numpy()
        )

        self.previous_points_3d = points_3d
        self.previous_left_features = left_features

        return (
            pose_update
            if pose_update is not None
            else RigidTransform.identity()
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

    @torch.compile()
    @torch.inference_mode()
    def match(
        self,
        descriptors_a: torch.Tensor,
        descriptors_b: torch.Tensor,
        minimum_cosine_similarity: float = 0.82,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        return index0, index1

    def triangulate_points(
        self,
        left_features: ExtractedFeatures,
        right_features: ExtractedFeatures,
    ) -> tuple[np.ndarray, ExtractedFeatures, ExtractedFeatures]:
        points_4d = np.asarray(
            cv2.triangulatePoints(
                self.left_calibration,
                self.right_calibration,
                left_features.keypoints.numpy().T,
                right_features.keypoints.numpy().T,
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
    ) -> None | RigidTransform:
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

        return RigidTransform.from_components(
            translation_vector.squeeze(1),
            Rotation.from_rotvec(rotation_vector.squeeze(1)),
        )
