from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import RigidTransform


@dataclass(frozen=True)
class MapLandmarkView:
    # [N, 3] array
    points: np.ndarray
    # [N, 32] array
    descriptors: np.ndarray


@dataclass(frozen=True)
class LocalLandmarkView:
    # [N, 3] array
    points: torch.FloatTensor
    # [N, 32] array
    descriptors: torch.FloatTensor


def filter_landmarks_in_view(
    fovx_tan: torch.Tensor,
    fovy_tan: torch.Tensor,
    map_to_camera: RigidTransform,
    landmarks: MapLandmarkView,
) -> LocalLandmarkView:
    device = fovx_tan.device

    rotation = torch.from_numpy(map_to_camera.rotation.as_matrix()).to(device)
    translation = torch.from_numpy(map_to_camera.translation).to(device)
    points = torch.from_numpy(landmarks.points).to(device)

    assert points.size(1) == 3
    points_in_camera = points @ rotation.T + translation
    depth = points_in_camera[:, 2]
    in_viewport = (points[:, 0].abs() <= depth * fovx_tan) & (
        points[:, 1].abs() <= depth * fovy_tan
    )

    return LocalLandmarkView(
        torch.FloatTensor(points_in_camera[in_viewport]),
        torch.FloatTensor(
            torch.from_numpy(landmarks.descriptors).to(device)[in_viewport]
        ),
    )
