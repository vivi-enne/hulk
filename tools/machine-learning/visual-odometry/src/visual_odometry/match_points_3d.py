import cv2
import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation


class TemporalFeatureTracker:
    def __init__(self) -> None:
        self.camera_to_world = RigidTransform(np.eye(4))

    def current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns position and heading
        """
        heading = self.camera_to_world.rotation.as_matrix()[:, -1]
        position = self.camera_to_world.translation

        return (position, heading)

    def update(
        self,
        previous_points_3d: np.ndarray,
        current_points_2d: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> bool:
        if len(current_points_2d) < 4:
            return False
        success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(
            previous_points_3d,
            current_points_2d,
            camera_matrix,
            None,
        )
        previous_camera_to_new_camera = RigidTransform.from_components(
            translation_vector.squeeze(1),
            Rotation.from_rotvec(rotation_vector.squeeze(1)),
        )
        self.camera_to_world = (
            self.camera_to_world * previous_camera_to_new_camera
        )
        return success
