import numpy as np
import numpy.typing as npt

class Viewport:
    fov_x: float
    fov_y: float

class Isometry:
    def __new__(
        cls,
        translation: npt.NDArray[np.float32],
        rotation: npt.NDArray[np.float32],
    ) -> "Isometry": ...
    @staticmethod
    def zero() -> "Isometry": ...

class ExtractedFeatures:
    keypoints: npt.NDArray[np.float32]
    descriptors: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]

class XFeatOutput:
    keypoints: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]
    descriptors: npt.NDArray[np.float32]
