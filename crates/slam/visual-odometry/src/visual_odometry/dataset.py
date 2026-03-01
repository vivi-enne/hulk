from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable

import cv2
import numpy as np


@dataclass(frozen=True, kw_only=True)
class KittiOdometryItem:
    time: float
    left: np.ndarray
    right: np.ndarray


class KittyOdometryCalibration:
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray

    def __init__(self, calibration_path: Path) -> None:
        projections = []
        for line in calibration_path.read_text().splitlines():
            values = [float(value) for value in line.split()[1:]]
            projections.append(np.array(values).reshape((3, 4)))
        [self.p0, self.p1, self.p2, self.p3] = projections


class KittiOdometrySequence:
    folder: Path
    calibration: KittyOdometryCalibration

    def __init__(self, path: Path) -> None:
        self.calibration = KittyOdometryCalibration(path / "calib.txt")
        self.times = list(
            map(float, path.joinpath("times.txt").read_text().splitlines())
        )
        self.left_image_path = path / "image_0"
        self.right_image_path = path / "image_1"

    def iterate(self) -> Generator[KittiOdometryItem, None, None]:
        for idx, time in enumerate(self.times):
            left_image = np.asarray(
                cv2.imread(self.left_image_path / f"{idx:06}.png")
            )
            right_image = np.asarray(
                cv2.imread(self.right_image_path / f"{idx:06}.png")
            )

            yield KittiOdometryItem(
                time=time, left=left_image, right=right_image
            )


class KittiOdometryDataset:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, name: str) -> KittiOdometrySequence:
        path = self.path / "sequences" / name
        if not (path.exists() and path.is_dir()):
            raise ValueError(f"expected {path} to point to a kitti sequence")
        return KittiOdometrySequence(path)
