from pathlib import Path

import click
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import FeatureDetector, Point2d
from visual_odometry.dataset import KittiOdometryDataset, KittiOdometrySequence
from visual_odometry.essential_matrix import get_homography_matrix
from visual_odometry.feature_detect_and_match import FeatureDetectMatcher
from visual_odometry.match_points_3d import TemporalFeatureTracker

mpl.use("QtAgg")


def draw_matches(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_matches: np.ndarray,
    right_matches: np.ndarray,
) -> np.ndarray:
    _, width, _ = left_image.shape
    image = np.concatenate([left_image, right_image], axis=1)

    for p1, p2 in zip(left_matches, right_matches, strict=True):
        cv2.line(
            image,
            (int(p1[0]), int(p1[1])),
            (int(p2[0] + width), int(p2[1])),
            (0, 255, 0),
            1,
        )
        cv2.circle(image, (int(p1[0]), int(p1[1])), 2, (0, 255, 0), -1)
        cv2.circle(image, (int(p2[0] + width), int(p2[1])), 2, (0, 255, 0), -1)
    return image


def play_sequence(sequence: KittiOdometrySequence) -> None:
    feature_extractor = FeatureDetectMatcher()
    tracker = TemporalFeatureTracker()

    previous_left_features = None
    previous_points_3d = None

    plt.ion()
    figure = plt.figure()
    axis = figure.add_subplot(projection="3d")
    trajectory = np.zeros((0, 3))

    for entry in sequence.iterate():
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            while cv2.waitKey() != ord(" "):
                pass

        left_features = feature_extractor.extract(entry.left)
        right_features = feature_extractor.extract(entry.right)
        left_matches, right_matches = feature_extractor.match(
            left_features, right_features
        )
        left_features, right_features = (
            left_features[left_matches],
            right_features[right_matches],
        )

        points_4d = np.asarray(
            cv2.triangulatePoints(
                sequence.calibration.p0,
                sequence.calibration.p1,
                left_features.keypoints.T,
                right_features.keypoints.T,
            ).T
        )
        valid_mask = np.abs(points_4d[:, 3]) > 1e-6
        left_features = left_features[valid_mask]
        right_features = right_features[valid_mask]

        points_3d = cv2.convertPointsFromHomogeneous(
            points_4d[valid_mask]
        ).squeeze()

        if previous_points_3d is None or previous_left_features is None:
            previous_points_3d = points_3d
            previous_left_features = left_features
            continue

        # Match previous points with current points
        previous_matches, current_matches = feature_extractor.match(
            previous_left_features, left_features
        )

        tracker.update(
            previous_points_3d[previous_matches],
            left_features[current_matches].keypoints,
            sequence.calibration.p0[:3, :3],
        )
        position, _ = tracker.current_pose()
        trajectory = np.concat([trajectory, position.reshape(1, 3)])

        axis.clear()
        # axis.scatter(
        #     points_3d[:, 0],
        #     points_3d[:, 1],
        #     points_3d[:, 2],
        #     s=5,
        # )
        axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])

        # axis.set_xlim(-5, 5)
        # axis.set_ylim(-5, 5)
        # axis.set_zlim(0, 30)
        # axis.set_box_aspect((10, 10, 30))

        figure.canvas.draw()
        figure.canvas.flush_events()

        image = draw_matches(
            entry.left,
            entry.right,
            left_features.keypoints,
            right_features.keypoints,
        )
        cv2.imshow("KITTI", image)

        previous_points_3d = points_3d
        previous_left_features = left_features

    cv2.destroyAllWindows()


@click.command()
@click.argument("dataset-folder", type=click.Path(path_type=Path))
def main(dataset_folder: Path):
    dataset = KittiOdometryDataset(dataset_folder)
    sequence = dataset.load("00")
    play_sequence(sequence)


if __name__ == "__main__":
    main()
