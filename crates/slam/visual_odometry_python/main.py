from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm
from visual_odometry.dataset import KittiOdometryDataset, KittiOdometrySequence
from visual_odometry.vo import VisualOdometry


def draw_matches(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_matches: np.ndarray | None,
    right_matches: np.ndarray | None,
) -> np.ndarray:
    _, width, _ = left_image.shape
    image = np.concatenate([left_image, right_image], axis=1)
    if left_matches is None or right_matches is None:
        return image

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


def play_sequence_headless(sequence: KittiOdometrySequence) -> None:
    vo = VisualOdometry(
        left_calibration=sequence.calibration.p0,
        right_calibration=sequence.calibration.p1,
        device="cuda",
    )
    for entry in tqdm(sequence.iterate()):
        _, _, proposed_points, _ = vo.step(entry.left, entry.right)
        print(f"proposed {len(proposed_points)} points")


def play_sequence(sequence: KittiOdometrySequence) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("QtAgg")

    vo = VisualOdometry(
        left_calibration=sequence.calibration.p0,
        right_calibration=sequence.calibration.p1,
        device="cuda",
    )
    current = RigidTransform.identity()

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

        update, proposed_points, _ = vo.step(entry.left, entry.right)
        print(f"proposed {len(proposed_points)} points")
        current = current * update

        trajectory = np.concat([trajectory, current.translation.reshape(1, 3)])

        axis.clear()
        # axis.scatter(
        #     points_3d[:, 0],
        #     points_3d[:, 1],
        #     points_3d[:, 2],
        #     s=5,
        # )
        axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        axis.axis("equal")

        # axis.set_xlim(-5, 5)
        # axis.set_ylim(-5, 5)
        # axis.set_zlim(0, 30)
        # axis.set_box_aspect((10, 10, 30))

        figure.canvas.draw()
        figure.canvas.flush_events()

        image = draw_matches(
            entry.left,
            entry.right,
            None,
            None,
        )
        cv2.imshow("KITTI", image)

    cv2.destroyAllWindows()


@click.command()
@click.argument("kitti-sequence", type=click.Path(path_type=Path))
@click.option("--headless", is_flag=True, type=click.BOOL)
def main(*, kitti_sequence: Path, headless: bool):
    dataset = KittiOdometryDataset(kitti_sequence.parent)
    sequence = dataset.load(kitti_sequence.name)
    if headless:
        play_sequence_headless(sequence)
    else:
        play_sequence(sequence)


if __name__ == "__main__":
    main()
