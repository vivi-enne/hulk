import datetime
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import polars.selectors as ps

# from scipy.stats import linregress
from sklearn.linear_model import LinearRegression


def main():
    robot_number = 40
    df = pl.read_parquet(f"dataframe_sensor_data_{robot_number}.parquet")
    positions = {
        "head": ["pitch"],
        # "yaw"],
        "left_arm": ["shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll", "wrist_yaw", "hand"],
        "right_arm": ["shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll", "wrist_yaw", "hand"],
        "left_leg": [
            "ankle_pitch",
            "ankle_roll",
            "hip_pitch",
            "hip_roll",
            "hip_yaw_pitch",
            "knee_pitch",
        ],
        "right_leg": ["ankle_pitch", "ankle_roll", "hip_pitch", "hip_roll", "hip_yaw_pitch", "knee_pitch"],
    }
    joints = [f"{part}.{joint}" for part, joints in positions.items() for joint in joints]

    print(df)

    results = []

    x_column = "motor_commands.positions"
    regression_column = "sensor_data.positions"

    # k_history_values = [1, 2]  # np.arange(0, 2)
    k_history_values = np.arange(0, 5)
    k_horizon_values = np.arange(0, 1 // 0.012)
    # k_horizon_values = np.arange(0, 2)
    # k_horizon_values = np.array([0, 5, 10, 15])

    unique_joints = len(joints)
    unique_k_horizon = len(k_horizon_values)

    matrix = np.zeros((unique_joints, unique_k_horizon))

    for k_history in k_history_values:
        for y, joint in enumerate(joints):
            # for x, k_horizon in enumerate(k_horizon_values):
            print(f"{joint = }, {k_history = }")  # , {k_horizon = }")
            error = predict_error(df, x_column, regression_column, joint=joint, k_history=k_history + 1, k_horizon_values=k_horizon_values)
            results.append((joint, k_history, k_horizon_values, error))
            matrix[y, :] = error

        plot_error_matrix(matrix, unique_k_horizon, unique_joints, k_horizon_values, joints, k_history, robot_number)


def plot_error_matrix(matrix, unique_k_horizon, unique_joints, k_horizon_values, joints, k_history, robot_number):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use imshow to plot the heatmap
    cax = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="upper")

    # Set axis labels
    ax.set_xlabel("k_horizon", fontsize=12)
    ax.set_ylabel("Input String", fontsize=12)
    ax.set_title(f"Heatmap of Average Error {robot_number}, {k_history}", fontsize=14)

    # Customize x and y ticks
    ax.set_xticks(range(unique_k_horizon))
    ax.set_xticklabels(k_horizon_values)
    ax.set_yticks(range(unique_joints))
    ax.set_yticklabels(joints)

    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label("Average Error [deg]", fontsize=12)

    # Show plot
    plt.tight_layout()
    fig.savefig(f"images/{datetime.datetime.now()}_heatmap_{robot_number}_{k_history}.jpeg")
    # plt.show()


def predict_error(df: pl.DataFrame, x_column: str, regression_column: str, joint: str = "left_leg.knee_pitch", k_history: int = 3, k_horizon_values=[2]) -> float:
    results = []

    y_column = regression_column
    d = df

    df_walk = (
        df.with_columns(pl.col("motion_command.Walk").is_not_null().rle_id().alias("chunk_id"))
        .filter(pl.col("motion_command.Walk").is_not_null())
        .select(~ps.starts_with("motion_command"))
    )

    df_shifted = (
        df_walk.with_columns([pl.col(x_column).shift(i).over("chunk_id").alias(f"feature_{i}") for i in range(k_history)])
        .with_columns(pl.col(y_column).shift(-k_horizon).over("chunk_id").alias(f"{y_column}_shifted_{k_horizon}") for k_horizon in k_horizon_values)
        .drop_nulls()
    )

    unpacked_df = (
        df_shifted.with_columns(unpack_joints([x_column, "sensor_data.positions"]))
        .with_columns(unpack_joints([f"feature_{i}" for i in range(k_history)]))
        .with_columns(unpack_joints([f"sensor_data.positions_shifted_{k_horizon}" for k_horizon in k_horizon_values]))
    )

    # df_joint = unpacked_df.select(ps.ends_with("left_leg.knee_pitch") & ~ps.starts_with("sensor_data.positions."))
    x = unpacked_df.select(ps.starts_with("feature") & ps.ends_with(f"{joint}"))  # | ps.starts_with("log_time"))
    y = unpacked_df.select(ps.starts_with("sensor_data.positions_shifted") & ps.ends_with(f"{joint}"))  # | ps.starts_with("log_time"))

    if x.is_empty():
        x = unpacked_df.select(ps.starts_with(x_column) & ps.ends_with(joint))

    # TODO: split at chunk border
    number_train = int(2 * len(x) / 3)
    x_train = x[:number_train].to_numpy()
    y_train = y[:number_train].to_numpy()

    # x_train = np.random.random(x_train.shape)
    # y_train = np.random.random(y_train.shape)

    # fig, axes = plt.subplots(1, 4)
    # axes[0].plot(unpacked_df["log_time"][:number_train], np.rad2deg(y_train[:, 0]))

    # axes[1].plot(x_train[:, 0], y_train[:, 0])

    model = LinearRegression()
    model.fit(x_train, y_train)

    x_val = x[number_train + 1 :].to_numpy()
    y_val = y[number_train + 1 :].to_numpy()

    # x_val = np.random.random(x_val.shape)
    # y_val = np.random.random(y_val.shape)

    pred = model.predict(x_val)

    # axes[1].plot(unpacked_df["log_time"][number_train + 1 :], np.rad2deg(y_val[:, 0]), label="y_val[:,0]")
    # axes[1].plot(unpacked_df["log_time"][number_train + 1 :], np.rad2deg(pred[:, 0]), label="pred[:,0]")
    # axes[1].set_xlabel("log_time", fontsize=12)
    # axes[1].set_title(f"{k_horizon_values[0] = }", fontsize=14)
    # plt.legend()

    # axes[2].plot(unpacked_df["log_time"][number_train + 1 :], np.rad2deg(y_val[:, 1]), label="y_val[:,1]")
    # axes[2].plot(unpacked_df["log_time"][number_train + 1 :], np.rad2deg(pred[:, 1]), label="pred[:,1]")
    # axes[2].set_xlabel("log_time", fontsize=12)
    # axes[2].set_title(f"{k_horizon_values[1] = }", fontsize=14)
    # plt.legend()

    # axes[3].plot(unpacked_df["log_time"][number_train + 1 :], np.rad2deg(y_val[:, 2]), label="y_val[:,2]")
    # axes[3].plot(unpacked_df["log_time"][number_train + 1 :], np.rad2deg(pred[:, 2]), label="pred[:,2]")
    # axes[3].set_xlabel("log_time", fontsize=12)
    # axes[3].set_title(f"{k_horizon_values[2] = }", fontsize=14)
    # plt.legend()
    # plt.show()

    error = abs(sum(pred - y_val)) / len(pred)
    error = np.rad2deg(error)

    return error


def unpack_joints(columns) -> list[pl.Expr]:
    positions = {
        "head": ["pitch", "yaw"],
        "left_arm": ["shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll", "wrist_yaw", "hand"],
        "right_arm": ["shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll", "wrist_yaw", "hand"],
        "left_leg": ["ankle_pitch", "ankle_roll", "hip_pitch", "hip_roll", "hip_yaw_pitch", "knee_pitch"],
        "right_leg": ["ankle_pitch", "ankle_roll", "hip_pitch", "hip_roll", "hip_yaw_pitch", "knee_pitch"],
    }

    return [pl.col(column).struct.field(part).struct.field(joint).alias(f"{column}.{part}.{joint}") for column in columns for part, joints in positions.items() for joint in joints]


if __name__ == "__main__":
    main()
