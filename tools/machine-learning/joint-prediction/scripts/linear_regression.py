import datetime

import click
import cvxpy as cp
import numpy as np
import plotly.graph_objects as go
import polars as pl
import polars.selectors as ps
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge


class LastSensordataRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, index_of_last_sensor_data: int = 0):
        self.index_of_last_sensor_data = index_of_last_sensor_data

    def fit(self, X, y):
        return self

    def predict(self, X):
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        return X[:, self.index_of_last_sensor_data]


def fit_constrained_ridge(X_train, y_train, alpha: float = 1.0) -> Ridge:
    model = Ridge(fit_intercept=False)
    n_features = X_train.shape[1]

    w = cp.Variable(n_features)
    objective = cp.Minimize(
        cp.sum_squares(X_train @ w - y_train) + alpha * cp.sum_squares(w),
    )
    constraints = [cp.sum(w) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    model.coef_ = w.value
    model.intercept_ = 0.0
    return model


def xyz_name(index: int) -> str:
    return {
        0: "x",
        1: "y",
        2: "z",
    }[index]


def unnest_structs(df: pl.DataFrame, name: str, depth: int) -> pl.DataFrame:
    column = df.select(name)
    for i in range(depth):
        struct_cols = column.columns
        new_columns = []

        for col in struct_cols:
            values = column[col]
            if isinstance(values.dtype, pl.List):
                values = values.list.to_struct(fields=xyz_name)

            unnested = values.struct.unnest()
            renamed = unnested.rename(
                {sub_col: f"{col}.{sub_col}" for sub_col in unnested.columns},
            )
            new_columns.append(renamed)
        column = pl.concat(
            [column.drop(struct_cols), *new_columns],
            how="horizontal",
        )

    return pl.concat([df.drop(name), column], how="horizontal")


def train_test_split(
    df: pl.DataFrame,
    group_key: str,
    test_fraction: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    test_ids = df[group_key].unique().sample(fraction=test_fraction)
    train_df = df.filter(~pl.col(group_key).is_in(test_ids))
    test_df = df.filter(pl.col(group_key).is_in(test_ids))
    return train_df, test_df


def xy_split(
    df: pl.DataFrame,
    feature_prefix: list[str],
    target_prefix: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        df.select([ps.starts_with(prefix) for prefix in feature_prefix]),
        df.select([ps.starts_with(prefix) for prefix in target_prefix]),
    )


def shift_expr(feature: str, k: int, group: str) -> pl.Expr:
    return pl.col(feature).shift(k).over(group).alias(f"{feature}@t{-k:+}")


def generate_fit_data(
    X: pl.DataFrame,
    y: pl.DataFrame,
    k_history: int,
    k_horizon: int,
    group_key: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    past_features = [
        shift_expr(feature, k, group_key)
        for feature in X.columns
        if feature != group_key
        for k in range(k_history + 1)
    ]

    future_features = [
        shift_expr(target, -k_horizon, group_key)
        for target in y.columns
        if target != group_key
    ]
    X = X.select(past_features)
    y = y.select(future_features)

    is_not_null = (
        X.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
        & y.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
    )

    return X.filter(is_not_null), y.filter(is_not_null)


def filter_walk(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("motion_command.Walk")
            .is_not_null()
            .rle_id()
            .alias("chunk_id"),
        )
        .filter(pl.col("motion_command.Walk").is_not_null())
        .select(~ps.starts_with("motion_command"))
    )


def improve_over_baseline(
    X_train: pl.DataFrame,
    y_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_test: pl.DataFrame,
) -> float:
    baseline = LastSensordataRegressor()
    baseline_prediction = baseline.fit(X_train, y_train).predict(X_test)
    baseline_error = np.square(
        baseline_prediction - y_test.to_numpy().flatten(),
    )
    # model = fit_constrained_ridge(X_train.to_numpy(), y_train.to_numpy().flatten())
    # model = Ridge()
    model = HistGradientBoostingRegressor("absolute_error")
    prediction = model.fit(X_train, y_train.to_series()).predict(X_test)
    model_error = np.square(prediction - y_test.to_numpy().flatten())

    return np.rad2deg(np.mean(model_error - baseline_error))


def fit_model(
    X_train: pl.DataFrame,
    y_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_test: pl.DataFrame,
    k_history: int,
    k_horizon: int,
    group_key: str,
) -> float:
    X_train, y_train = generate_fit_data(
        X_train,
        y_train,
        k_history,
        k_horizon,
        group_key,
    )
    X_test, y_test = generate_fit_data(
        X_test,
        y_test,
        k_history,
        k_horizon,
        group_key,
    )
    return improve_over_baseline(X_train, y_train, X_test, y_test)


@click.command()
@click.argument(
    "parquet",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
def main(parquet: str):
    df = pl.read_parquet(parquet)
    df = filter_walk(df)

    df = unnest_structs(df, "motor_commands.positions", 2)
    df = unnest_structs(df, "sensor_data.positions", 2)
    df = unnest_structs(df, "sensor_data.inertial_measurement_unit", 2)

    features = [
        "chunk_id",
        "sensor_data.inertial_measurement_unit.angular_velocity.y",
        "sensor_data.inertial_measurement_unit.angular_velocity.x",
        "sensor_data.inertial_measurement_unit.angular_velocity.z",
        "sensor_data.inertial_measurement_unit.linear_acceleration.y",
        "sensor_data.inertial_measurement_unit.linear_acceleration.x",
        "sensor_data.inertial_measurement_unit.linear_acceleration.z",
        "sensor_data.inertial_measurement_unit.roll_pitch.x",
        "sensor_data.inertial_measurement_unit.roll_pitch.y",
        "motor_commands.positions.left_leg.ankle_pitch",
        "motor_commands.positions.right_leg.ankle_pitch",
        "sensor_data.positions.left_leg.ankle_pitch",
        "sensor_data.positions.right_leg.ankle_pitch",
    ]
    targets = [
        "chunk_id",
        # "sensor_data.positions.left_leg.ankle_pitch",
        "sensor_data.inertial_measurement_unit.angular_velocity.y",
    ]

    train_df, test_df = train_test_split(df, "chunk_id", test_fraction=0.2)
    X_train, y_train = xy_split(train_df, features, targets)
    X_test, y_test = xy_split(test_df, features, targets)

    k_history_values = np.arange(0, 50, 1)
    k_horizon_values = np.arange(0, 22)
    scores = np.zeros((k_history_values.size, k_horizon_values.size))

    # with tqdm(total=k_history_values.size * k_horizon_values.size) as pbar:
    #     for (i, k_history), (j, k_horizon) in product(
    #         enumerate(k_history_values), enumerate(k_horizon_values)
    #     ):
    #         scores[i, j] = fit_model(
    #             X_train,
    #             y_train,
    #             X_test,
    #             y_test,
    #             k_history,
    #             k_horizon,
    #             "chunk_id",
    #         )
    #         pbar.update(1)

    # plot_error_matrix(
    #     scores,
    #     k_history_values,
    #     k_horizon_values,
    # )

    k_history = 25
    scores = np.zeros_like(k_horizon_values, dtype=np.float64)
    # for j, k_horizon in enumerate(tqdm(k_horizon_values)):
    #     scores[j] = fit_model(
    #         X_train,
    #         y_train,
    #         X_test,
    #         y_test,
    #         k_history,
    #         k_horizon,
    #         "chunk_id",
    #     )
    fig = go.Figure()
    fig.add_bar(x=k_horizon_values, y=scores)
    fig.show()

    k_history = 25
    k_horizon = 2
    X_train, y_train = generate_fit_data(
        X_train,
        y_train,
        k_history,
        k_horizon,
        "chunk_id",
    )
    X_test, y_test = generate_fit_data(
        X_test,
        y_test,
        k_history,
        k_horizon,
        "chunk_id",
    )
    model = HistGradientBoostingRegressor()
    # model = RandomForestRegressor()
    model.fit(X_train, y_train.to_series())
    prediction = model.predict(X_test)
    # print(model.coef_.tolist(), model.intercept_)

    baseline = LastSensordataRegressor().fit(X_train, y_train).predict(X_test)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_scatter(
        y=prediction,
        mode="lines",
        name="Prediction",
        row=1,
        col=1,
    )
    fig.add_scatter(y=baseline, mode="lines", name="Baseline", row=1, col=1)
    fig.add_scatter(
        y=y_test.to_numpy().flatten(),
        mode="lines",
        name="Ground Truth",
        row=1,
        col=1,
    )
    fig.add_scatter(
        y=np.abs(prediction - y_test.to_numpy().flatten()),
        mode="lines",
        name="Error",
        row=2,
        col=1,
    )
    fig.show()


def plot_error_matrix(
    matrix,
    k_history_values,
    k_horizon_values,
):
    fig = go.Figure()
    fig.add_contour(
        z=matrix,
        x=k_horizon_values,
        y=k_history_values,
        colorscale="inferno",
        showscale=True,
        text=matrix,
        texttemplate="%{text:.4f}°",
    )
    fig.update_layout(
        xaxis_title="Horizon Values",
        yaxis_title="History Values",
    )

    fig.show()
    fig.write_html(f"images/{datetime.datetime.now()}_heatmap.html")


if __name__ == "__main__":
    main()
