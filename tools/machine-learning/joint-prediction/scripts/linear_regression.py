import datetime
from collections.abc import Callable, Iterable
from typing import Self

import click
import numpy as np
import plotly.graph_objects as go
import polars as pl
import polars.selectors as ps
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from polars._typing import ColumnNameOrSelector
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)


class LastSensordataRegressor(BaseEstimator, RegressorMixin):
    index_of_last_sensor_data: int

    def __init__(self, index_of_last_sensor_data: int = 0) -> None:
        self.index_of_last_sensor_data = index_of_last_sensor_data

    def fit(
        self,
        x: NDArray[np.float64] | pl.DataFrame,
        y: NDArray[np.float64] | pl.DataFrame,
    ) -> Self:
        _ = x, y
        return self

    def predict(
        self,
        x: NDArray[np.float64] | pl.DataFrame,
    ) -> NDArray[np.float64]:
        if isinstance(x, pl.DataFrame):
            x = x.to_numpy()
        return x[:, self.index_of_last_sensor_data]


# def fit_constrained_ridge(X_train, y_train, alpha: float = 1.0) -> Ridge:
#     model = Ridge(fit_intercept=False)
#     n_features = X_train.shape[1]
#
#     w = cp.Variable(n_features)
#     objective = cp.Minimize(
#         cp.sum_squares(X_train @ w - y_train) + alpha * cp.sum_squares(w),
#     )
#     constraints = [cp.sum(w) == 1]
#     problem = cp.Problem(objective, constraints)
#     problem.solve()
#     model.coef_ = w.value
#     model.intercept_ = 0.0
#     return model


def xyz_name(index: int) -> str:
    return {
        0: "x",
        1: "y",
        2: "z",
    }[index]


def unnest_structs(
    df: pl.DataFrame,
    exprs: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
    *,
    depth: int = 1,
    list_to_struct_fields: list[str] | Callable[[int], str] = xyz_name,
) -> pl.DataFrame:
    if depth < 1:
        return df

    nested_df = df.select(exprs)

    for _ in range(depth):
        current_cols = nested_df.columns
        unnested_columns: list[pl.DataFrame] = []

        for current_col in current_cols:
            series = nested_df[current_col]
            if isinstance(series.dtype, pl.List):
                series = series.list.to_struct(fields=list_to_struct_fields)

            unnested = series.struct.unnest()

            rename_map = {
                subcol: f"{current_col}.{subcol}"
                for subcol in unnested.columns
            }
            renamed = unnested.rename(rename_map)
            unnested_columns.append(renamed)

        nested_df = pl.concat(
            [nested_df.drop(current_cols), *unnested_columns],
            how="horizontal",
        )

    return pl.concat([df.drop(exprs), nested_df], how="horizontal")


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
    x: pl.DataFrame,
    y: pl.DataFrame,
    k_history: int,
    k_horizon: int,
    group_key: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    past_features = [
        shift_expr(feature, k, group_key)
        for feature in x.columns
        if feature != group_key
        for k in range(k_history + 1)
    ]

    future_features = [
        shift_expr(target, -k_horizon, group_key)
        for target in y.columns
        if target != group_key
    ]
    x = x.select(past_features)
    y = y.select(future_features)

    is_not_null = (
        x.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
        & y.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
    )

    return x.filter(is_not_null), y.filter(is_not_null)


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
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
) -> float:
    baseline = LastSensordataRegressor()
    baseline_prediction = baseline.fit(x_train, y_train).predict(x_test)
    baseline_error = np.square(
        baseline_prediction - y_test.to_numpy().flatten(),
    )
    model = HistGradientBoostingRegressor("absolute_error")
    model.fit(x_train, y_train.to_series())
    prediction = model.predict(x_test)
    model_error = np.square(prediction - y_test.to_numpy().flatten())

    return np.rad2deg(np.mean(model_error - baseline_error))


def fit_model(
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
    k_history: int,
    k_horizon: int,
    group_key: str,
) -> float:
    x_train, y_train = generate_fit_data(
        x_train,
        y_train,
        k_history,
        k_horizon,
        group_key,
    )
    x_test, y_test = generate_fit_data(
        x_test,
        y_test,
        k_history,
        k_horizon,
        group_key,
    )
    return improve_over_baseline(x_train, y_train, x_test, y_test)


@click.command()
@click.argument(
    "parquet",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
def main(parquet: str) -> None:
    data = pl.read_parquet(parquet)
    data = filter_walk(data)

    data = unnest_structs(data, "motor_commands.positions", depth=2)
    data = unnest_structs(data, "sensor_data.positions", depth=2)
    data = unnest_structs(
        data,
        "sensor_data.inertial_measurement_unit",
        depth=2,
    )

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

    train_df, test_df = train_test_split(data, "chunk_id", test_fraction=0.2)
    x_train, y_train = xy_split(train_df, features, targets)
    x_test, y_test = xy_split(test_df, features, targets)

    # k_history_values = np.arange(0, 50, 1)
    # k_horizon_values = np.arange(0, 22)
    # scores = np.zeros((k_history_values.size, k_horizon_values.size))

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
    # scores = np.zeros_like(k_horizon_values, dtype=np.float64)
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
    # fig = go.Figure()
    # fig.add_bar(x=k_horizon_values, y=scores)
    # fig.show()

    k_history = 25
    k_horizon = 2
    x_train, y_train = generate_fit_data(
        x_train,
        y_train,
        k_history,
        k_horizon,
        "chunk_id",
    )
    x_test, y_test = generate_fit_data(
        x_test,
        y_test,
        k_history,
        k_horizon,
        "chunk_id",
    )
    model = HistGradientBoostingRegressor()
    # model = RandomForestRegressor()
    model.fit(x_train, y_train.to_series())
    prediction = model.predict(x_test)
    # print(model.coef_.tolist(), model.intercept_)

    baseline = LastSensordataRegressor().fit(x_train, y_train).predict(x_test)

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
    matrix: NDArray[np.float64],
    k_history_values: NDArray[np.int64],
    k_horizon_values: NDArray[np.int64],
) -> None:
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
    fig.write_html(
        f"images/{datetime.datetime.now(tz=datetime.UTC)}_heatmap.html",
    )


if __name__ == "__main__":
    main()
