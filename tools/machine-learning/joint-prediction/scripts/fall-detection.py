import click
from mcap.reader import make_reader
import msgpack
from datetime import datetime
from typing import Callable, Any
from tqdm import tqdm
import polars as pl
import hashlib
from pathlib import Path
import plotly.graph_objects as go
import numpy as np


def sha256sum(filename: str) -> str:
    with open(filename, "rb", buffering=0) as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class ReadTopic:
    def __init__(self, name: str, extractor: None | Callable[[Any], Any] = None):
        self.name = name
        self.extractor = extractor or ReadTopic.identity

    @staticmethod
    def identity(value: Any) -> Any:
        return value


class ReadFallState(ReadTopic):
    def __init__(self):
        super().__init__("Control.main_outputs.fall_state", ReadFallState.extractor)

    @staticmethod
    def extractor(value: dict[str, Any] | str) -> str:
        if isinstance(value, str):
            return value

        if "Falling" in value:
            return f"Falling{next(iter(value['Falling']['direction'].keys()))}"

        if "Fallen" in value:
            return f"Fallen{value['Fallen']['kind']}"

        if "StandingUp" in value:
            return f"StandingUp{value['StandingUp']['kind']}"

        raise ValueError(f"Unknown fall state: {value}")


class ReadSensorValues(ReadTopic):
    def __init__(self):
        super().__init__("Control.main_outputs.sensor_data", ReadSensorValues.extractor)

    @staticmethod
    def extractor(value: dict[str, Any]) -> dict[str, Any]:
        left_fsrs = sum(value["force_sensitive_resistors"]["left"].values())
        right_fsrs = sum(value["force_sensitive_resistors"]["right"].values())
        linear_acceleration = value["inertial_measurement_unit"]["linear_acceleration"]
        roll_pitch = value["inertial_measurement_unit"]["roll_pitch"]
        angular_velocity = value["inertial_measurement_unit"]["angular_velocity"]

        return {
            "left_fsrs": left_fsrs,
            "right_fsrs": right_fsrs,
            "linear_acceleration": linear_acceleration,
            "roll_pitch": roll_pitch,
            "angular_velocity": angular_velocity,
        }


def iterate_messages(reader, topics: list[ReadTopic]):
    topic_table = {topic.name: topic.extractor for topic in topics}
    statistics = reader.get_summary().statistics.channel_message_counts
    message_count = sum(
        statistics[id]
        for id, channel in reader.get_summary().channels.items()
        if channel.topic in topic_table
    )
    pbar = tqdm(total=message_count)
    for i, (_, channel, message) in enumerate(
        reader.iter_messages(topics=topic_table.keys())
    ):
        log_time = datetime.fromtimestamp(message.log_time * 1e-9)
        message_data = msgpack.unpackb(message.data)
        data = topic_table[channel.topic](message_data)

        pbar.update(1)
        yield {"log_time": log_time, channel.topic: data}
    pbar.close()


def load_parquet(mcap_file: str) -> pl.DataFrame:
    hash = sha256sum(mcap_file)
    parquet_location = Path.cwd().joinpath("parquets", f"{hash}.parquet")

    if parquet_location.exists():
        return pl.read_parquet(parquet_location)

    with open(mcap_file, "rb") as file:
        reader = make_reader(file)
        orientation = pl.DataFrame(
            iterate_messages(
                reader, [ReadTopic("Control.main_outputs.filtered_orientation")]
            )
        )
        fall_states = pl.DataFrame(iterate_messages(reader, [ReadFallState()]))
        sensors = pl.DataFrame(iterate_messages(reader, [ReadSensorValues()]))

    dataframe = sensors.join(fall_states, on="log_time").join(
        orientation, on="log_time"
    )
    dataframe.write_parquet(parquet_location)

    return dataframe


@click.command()
@click.argument(
    "mcap-file", type=click.Path(exists=True, dir_okay=False, readable=True)
)
def main(mcap_file):
    df = load_parquet(mcap_file)
    lin_accel = (
        df.select(
            pl.col("Control.main_outputs.sensor_data")
            .struct.field("linear_acceleration")
            .list.to_struct()
            .alias("la")
        )
        .unnest("la")
        .to_numpy()
    )
    orientation = lin_accel @ np.array([1.0, 0.0, 0.0])
    fig = go.Figure()
    fig.add_scatter(y=orientation)
    fig.add_scatter(y=df["Control.main_outputs.fall_state"].rank("dense"))
    fig.show()


if __name__ == "__main__":
    main()
