from datetime import datetime
from mcap.reader import make_reader
import msgpack
import polars as pl
import click


@click.command()
@click.argument(
    "mcap-file", type=click.Path(exists=True, dir_okay=False, readable=True)
)
def main(mcap_file: str):
    with open(mcap_file, "rb") as file:
        reader = make_reader(file)
        df = pl.DataFrame()

        for i, channel in enumerate(reader.get_summary().channels.values()):
            topic = channel.topic

            # topic = "Control.main_outputs.condition_input"
            # topic = "Control.main_outputs.ground_to_field"
            # topic = "Control.main_outputs.sensor_data"
            # topic = "Control.additional_outputs.whistle_in_set_ball_position"
            topics = [
                "Control.main_outputs.motion_command",
                "Control.main_outputs.sensor_data",
                "Control.main_outputs.motor_command",
            ]

            if topic in topics:
                number_messages = (
                    reader.get_summary().statistics.channel_message_counts[channel.id]
                )
                df = pl.from_dicts(
                    iterate_messages(reader, [topic]),
                    infer_schema_length=number_messages,
                    strict=False,
                )

                if i == 0:
                    df = pl.from_dicts([message_data])
                else:
                    row = pl.from_dicts([message_data])

                    # Align columns: Ensure both DataFrames have the same columns
                    all_columns = set(df.columns).union(row.columns)
                    aligned_df = df.with_columns(
                        [
                            pl.lit(None).alias(col)
                            for col in all_columns
                            if col not in df.columns
                        ]
                    )
                    aligned_row = row.with_columns(
                        [
                            pl.lit(None).alias(col)
                            for col in all_columns
                            if col not in row.columns
                        ]
                    )

                    # Reorder columns to match
                    aligned_df = aligned_df.select(sorted(aligned_df.columns))
                    aligned_new_rows_df = aligned_row.select(
                        sorted(aligned_row.columns)
                    )

                    print(f"{aligned_new_rows_df = }")
                    # Extend the original DataFrame with the new rows
                    df = aligned_df.vstack(aligned_new_rows_df)

            number_messages = reader.get_summary().statistics.channel_message_counts[i]

            print(f"{topic = }, {i = }, {number_messages = }")

            df = pl.from_dicts(
                iterate_messages(reader, [topic]),
                infer_schema_length=number_messages,
                strict=False,
            )

            print(df)

            df.write_parquet("dataframe_sensor_data.parquet")

            ## main_outputs.motion_command
            ## main_outputs.sensor_data
            ## main_outputs.motor_commands

        print("end")


def iterate_messages(reader, topics: list[str]):
    # schema = None
    # row = None
    # flag = "None"
    for i, (_, channel, message) in enumerate(reader.iter_messages(topics=topics)):
        message_data = msgpack.unpackb(message.data)
        log_time = datetime.fromtimestamp(message.log_time * 1e-9)

        if type(message_data) is dict:
            print(f"{message_data = }")
            yield {"log_time": log_time, **message_data}
        else:
            topic = channel.topic.split(".")[-1]
            print(f"{message_data = }, {topic = }")
            yield {"log_time": log_time, topic: message_data}

        ## TODO: handle enums


if __name__ == "__main__":
    main()
