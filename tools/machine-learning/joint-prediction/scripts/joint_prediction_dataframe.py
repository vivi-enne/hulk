from datetime import datetime
from mcap.reader import make_reader
import msgpack
import polars as pl

# from tqdm import tqdm


def main():
    print("Hello from mcap-use-test!")

    with open(
        "2024-07-18-HULKs-vs-NomadZ/first-half/10.1.24.32/2024-07-18_10:48:50/outputs.mcap",
        "rb",
    ) as file:
        # with open("2024-07-18-HULKs-vs-NomadZ/first-half/10.1.24.36/2024-07-18_10:48:52/outputs.mcap", "rb") as file:
        reader = make_reader(file)

        df = pl.DataFrame()
        j = 0

        for i, channel in enumerate(reader.get_summary().channels.values()):
            topic = channel.topic
            # TODO: check role

            # if topic == "Control.main_outputs.motion_command" or topic == "Control.main_outputs.sensor_data" or topic == "Control.main_outputs.motor_commands" or topic == "Control.main_outputs.role":
            if (
                topic == "Control.main_outputs.role"
            ):  # or topic == "Control.main_outputs.sensor_data" or topic == "Control.main_outputs.motor_commands":
                print(f"{topic = }")

                number_messages = (
                    reader.get_summary().statistics.channel_message_counts[channel.id]
                )

                # df = pl.from_dicts(iterate_messages(reader, [topic]), infer_schema_length=number_messages, strict=False)

                if j == 0:
                    df = pl.from_dicts(
                        iterate_messages(reader, [topic]),
                        infer_schema_length=number_messages,
                        strict=False,
                    )
                    print(f"{df.columns = }")
                else:
                    row = pl.from_dicts(
                        iterate_messages(reader, [topic]),
                        infer_schema_length=number_messages,
                        strict=False,
                    )
                    print(f"{row.columns = }")

                    # # Align columns: Ensure both DataFrames have the same columns
                    # all_columns = set(df.columns).union(row.columns)
                    # aligned_df = df.with_columns([pl.lit(None).alias(col) for col in all_columns if col not in df.columns])
                    # aligned_row = row.with_columns([pl.lit(None).alias(col) for col in all_columns if col not in row.columns])

                    # # Reorder columns to match
                    # aligned_df = aligned_df.select(sorted(aligned_df.columns))
                    # aligned_new_rows_df = aligned_row.select(sorted(aligned_row.columns))

                    # print(f"{aligned_new_rows_df = }")

                    # Ensure both DataFrames have the same number of rows
                    if len(df) != len(row):
                        raise ValueError(
                            "Both DataFrames must have the same number of rows to align."
                        )

                    # Align columns: Ensure both DataFrames have the same columns
                    all_columns = set(df.columns).union(row.columns)

                    # Add missing columns to both DataFrames
                    df = df.with_columns(
                        [
                            pl.lit(
                                None,
                                dtype=row.schema[col]
                                if col in row.schema
                                else pl.Object,
                            ).alias(col)
                            for col in all_columns
                            if col not in df.columns
                        ]
                    )
                    row = row.with_columns(
                        [
                            pl.lit(
                                None,
                                dtype=df.schema[col] if col in df.schema else pl.Object,
                            ).alias(col)
                            for col in all_columns
                            if col not in row.columns
                        ]
                    )

                    # Merge the row data into the original DataFrame on a row-by-row basis
                    df = df.with_columns(
                        [
                            pl.when(row[col].is_null())
                            .then(df[col])
                            .otherwise(row[col])
                            .alias(col)
                            for col in row.columns
                        ]
                    )

                j += 1

            ## main_outputs.motion_command
            ## main_outputs.sensor_data
            ## main_outputs.motor_commands

        df.write_parquet("dataframe_sensor_data_36.parquet")

        print("end")


def iterate_messages(reader, topics: list[str]):
    # schema = None
    # row = None
    # flag = "None"
    for i, (_, channel, message) in enumerate(reader.iter_messages(topics=topics)):
        message_data = msgpack.unpackb(message.data)
        log_time = datetime.fromtimestamp(message.log_time * 1e-9)

        if type(message_data) is dict:
            # print(f"{message_data = }")
            topic = channel.topic.split(".")[-1]
            # new_data = {"log_time": log_time, **message_data}
            new_data = {"log_time": log_time}
            new_data.update(
                {f"{topic}.{key}": value for key, value in message_data.items()}
            )

            yield new_data
        else:
            topic = channel.topic.split(".")[-1]
            # print(f"{message_data = }, {topic = }")
            yield {"log_time": log_time, topic: message_data}

        ## TODO: handle enums


if __name__ == "__main__":
    main()
