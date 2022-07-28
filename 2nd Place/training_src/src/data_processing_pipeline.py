import pandas as pd
import datetime

from src.const import AIRPORTS, ARR_FILE, CONFIG_FILE, DATA_DIR, DEP_FILE, WEATHER_FILE
from src.feature_processing_functions import (
    ensure_datetime,
    process_config_changes,
    process_runways,
    process_time,
    process_true_rate,
    process_weather,
    relabel_configs,
)
from src.helpers import get_file_path


def create_train_val_test_masks(
    complete_df, train_timestamps, val_timestamps, test_timestamps
):
    complete_df["train_set"] = complete_df.index.isin(train_timestamps)
    complete_df["val_set"] = complete_df.index.isin(val_timestamps)
    complete_df["test_set"] = complete_df.index.isin(test_timestamps)
    return complete_df


def process_airport(
    airport, date_range, train_timestamps, val_timestamps, test_timestamps
):

    processed_time_df = process_time(date_range)
    df = processed_time_df.copy(deep=True)

    df_dep = pd.read_csv(get_file_path(airport, DEP_FILE))
    processed_dep_rate_df = process_true_rate(date_range, df_dep, op_type="dep")

    df_arr = pd.read_csv(get_file_path(airport, ARR_FILE))
    processed_arr_rate_df = process_true_rate(date_range, df_arr, op_type="arr")

    df_weather = pd.read_csv(get_file_path(airport, WEATHER_FILE))
    processed_weather_df = process_weather(date_range, df_weather)

    df_config = pd.read_csv(get_file_path(airport, CONFIG_FILE))
    processed_changes_df = process_config_changes(
        date_range, df_config, airport=airport
    )
    processed_rways_df = process_runways(date_range, df_config, airport=airport)

    df = pd.concat(
        [
            processed_time_df,
            processed_dep_rate_df,
            processed_arr_rate_df,
            processed_weather_df,
            processed_changes_df,
            processed_rways_df,
        ],
        axis=1,
    )

    df = create_train_val_test_masks(
        df, train_timestamps, val_timestamps, test_timestamps
    )
    df = relabel_configs(df, airport)

    df.to_csv(f"{DATA_DIR}/{airport}/{airport}_processed_data.csv")


def process_data(train_timestamps, val_timestamps, test_timestamps):

    start_date = datetime.datetime(2020, 11, 1, 0, 0, 0)
    end_date = datetime.datetime(2021, 11, 1, 0, 0, 0)
    date_range = (pd.date_range(start=start_date, end=end_date, freq="30T")).to_frame(
        name="timestamp"
    )

    for airport in AIRPORTS:

        process_airport(
            airport, date_range, train_timestamps, val_timestamps, test_timestamps
        )

        print("finished ", airport)


if __name__ == "__main__":

    train_labels_df = ensure_datetime(
        pd.read_csv(f"{DATA_DIR}/open_train_labels.csv.bz2")
    )
    train_timestamps = sorted(list(train_labels_df["timestamp"].unique()))

    open_sub_df = ensure_datetime(pd.read_csv(f"{DATA_DIR}/open_submission_format.csv"))
    addval_start_date = datetime.datetime(2021, 10, 18, 10, 0, 0)
    addval_end_date = datetime.datetime(2021, 10, 31, 16, 0, 0)
    addval_daterange = (
        pd.date_range(start=addval_start_date, end=addval_end_date, freq="60T")
    ).to_frame(name="timestamp")
    addval_timestamps = sorted(list(addval_daterange["timestamp"].unique()))
    val_timestamps = sorted(list(open_sub_df["timestamp"].unique())) + addval_timestamps

    test_start_date = datetime.datetime(2020, 11, 1, 4, 0, 0)
    test_end_date = datetime.datetime(2020, 11, 6, 22, 0, 0)
    test_daterange = (
        pd.date_range(start=test_start_date, end=test_end_date, freq="60T")
    ).to_frame(name="timestamp")
    test_timestamps = sorted(list(test_daterange["timestamp"].unique()))

    process_data(train_timestamps, val_timestamps, test_timestamps)
