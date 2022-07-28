import pandas as pd
import numpy as np
import datetime
import os

from src.const import *
from src.const import CONFIG_FILE
from src.feature_processing_functions import *
from src.helpers import get_file_path


def num_skip(path, n_keep=1000):

    num_lines = sum(1 for line in open(path))

    if num_lines - n_keep < 0:
        skip = 0
    else:
        skip = num_lines - n_keep

    skip_list = [i for i in range(1, skip)]

    return skip_list


def process_airport(airport, date_range):

    processed_time_df = process_time(date_range)
    df = processed_time_df.copy(deep=True)

    dep_path = get_file_path(airport, DEP_FILE)
    df_dep = pd.read_csv(dep_path, skiprows=num_skip(dep_path, 500))
    processed_dep_rate_df = process_true_rate(date_range, df_dep, op_type="dep")

    arr_path = get_file_path(airport, ARR_FILE)
    df_arr = pd.read_csv(arr_path, skiprows=num_skip(arr_path, 500))
    processed_arr_rate_df = process_true_rate(date_range, df_arr, op_type="arr")

    weather_path = get_file_path(airport, WEATHER_FILE)
    df_weather = pd.read_csv(weather_path, skiprows=num_skip(weather_path, 200))
    processed_weather_df = process_weather(date_range, df_weather)

    config_path = get_file_path(airport, CONFIG_FILE)
    df_config = pd.read_csv(config_path, skiprows=num_skip(config_path, 200))
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

    df = relabel_configs(df, airport)

    return df
