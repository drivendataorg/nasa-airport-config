
import pandas as pd
from typing import Sequence
from functools import wraps
import time
import numpy as np
import datetime

from src.const import *

def split_gufi(df:pd.DataFrame, gufi_column: str= 'gufi') -> pd.DataFrame:
    df[GUFI_PARTS] = df[gufi_column].str.split(pat = '.',expand = True)
    return df

def drop_gufi_cols(df: pd.DataFrame):
    df = df.drop(columns = GUFI_PARTS)

def ensure_datetime(df: pd.DataFrame, columns: Sequence[str] = DATETIME_COLS)-> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            if df[column].dtype == str:
                df[column] = df[column].str.replace("T"," ")
            df[column] = pd.to_datetime(df[column])
    return df

def drop_datetime_cols(df: pd.DataFrame)->pd.DataFrame:
    return df

def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +datetime.timedelta(hours=t.minute//30))

def feature_processing_function(func):
    @wraps(func)
    def function_wrapper(date_range,*raw_feature_dfs, **kwargs):   #ONLY PROVIDE A DATERANGE WITH REGULAR STEPS
        name = func.__name__

        print("--------------------------------------")
        # print(f"--> Formatting inputs for {name}")


        processed_feature_dfs = []
        for raw_feature_df in raw_feature_dfs:

            # columns = list(raw_feature_df.columns)

            raw_feature_df = ensure_datetime(raw_feature_df)

            # if 'gufi' in columns:
            #     raw_feature_df = split_gufi(raw_feature_df)
            
            processed_feature_dfs.append(raw_feature_df)
        
        processed_feature_dfs_args = tuple(processed_feature_dfs)
        
        print(f"--> Running {name}")
        start = time.time()
        processed_feature_df = func(date_range,*processed_feature_dfs_args, **kwargs)
        end = time.time()
        # print(f"--> Finished {name} in {round(end-start)}s")
        # print(f"--> Formatting outputs")
        
        processed_feature_df = drop_datetime_cols(processed_feature_df)
        #ENFORCE len(date_range) == len(df) and that date_range == df.index
        
        # print(f"--> Finished")
        print("--------------------------------------")
        return processed_feature_df


    return function_wrapper

@feature_processing_function
def process_true_rate(date_range, raw_operation_df: pd.DataFrame, maximum_lookback = 30,op_type = 'dep') -> pd.DataFrame:

    raw_operation_df['minute'] = raw_operation_df['timestamp'].dt.floor('T')
    val_counts = raw_operation_df['minute'].value_counts().to_frame(name = 'counts')
    all_minutes = pd.date_range(start=date_range['timestamp'].iloc[0], end=date_range['timestamp'].iloc[-1], freq='1T').to_frame(name = 'timestamp')
    minute_rates  = all_minutes.merge(val_counts,how='left', left_index = True, right_index = True)
    minute_rates['counts'].fillna(0, inplace=True)
    frequency = int((date_range['timestamp'].iloc[1] - date_range['timestamp'].iloc[0]).total_seconds()/60)
    counts_list = list(minute_rates['counts'])
    buckets = [[0,30],[0,15],[15,30],[0,10],[10,20],[20,30],[0,5],[5,10],[10,15],[15,20],[20,25],[25,30]]
    bucket_names = [f"rate_{bucket[0]}_{bucket[1]}_{op_type}" for bucket in buckets]
    rate_features = np.zeros((len(date_range),len(buckets)))

    for i in range(len(date_range)):
        
        if i*frequency >= maximum_lookback:
            counts_slice = counts_list[i*frequency-maximum_lookback:i*frequency]
            output_row = [sum(counts_slice[bucket[0]:bucket[1]]) for bucket in buckets]

        else:
            output_row = np.zeros(len(buckets))
        rate_features[i] = np.array(output_row) 
    
    rate_df = pd.DataFrame(rate_features, columns=bucket_names)
    rate_df.set_index(date_range['timestamp'],inplace=True)
    return rate_df

@feature_processing_function
def process_projected_rate(date_range, raw_rate_df, rate_type = 'scheduled_runway_arrival', horizon = 10):
    
    start_date = date_range['timestamp'].iloc[0]
    end_date   = date_range['timestamp'].iloc[-1]
    mask = (raw_rate_df['timestamp']>= start_date) & (raw_rate_df['timestamp']<=end_date)
    raw_rate_df = raw_rate_df[mask]
    raw_rate_df.drop_duplicates(inplace = True)
    raw_rate_df['t_delta'] = ((raw_rate_df[f"{rate_type}_time"] - raw_rate_df['timestamp']).astype('timedelta64[s]')/60)
    raw_rate_df = raw_rate_df[raw_rate_df['t_delta']>0].copy(deep = True)
    raw_rate_df['minutes_of_hour'] = raw_rate_df['timestamp'].dt.minute
    raw_rate_df['mins_since_idx'] = raw_rate_df['minutes_of_hour'] - (raw_rate_df['minutes_of_hour']>30)*30
    raw_rate_df['buckets_into_future'] = ((raw_rate_df['t_delta']+raw_rate_df['mins_since_idx'])/30).astype(int)
    raw_rate_df['since_start'] =((raw_rate_df['timestamp'] - start_date).astype('timedelta64[s]')/(60*30)).astype(int)
    raw_rate_df = raw_rate_df[raw_rate_df['buckets_into_future'] >0]
    raw_rate_df = raw_rate_df[raw_rate_df['buckets_into_future'] <horizon]
    raw_rate_df = raw_rate_df.drop_duplicates(subset = ['buckets_into_future','since_start', 'gufi'],keep='last').reset_index(drop = True)

    date_range_list = list(date_range['timestamp'])
    rate_matrix = np.zeros((len(date_range_list)+1,horizon))
    buckets_list = list(raw_rate_df.buckets_into_future)
    since_start_list = list(raw_rate_df.since_start)

    for i,j in zip(since_start_list,buckets_list):
        rate_matrix[i+1,j-1] += 1

    df_rate_matrix = pd.DataFrame(rate_matrix[:-1,:], columns = [f"next_{i*30 + 30}_min_{rate_type}" for i in range(horizon)])
    df_rate_matrix.set_index(date_range["timestamp"],inplace = True)
    return df_rate_matrix

@feature_processing_function
def process_time(date_range):

    df = pd.DataFrame()
    df['hour'] = date_range['timestamp'].dt.hour
    df['day_of_the_week'] = date_range['timestamp'].dt.dayofweek

    day = 24
    week = 7

    df['Day_Sin'] = np.sin( (df['hour']+1)  * (2 * np.pi / day))
    df['Day_Cos'] = np.cos( (df['hour']+1)  * (2 * np.pi / day))
    df['Week_Sin'] = np.sin( (df['day_of_the_week']+1) * (2 * np.pi / week))
    df['Week_Cos'] = np.cos( (df['day_of_the_week']+1) * (2 * np.pi / week))

    return df

@feature_processing_function
def process_config_changes(date_range,raw_config_df,airport = ''):
    if airport == '':
        print("PROVIDE AIRPORT")

    config_df = raw_config_df.set_index('timestamp')
    config_df = config_df.reindex(date_range['timestamp'],method = 'ffill')

    config_df = relabel_configs(config_df,airport)
    config_df['previous_airport_config'] = config_df['airport_config'].shift(1).fillna("UNK")
    config_df['airport_config'] = config_df['airport_config'].fillna("UNK")
    config_df['change_in_this_period'] = (config_df['airport_config'] != config_df['previous_airport_config']).astype(int)

    config_df['configs_begin'] = ""
    configs = list(CONFIGS[airport].keys())
    for c in configs:

        config_df[c] = (config_df['airport_config'] == c).astype(int)

    config_df['configs_end'] = ""
    config_df = config_df.drop(columns = ['previous_airport_config'])

    return config_df

@feature_processing_function
def process_weather(date_range,raw_weather_df):
    # Fill in mising elements
    weather_df = raw_weather_df.fillna(0)

    # Fill in string weather information
    weather_df["precip"] = weather_df["precip"].astype(int)

    Cloud_Num = [("CL", 0), ("FW", 0.1), ("SC", 0.4), ("BK", 0.8), ("OV", 1)]
    for (c, c_n) in Cloud_Num:
        weather_df["cloud"] = weather_df["cloud"].replace(c, c_n)
        
    Lightning_Num = [("N", 0), ("L", 0.2), ("M", 0.5), ("H", 1)]
    for (c, c_n) in Lightning_Num:
        weather_df["lightning_prob"] = weather_df["lightning_prob"].replace(c, c_n)

    # Break windspeed and direction into components
    weather_df["east_wind"] = weather_df.wind_speed*np.cos(np.deg2rad(weather_df.wind_direction))
    weather_df["north_wind"] = weather_df.wind_speed*np.sin(np.deg2rad(weather_df.wind_direction))
    weather_df.drop(columns=["wind_direction", "wind_speed"])

    # Create a column to measure the relative prediction timedelta
    weather_df["prediction_delta"] = round(pd.to_timedelta(weather_df["forecast_timestamp"] - weather_df["timestamp"]).dt.total_seconds()/3600,1)

    # Sort the pandas dataframe
    weather_df = weather_df.sort_values(["timestamp", "prediction_delta"], ascending = (True, True))

    # Group the data and unstack
    weather_df = weather_df.groupby(["timestamp", "prediction_delta"]).mean()
    weather_df = weather_df[weather_df.columns].unstack()


    weather_df = weather_df.reindex(date_range['timestamp'],method = 'ffill')
    weather_df.fillna(0, inplace = True)

    return weather_df

@feature_processing_function
def process_taxi():
    return None

@feature_processing_function
def process_runways(date_range,raw_config_df,airport = ''):
    if airport == "":
        print("PROVIDE AIRPORT")
    config_df = raw_config_df.set_index('timestamp')
    config_df = config_df.reindex(date_range['timestamp'],method = 'ffill')

    config_df[['departures_str','arrivals_str']] = config_df['airport_config'].str.split('_A_', expand = True)
    config_df['departures_str'] = config_df['departures_str'].str.strip(to_strip = 'D_')

    departure_rways = DEP_RWAYS[airport]
    arrival_rways = ARR_RWAYS[airport]
    all_rways = RWAYS[airport]


    config_df['departures_str'] = "_" + config_df['departures_str'].fillna("")
    config_df['arrivals_str'] = "_" + config_df['arrivals_str'].fillna("")

    
    config_df['dep_rways_begin'] = ""
    for dep_rway in departure_rways:
        config_df[f"{dep_rway}_departure"] = config_df['departures_str'].str.contains(f"_{dep_rway}").astype(int)

    config_df['dep_rways_end'] = ""
    config_df['arr_rways_begin'] = ""

    for arr_rway in arrival_rways:
        config_df[f"{arr_rway}_arrival"] = config_df['arrivals_str'].str.contains(f"_{arr_rway}").astype(int)
    
    config_df['arr_rways_end'] = ""
    config_df['used_rways_begin'] = ""
    for rway in all_rways:
        col_name = f"{rway}_used"
        if rway in departure_rways and rway in arrival_rways:
            config_df[col_name] = config_df[f"{rway}_arrival"] | config_df[f"{rway}_departure"]
        elif rway not in departure_rways:
            config_df[col_name] = config_df[f"{rway}_arrival"]
        elif rway not in arrival_rways:
            config_df[col_name] = config_df[f"{rway}_departure"]
    config_df['used_rways_end'] = ""

    return config_df.drop(columns = ['airport_config', 'departures_str','arrivals_str'])
    
def create_train_test_masks(complete_df, airport,train_labels, open_submission):
    train_labels_airport = train_labels[train_labels.airport == airport].copy(deep=True)
    open_submission_airport = open_submission[open_submission.airport == airport].copy(deep=True)
    
    train_timestamps = set(train_labels_airport.timestamp)
    test_timestamps = set(open_submission_airport.timestamp)

    complete_df['train_set'] = complete_df.index.isin(train_timestamps)
    complete_df['test_set']  = complete_df.index.isin(test_timestamps)
    return complete_df

def relabel_configs(df,airport):

    configs = list(CONFIGS[airport].keys())
    mask = (~df['airport_config'].isin(configs))
    df[mask] = 'other'

    return df