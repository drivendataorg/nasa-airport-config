from pathlib import Path
import pandas as pd
from loguru import logger
import multiprocessing
from multiprocessing import freeze_support
import pickle

# Assigning data directory
root_directory = Path(__file__).parents[1]
train_path = root_directory / "training_data"
test_path = root_directory / "data"
processed_directory = root_directory / "processed_tables"
processed_directory.mkdir(exist_ok=True, parents=True)

# Train_directory
airport_directories_train = sorted(path for path in train_path.glob("*"))[:10]
# Test directory
airport_directories_test = sorted(path for path in test_path.glob("*"))[:10]


def read_airport_configs(airport_directory):
    """
    This function reads the airport configuration features for a given airport
    data directory. We used the benchmark code (with possible slight changes).
    """
    airport_code = airport_directory.name
    filename = f"{airport_code}_airport_config.csv.bz2"
    filepath = airport_directory / filename
    airport_config_df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return airport_code, airport_config_df


# Make a dictionary that maps each airport code to its configuration
# For training (for more information,  refer to competition benchmark)
airport_config_df_map_train = {}
for airport_directory_train in sorted(airport_directories_train):
    airport_code_train, airport_config_df_train = read_airport_configs(
        airport_directory_train
    )
    airport_config_df_map_train[airport_code_train] = airport_config_df_train
# For testing
airport_config_df_map_test = {}
for airport_directory in sorted(airport_directories_test):
    airport_code, airport_config_df = read_airport_configs(airport_directory)
    airport_config_df_map_test[airport_code] = airport_config_df

# Read train labels
if not (processed_directory / "train_labels").exists():
    open_train_labels = pd.read_csv(
        train_path / "prescreened_train_labels.csv.bz2",
        parse_dates=["timestamp"],
    )


def make_uniform(pred_frame: pd.DataFrame):
    """
    We used the benchmark code to write this function (with possible slight
    changes).
    """
    indices = pred_frame["config"].values
    uniform = pd.Series(1, index=indices)
    uniform /= uniform.sum()
    return uniform


def make_config_dist(
    airport_code: str, airport_config_df: pd.DataFrame, normalize: bool = False
) -> pd.Series:
    """
    We used the benchmark code to write this function (with possible slight
    changes). This function develops distribution of past configurations of the
    airports.
    """
    config_timecourse = (
        airport_config_df.set_index("timestamp")
        .airport_config.resample("15min")
        .ffill()
        .dropna()
    )

    if len(config_timecourse) > 2880:
        config_dist = config_timecourse[-2880:].value_counts()
    else:
        config_dist = config_timecourse.value_counts()
    if normalize:
        config_dist /= config_dist.sum()

    # Prepend the airport code to the configuration strings
    prefix = pd.Series(f"{airport_code}:", index=config_dist.index)
    config_dist.index = prefix.str.cat(config_dist.index)

    return config_dist


def censor_data(airport_config_df: pd.DataFrame, timestamp: pd.Timestamp):
    """
    We used the benchmark code to write this function (with possible slight
    changes). This function makes sure that we'll use features up to the time
    (current time) that we want to predict the future configurations for.
    """
    mask = airport_config_df["timestamp"] <= timestamp
    subset = airport_config_df[mask]
    try:
        current = subset.iloc[-1].airport_config
    except:
        current = 0

    return current, subset


def make_predictors(pred_frame, test=False):
    """
    This function creates a table containing all predictors that we used
    for predicting future airport configurations on a certain timestamp.

    args:
        pred_frame: The timestamp that we want to predict its future
        configurations.

        test: Specifies whether we want the output data for training or
        testing.

    return:
        Table with the predictors and label. Predictors are airport
        configuration distribution, current and 10 last configurations
        (ccon to ccon-10), 10 last configurations durations (d1 to d10),
        lookahead, hour, day, and week of current configuration.
    """

    first = pred_frame.iloc[0]
    airport_code, timestamp, lookahead, _, _ = first
    month, weekday, hour = timestamp.month, timestamp.weekday(), timestamp.hour
    if test is False:
        label = list(
            pred_frame[pred_frame["active"] == 1]["config"].str[5:].astype(str)
        )
    # Select the data we are allowed to use based on benchmark censor_data
    # function
    if test:
        airport_config_df = airport_config_df_map_test[airport_code]
        current, subset = censor_data(airport_config_df, timestamp)

    else:
        airport_config_df = airport_config_df_map_train[airport_code]
        current, subset = censor_data(airport_config_df, timestamp)

    current_new = [current]
    timestamp_new = [timestamp]
    duration = []
    subset_new = subset

    # Map configurations to numbers
    config_keys = pred_frame["config"].str[5:].to_list()
    number_values = list(range(1, (len(config_keys) + 1)))
    map_dict = dict(zip(config_keys, number_values))

    # Loop for extracting 10 previous configurations and durations
    for ii in range(10):
        try:
            index_false = max(
                subset_new.index[subset_new["airport_config"] != current_new[ii]]
            )
            timestamp_new.append(subset.timestamp[index_false])
            current_temp, subset_new = censor_data(
                airport_config_df, timestamp_new[ii + 1]
            )
            current_new.append(current_temp)
            duration.append((timestamp_new[ii] - timestamp_new[ii + 1]).seconds / 60)
        except:
            current_new.append(0)
            duration.append(0)

    columns_names = [
        "ccon",
        "ccon-1",
        "ccon-2",
        "ccon-3",
        "ccon-4",
        "ccon-5",
        "ccon-6",
        "ccon-7",
        "ccon-8",
        "ccon-9",
        "ccon-10",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "d8",
        "d9",
        "d10",
        "lookahead",
    ]
    row_values = [current_new + duration + [lookahead]]
    past_config = pd.DataFrame(row_values, columns=columns_names)

    for ii in range(11):
        past_config.iloc[:, ii] = past_config.iloc[:, ii].map(map_dict).fillna(0)

    # Extract the distributions of past configurations
    config_dist = make_config_dist(airport_code, subset, normalize=True)
    predictive_distribution = pd.DataFrame({"uniform": make_uniform(pred_frame)})
    predictive_distribution["config_dist"] = config_dist.reindex(
        predictive_distribution.index
    ).fillna(0)
    other = config_dist.sum() - predictive_distribution.config_dist.sum()
    predictive_distribution.loc[f"{airport_code}:other", "config_dist"] += other
    predictors = predictive_distribution.T.iloc[[1], :]
    predictors = predictors.rename({"config_dist": 0})
    final_table = pd.concat([predictors, past_config], axis=1)

    if test is False:
        final_table[["hour", "weekdayday", "month", "label"]] = [
            hour,
            weekday,
            month,
            label[0],
        ]
    else:
        final_table[["hour", "weekdayday", "month"]] = [hour, weekday, month]

    if predictive_distribution.config_dist.sum() == 0:
        final_table.iloc[0, int(final_table["ccon"] - 1)] = 1
    return final_table


# Sorting labels
if not (processed_directory / "train_labels").exists():
    training_labels = (
        open_train_labels.set_index(["airport", "timestamp", "lookahead"])
        .reset_index()
        .sort_values(["airport", "timestamp", "lookahead", "config"])
    )

    predictions = training_labels.copy()


def make_all_predictors(grouped_airport, key_airport, ii=None, test=False):
    """
    Create a DataFrame containing predictors and airport configurations
    (labels). This function process multiple data points using
    “make_predictors” function.

    args:
        grouped_airport: Airports configuration table grouped by airports’
        names.
        key_airport: Name of the airport
        ii: We divided the table of training labels to 5 parts. ii is the part
        number.
        test: Specifies whether we are using this function for preprocessing
        training or testing data.

    return:
        Preprocessed table of predictors if we processed testing predictors.
        For training predictors, we used multiple CPUs and we saved the results
        as csv instead of returning them. We joined csv files at the end of
        the preprocessing code.
    """
    grouped = grouped_airport.groupby(["airport", "timestamp", "lookahead"], sort=False)

    for idx, (key, pred_frame) in enumerate(grouped):
        pred_dist = make_predictors(pred_frame, test)

        if idx == 0:
            all_preds = pred_dist
        else:
            all_preds = pd.concat([all_preds, pred_dist])

    path_2 = processed_directory / key_airport
    path_2.mkdir(exist_ok=True, parents=True)

    if test:
        return all_preds
    else:
        all_preds.to_csv(path_2 / f"{key_airport}_{ii}.csv", index=False)


# Processing training data and preparing them for training
# We used multiple CPUs to speed up the preprocessing step
numberofcores = 5

if __name__ == "__main__":
    freeze_support()
    if not (processed_directory / "train_labels").exists():
        grouped_airports = predictions.groupby(["airport"], sort=False)
        processes = []

        for key_airport, grouped_airport in grouped_airports:
            grouped_all = grouped_airport.groupby(
                ["timestamp", "lookahead"], sort=False
            )

            for key, group in grouped_all:
                label_len = len(group)

            len_data = len(grouped_all)

            for ii in range(numberofcores):
                st = ii * int(len_data / 5) * label_len
                end = (ii + 1) * int(len_data / 5) * label_len
                if ii == (numberofcores - 1):
                    end = len_data * label_len
                p = multiprocessing.Process(
                    target=make_all_predictors,
                    args=[grouped_airport.iloc[st:end, :], key_airport, ii],
                )
                p.start()
                processes.append(p)
            for process in processes:
                process.join()
            logger.info("{}--------done", key_airport)

    # Join features and labels extracted from training data and produce single
    # file for training ("train_labels" file)
    if not (processed_directory / "train_labels").exists():
        airport_labels_folder = sorted(path for path in processed_directory.glob("*"))
        train_labels_airports = []

        for ii in range(len(airport_labels_folder)):

            for idx, file in enumerate(
                sorted(path for path in airport_labels_folder[ii].glob("*.csv"))
            ):
                cn = pd.read_csv(file)

                if idx == 0:
                    output = cn
                else:
                    output = pd.concat([output, cn])
            train_labels_airports.append(output)
        with (processed_directory / "train_labels").open("wb") as fp:
            pickle.dump(train_labels_airports, fp)
