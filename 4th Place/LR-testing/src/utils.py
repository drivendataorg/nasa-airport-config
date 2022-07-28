from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

hyperParams = {
    ("katl", 30): [0.25, 10, 1],
    ("katl", 60): [0.25, 10, 1],
    ("katl", 90): [0.25, 8, 0.8],
    ("katl", 120): [0.25, 7, 0.7],
    ("katl", 150): [0.25, 4, 0.4],
    ("katl", 180): [0.25, 7, 0.3],
    ("katl", 210): [0.25, 4, 0.4],
    ("katl", 240): [0.25, 5, 0.2],
    ("katl", 270): [0.25, 6, 0.3],
    ("katl", 300): [0.25, 5, 0.2],
    ("katl", 330): [0.25, 6, 0.1],
    ("katl", 360): [0.25, 4, 0.3],
    ("kclt", 30): [0.25, 10, 1],
    ("kclt", 60): [0.25, 6, 0.8],
    ("kclt", 90): [0.5, 7, 0.8],
    ("kclt", 120): [0.25, 5, 0.9],
    ("kclt", 150): [0.25, 4, 0.5],
    ("kclt", 180): [0.25, 4, 0.3],
    ("kclt", 210): [0.25, 4, 0.4],
    ("kclt", 240): [0.25, 7, 0.1],
    ("kclt", 270): [0.25, 5, 0.2],
    ("kclt", 300): [0.25, 6, 0.1],
    ("kclt", 330): [0.25, 4, 0.1],
    ("kclt", 360): [0.25, 6, 0.1],
    ("kden", 30): [0.5, 10, 1],
    ("kden", 60): [0.25, 5, 0.3],
    ("kden", 90): [0.25, 6, 0.3],
    ("kden", 120): [0.25, 7, 0.2],
    ("kden", 150): [0.25, 9, 0.1],
    ("kden", 180): [0.5, 6, 0.1],
    ("kden", 210): [0.25, 4, 0.1],
    ("kden", 240): [0.25, 4, 0.1],
    ("kden", 270): [1, 5, 0.2],
    ("kden", 300): [0.5, 4, 0.1],
    ("kden", 330): [0.75, 4, 0.1],
    ("kden", 360): [0.5, 4, 0.1],
    ("kdfw", 30): [0.25, 10, 1],
    ("kdfw", 60): [0.75, 9, 1],
    ("kdfw", 90): [0.25, 4, 1],
    ("kdfw", 120): [0.75, 7, 0.6],
    ("kdfw", 150): [0.75, 4, 0.8],
    ("kdfw", 180): [0.25, 4, 0.4],
    ("kdfw", 210): [0.5, 4, 0.6],
    ("kdfw", 240): [0.5, 6, 0.3],
    ("kdfw", 270): [0.25, 4, 0.2],
    ("kdfw", 300): [0.25, 7, 0.1],
    ("kdfw", 330): [0.5, 6, 0.1],
    ("kdfw", 360): [0.25, 5, 0.1],
    ("kjfk", 30): [0.25, 10, 1],
    ("kjfk", 60): [0.25, 9, 0.9],
    ("kjfk", 90): [0.25, 8, 0.9],
    ("kjfk", 120): [0.25, 7, 0.7],
    ("kjfk", 150): [0.25, 6, 0.9],
    ("kjfk", 180): [0.25, 4, 1],
    ("kjfk", 210): [0.25, 9, 0.3],
    ("kjfk", 240): [0.25, 7, 0.3],
    ("kjfk", 270): [0.25, 6, 0.3],
    ("kjfk", 300): [0.5, 7, 0.3],
    ("kjfk", 330): [0.25, 4, 0.4],
    ("kjfk", 360): [0.5, 7, 0.2],
    ("kmem", 30): [0.5, 10, 1],
    ("kmem", 60): [0.25, 4, 0.5],
    ("kmem", 90): [0.25, 4, 0.5],
    ("kmem", 120): [0.25, 9, 0.1],
    ("kmem", 150): [0.25, 7, 0.1],
    ("kmem", 180): [0.25, 7, 0.1],
    ("kmem", 210): [0.25, 5, 0.1],
    ("kmem", 240): [0.5, 4, 0.1],
    ("kmem", 270): [0.25, 4, 0.1],
    ("kmem", 300): [0.75, 4, 0.1],
    ("kmem", 330): [0.25, 4, 0.1],
    ("kmem", 360): [0.5, 4, 0.1],
    ("kmia", 30): [0.25, 10, 1],
    ("kmia", 60): [0.25, 7, 0.7],
    ("kmia", 90): [0.25, 4, 0.8],
    ("kmia", 120): [0.25, 9, 0.3],
    ("kmia", 150): [0.25, 5, 0.5],
    ("kmia", 180): [0.25, 4, 0.3],
    ("kmia", 210): [0.5, 4, 0.6],
    ("kmia", 240): [0.25, 5, 0.2],
    ("kmia", 270): [0.25, 4, 0.2],
    ("kmia", 300): [0.25, 7, 0.1],
    ("kmia", 330): [0.25, 5, 0.2],
    ("kmia", 360): [0.25, 4, 0.1],
    ("kord", 30): [0.25, 5, 1],
    ("kord", 60): [0.25, 5, 0.3],
    ("kord", 90): [0.25, 4, 0.4],
    ("kord", 120): [0.25, 4, 0.2],
    ("kord", 150): [0.25, 6, 0.1],
    ("kord", 180): [0.25, 4, 0.1],
    ("kord", 210): [0.25, 4, 0.1],
    ("kord", 240): [0.25, 4, 0.1],
    ("kord", 270): [0.25, 4, 0.1],
    ("kord", 300): [0.25, 4, 0.1],
    ("kord", 330): [0.25, 4, 0.1],
    ("kord", 360): [0.5, 4, 0.1],
    ("kphx", 30): [0.25, 10, 1],
    ("kphx", 60): [0.25, 7, 0.7],
    ("kphx", 90): [0.25, 6, 1],
    ("kphx", 120): [0.25, 4, 0.8],
    ("kphx", 150): [0.25, 4, 0.4],
    ("kphx", 180): [0.25, 4, 0.3],
    ("kphx", 210): [0.25, 5, 0.2],
    ("kphx", 240): [0.25, 9, 0.1],
    ("kphx", 270): [0.25, 7, 0.1],
    ("kphx", 300): [0.25, 6, 0.1],
    ("kphx", 330): [0.25, 4, 0.1],
    ("kphx", 360): [0.25, 5, 0.1],
    ("ksea", 30): [0.25, 10, 1],
    ("ksea", 60): [0.25, 10, 1],
    ("ksea", 90): [0.25, 8, 0.8],
    ("ksea", 120): [0.25, 4, 1],
    ("ksea", 150): [0.25, 4, 0.8],
    ("ksea", 180): [0.25, 9, 0.3],
    ("ksea", 210): [0.25, 9, 0.3],
    ("ksea", 240): [0.25, 5, 0.5],
    ("ksea", 270): [0.25, 4, 0.5],
    ("ksea", 300): [0.25, 4, 0.5],
    ("ksea", 330): [0.25, 4, 0.4],
    ("ksea", 360): [0.25, 7, 0.2],
}


def read_airport_configs(airport_directory: Path) -> Tuple[str, pd.DataFrame]:
    """Reads the airport configuration features for a given airport data directory."""
    airport_code = airport_directory.name
    filename = f"{airport_code}_airport_config.csv"
    filepath = airport_directory / filename
    airport_config_df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return airport_code, airport_config_df


def make_prediction(
    airport_config_df_map: Dict[str, pd.DataFrame],
    pred_frame: pd.DataFrame,
    hedge: float = 1,
    weight: float = 8,
    discount_factor: float = 0.89,
    accuracy: float = 1,
) -> pd.Series:
    # start with a uniform distribution
    uniform = make_uniform(pred_frame) * hedge
    predictive_distribution = pd.DataFrame({"uniform": uniform})

    # select the data we're allowed to use
    first = pred_frame.iloc[0]
    airport_code, timestamp, lookahead, _, _ = first
    airport_config_df = airport_config_df_map[airport_code]
    # if there is no data, return the uniform probability
    if len(airport_config_df) == 0:
        return uniform / uniform.sum()
    current, subset = censor_data(airport_config_df, timestamp)
    if len(subset) == 0:
        return uniform / uniform.sum()

    # make the distribution of past configurations
    config_dist = make_config_dist(airport_code, subset, normalize=True)
    predictive_distribution["config_dist"] = config_dist.reindex(
        predictive_distribution.index
    ).fillna(0)
    other = config_dist.sum() - predictive_distribution.config_dist.sum()
    predictive_distribution.loc[f"{airport_code}:other", "config_dist"] += other

    # put some extra weight on the current configuration (or `other`)
    current_key = f"{airport_code}:{current}"
    if current_key not in pred_frame.config.values:
        current_key = f"{airport_code}:other"
    # discount = pow(discount_factor, lookahead/30)
    discount = accuracy
    predictive_distribution["current"] = 0  # initalize a column of zeros
    predictive_distribution.loc[current_key, "current"] = weight * discount

    # combine the components and normalize the result
    mixture = predictive_distribution.sum(axis=1)

    predictive_distribution["mixture"] = mixture / mixture.sum()

    return predictive_distribution.mixture


def make_all_predictions(
    airport_config_df_map: Dict[str, pd.DataFrame], predictions: pd.DataFrame
):
    global hyperParams
    """Predicts airport configuration for all of the prediction frames in a table."""
    all_preds = []
    grouped = predictions.groupby(["airport", "timestamp", "lookahead"], sort=False)
    for key, pred_frame in tqdm(grouped):
        airport, timestamp, lookahead = key
        airport, timestamp, lookahead = key
        hyp = hyperParams[(airport, lookahead)]
        pred_dist = make_prediction(
            airport_config_df_map,
            pred_frame,
            hedge=hyp[0],
            weight=hyp[1],
            accuracy=hyp[2],
        )
        assert np.array_equal(pred_dist.index.values, pred_frame["config"].values)
        all_preds.append(pred_dist.values)

    predictions["active"] = np.concatenate(all_preds)


def make_all_predictions_test(
    airport_config_df_map: Dict[str, pd.DataFrame],
    predictions: pd.DataFrame,
    accuracy: float,
    hedge: float,
    weight: float,
):
    """Predicts airport configuration for all of the prediction frames in a table."""
    all_preds = []
    grouped = predictions.groupby(["airport", "timestamp", "lookahead"], sort=False)
    for key, pred_frame in tqdm(grouped):
        airport, timestamp, lookahead = key
        pred_dist = make_prediction(
            airport_config_df_map,
            pred_frame,
            accuracy=accuracy,
            hedge=hedge,
            weight=weight,
        )
        assert np.array_equal(pred_dist.index.values, pred_frame["config"].values)
        all_preds.append(pred_dist.values)

    predictions["active"] = np.concatenate(all_preds)


def make_uniform(pred_frame: pd.DataFrame) -> pd.Series:
    indices = pred_frame["config"].values
    uniform = pd.Series(1, index=indices)
    uniform /= uniform.sum()
    return uniform


def make_config_dist(
    airport_code: str, airport_config_df: pd.DataFrame, normalize: bool = False
) -> pd.Series:
    config_timecourse = (
        airport_config_df.set_index("timestamp")
        .airport_config.resample("15min")
        .ffill()
        .dropna()
    )
    config_dist = config_timecourse.value_counts()
    if normalize:
        config_dist /= config_dist.sum()

    # prepend the airport code to the configuration strings
    prefix = pd.Series(f"{airport_code}:", index=config_dist.index)
    config_dist.index = prefix.str.cat(config_dist.index)
    return config_dist


def censor_data(
    airport_config_df: pd.DataFrame, timestamp: pd.Timestamp
) -> Tuple[str, pd.DataFrame]:
    mask = airport_config_df["timestamp"] <= timestamp
    subset = airport_config_df[mask]
    if subset.shape[0] > 0:
        current = subset.iloc[-1].airport_config
    else:
        current = airport_config_df.iloc[-1].airport_config
    return current, subset
