from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from src.const import *

# JUST BECAUSE GIT NOT UPDATING


@dataclass
class DataPreparator(ABC):
    complete_df: pd.DataFrame
    airport: str
    rate_lookback: int = 0
    proj_rate_lookback: int = -1
    temperature_lookback: int = 0
    visibility_lookback: int = 0
    wind_lookback: int = 0
    precip_lookback: int = 0
    rway_lookback: int = 4
    changes_lookback: int = 4
    config_lookback: int = 4
    time_lookback: int = 0
    change_features: str = "change_in_this_period"
    time_features: Tuple[str] = ("Day_Sin", "Week_Cos")
    rate_features: Tuple[str] = ("rate_0_30_dep", "rate_25_30_arr")
    projected_rate_features: Tuple[str] = (
        "next_30_min_estimated_runway_departure",
        "next_300_min_scheduled_runway_arrival",
    )
    temperature_features: Tuple[str] = ("('temperature', 0.5)", "('temperature', 24.5)")
    visibility_features: Tuple[str] = ("('cloud_ceiling', 0.5)", "('cloud', 24.5)")
    wind_features: Tuple[str] = ("('east_wind', 0.5)", "('north_wind', 24.5)")
    precip_features: Tuple[str] = ("('precip', 0.5)", "('precip', 24.5)")
    config_features: Tuple[str] = ("configs_begin", "configs_end")
    dep_rway_features: Tuple[str] = ("dep_rways_begin", "dep_rways_end")
    arr_rway_features: Tuple[str] = ("arr_rways_begin", "arr_rways_end")
    used_rway_features: Tuple[str] = ("used_rways_begin", "used_rways_end")
    seed: int = 1

    def __post_init__(self):
        self.complete_columns = list(self.complete_df.columns)
        self.timestamps = self.complete_df.index
        self.n_rows = len(self.timestamps)

        self.config_features: Tuple[str] = (
            self.complete_columns[
                self.complete_columns.index(self.config_features[0]) + 1
            ],
            self.complete_columns[
                self.complete_columns.index(self.config_features[1]) - 1
            ],
        )
        self.dep_rway_features: Tuple[str] = (
            self.complete_columns[
                self.complete_columns.index(self.dep_rway_features[0]) + 1
            ],
            self.complete_columns[
                self.complete_columns.index(self.dep_rway_features[1]) - 1
            ],
        )
        self.arr_rway_features: Tuple[str] = (
            self.complete_columns[
                self.complete_columns.index(self.arr_rway_features[0]) + 1
            ],
            self.complete_columns[
                self.complete_columns.index(self.arr_rway_features[1]) - 1
            ],
        )
        self.used_rway_features: Tuple[str] = (
            self.complete_columns[
                self.complete_columns.index(self.used_rway_features[0]) + 1
            ],
            self.complete_columns[
                self.complete_columns.index(self.used_rway_features[1]) - 1
            ],
        )

        self.time_feature_df = self.complete_df.loc[
            :, self.time_features[0] : self.time_features[1]
        ]

        self.rate_feature_df = self.complete_df.loc[
            :, self.rate_features[0] : self.rate_features[1]
        ]

        # self.projected_rate_feature_df   = self.complete_df.loc[:,self.projected_rate_features[0]: self.projected_rate_features[1]]

        self.temperature_feature_df = self.complete_df.loc[
            :, self.temperature_features[0] : self.temperature_features[1]
        ]

        self.visibility_feature_df = self.complete_df.loc[
            :, self.visibility_features[0] : self.visibility_features[1]
        ]

        self.wind_feature_df = self.complete_df.loc[
            :, self.wind_features[0] : self.wind_features[1]
        ]

        self.precip_feature_df = self.complete_df.loc[
            :, self.precip_features[0] : self.precip_features[1]
        ]

        self.change_feature_df = self.complete_df.loc[
            :, self.change_features
        ].to_frame()

        self.config_feature_df = self.complete_df.loc[
            :, self.config_features[0] : self.config_features[1]
        ]

        self.dep_rway_feature_df = self.complete_df.loc[
            :, self.dep_rway_features[0] : self.dep_rway_features[1]
        ]
        self.arr_rway_feature_df = self.complete_df.loc[
            :, self.arr_rway_features[0] : self.arr_rway_features[1]
        ]
        self.used_rway_feature_df = self.complete_df.loc[
            :, self.used_rway_features[0] : self.used_rway_features[1]
        ]
        self.rway_feature_df = pd.concat(
            [
                self.dep_rway_feature_df,
                self.arr_rway_feature_df,
                self.used_rway_feature_df,
            ],
            axis=1,
        )

        self.feature_convolution()

        if self.temperature_lookback == -1:
            self.complete_convolved_feature_df = self.complete_convolved_feature_df.drop(
                columns=self.complete_columns[
                    self.complete_columns.index(
                        self.temperature_features[0]
                    ) : self.complete_columns.index(self.temperature_features[1])
                    + 1
                ]
            )

        self.train_mask = self.complete_df["train_set"]
        self.val_mask = self.complete_df["val_set"]
        self.test_mask = self.complete_df["test_set"]
        self.configuration_series = self.complete_df["airport_config"]
        self.train_features_df = self.complete_convolved_feature_df[self.train_mask]
        self.val_features_df = self.complete_convolved_feature_df[self.val_mask]
        self.test_features_df = self.complete_convolved_feature_df[self.test_mask]
        self.train_current_config = self.config_feature_df[self.train_mask]
        self.val_current_config = self.config_feature_df[self.val_mask]
        self.test_current_config = self.config_feature_df[self.test_mask]

        self.label_series = self.label_generator()
        self.train_timestamps = self.timestamps[self.train_mask]
        self.val_timestamps = self.timestamps[self.val_mask]
        self.test_timestamps = self.timestamps[self.test_mask]
        self.train_labels = self.label_series[self.train_mask]
        self.val_labels = self.label_series[self.val_mask]
        self.test_labels = self.label_series[self.test_mask]

        self.X_train = self.train_features_df.to_numpy()
        self.X_val = self.val_features_df.to_numpy()
        self.X_test = self.test_features_df.to_numpy()

        self.config_train = self.train_current_config.to_numpy()
        self.config_val = self.val_current_config.to_numpy()
        self.config_test = self.test_current_config.to_numpy()

        self.wind_train = self.wind_feature_df[self.train_mask].to_numpy()
        self.wind_val = self.wind_feature_df[self.val_mask].to_numpy()
        self.wind_test = self.wind_feature_df[self.test_mask].to_numpy()

        self.time_train = self.time_feature_df[self.train_mask].to_numpy()
        self.time_val = self.time_feature_df[self.val_mask].to_numpy()
        self.time_test = self.time_feature_df[self.test_mask].to_numpy()

        self.y_train = self.train_labels.to_numpy()
        self.y_val = self.val_labels.to_numpy()
        self.y_test = self.test_labels.to_numpy()

    def convolve_feature_df(
        self, feature_df: pd.DataFrame, lookback: int
    ) -> pd.DataFrame:
        # extend a single feature dataframe based on lookback
        feature_df_new = feature_df.copy(deep=True)
        if lookback > 0:

            for i in range(1, 1 + lookback):

                shifted_df = feature_df.shift(i)

                feature_df_new = pd.concat(
                    [feature_df_new, shifted_df.add_suffix(f"_{i}_lookback")], axis=1
                )

        feature_df_new = feature_df_new.fillna(0)
        return feature_df_new

    def feature_convolution(self):
        # extend each feature backward based on lookback, i.e. add a bunch of new columns to complete_df

        self.wind_feature_df_convolved = self.convolve_feature_df(
            self.wind_feature_df, self.wind_lookback
        )
        self.precip_feature_df_convolved = self.convolve_feature_df(
            self.precip_feature_df, self.precip_lookback
        )
        self.rate_feature_df_convolved = self.convolve_feature_df(
            self.rate_feature_df, self.rate_lookback
        )
        # self.projected_rate_feature_df_convolved = self.convolve_feature_df(self.projected_rate_feature_df,self.proj_rate_lookback)
        self.temperature_feature_df_convolved = self.convolve_feature_df(
            self.temperature_feature_df, self.temperature_lookback
        )
        self.visibility_feature_df_convolved = self.convolve_feature_df(
            self.visibility_feature_df, self.visibility_lookback
        )
        self.change_feature_df_convolved = self.convolve_feature_df(
            self.change_feature_df, self.changes_lookback
        )
        self.config_feature_df_convolved = self.convolve_feature_df(
            self.config_feature_df, self.config_lookback
        )
        self.rway_feature_df_convolved = self.convolve_feature_df(
            self.rway_feature_df, self.rway_lookback
        )

        self.complete_convolved_feature_df = pd.concat(
            [
                self.wind_feature_df_convolved,
                self.precip_feature_df_convolved,
                self.rate_feature_df_convolved,
                self.temperature_feature_df_convolved,
                self.visibility_feature_df_convolved,
                self.change_feature_df_convolved,
                self.config_feature_df_convolved,
                self.rway_feature_df_convolved,
            ],
            axis=1,
        )  # self.projected_rate_feature_df_convolved

    def describe_parameters(self):
        result = {}
        result["preparator_name"] = self.__class__.__name__
        result["airport"] = self.airport
        result["rate_lookback"] = self.rate_lookback
        result["proj_rate_lookback"] = self.proj_rate_lookback
        result["temperature_lookback"] = self.temperature_lookback
        result["visibility_lookback"] = self.visibility_lookback
        result["wind_lookback"] = self.wind_lookback
        result["precip_lookback"] = self.precip_lookback
        result["rway_lookback"] = self.rway_lookback
        result["changes_lookback"] = self.changes_lookback
        result["config_lookback"] = self.config_lookback

        return result

    @abstractmethod
    def label_generator(self):
        raise NotImplementedError("Fill me in")


@dataclass
class PredictionDataPreparator(DataPreparator):
    def __post_init__(self):
        self.complete_columns = list(self.complete_df.columns)
        self.timestamps = self.complete_df.index
        self.config_features: Tuple[str] = (
            self.complete_columns[
                self.complete_columns.index(self.config_features[0]) + 1
            ],
            self.complete_columns[
                self.complete_columns.index(self.config_features[1]) - 1
            ],
        )
        self.dep_rway_features: Tuple[str] = (
            self.complete_columns[
                self.complete_columns.index(self.dep_rway_features[0]) + 1
            ],
            self.complete_columns[
                self.complete_columns.index(self.dep_rway_features[1]) - 1
            ],
        )
        self.arr_rway_features: Tuple[str] = (
            self.complete_columns[
                self.complete_columns.index(self.arr_rway_features[0]) + 1
            ],
            self.complete_columns[
                self.complete_columns.index(self.arr_rway_features[1]) - 1
            ],
        )
        self.used_rway_features: Tuple[str] = (
            self.complete_columns[
                self.complete_columns.index(self.used_rway_features[0]) + 1
            ],
            self.complete_columns[
                self.complete_columns.index(self.used_rway_features[1]) - 1
            ],
        )

        self.time_feature_df = self.complete_df.loc[
            :, self.time_features[0] : self.time_features[1]
        ]
        self.rate_feature_df = self.complete_df.loc[
            :, self.rate_features[0] : self.rate_features[1]
        ]
        # self.projected_rate_feature_df   = self.complete_df.loc[:,self.projected_rate_features[0]: self.projected_rate_features[1]]
        self.temperature_feature_df = self.complete_df.loc[
            :, self.temperature_features[0] : self.temperature_features[1]
        ]
        self.visibility_feature_df = self.complete_df.loc[
            :, self.visibility_features[0] : self.visibility_features[1]
        ]
        self.wind_feature_df = self.complete_df.loc[
            :, self.wind_features[0] : self.wind_features[1]
        ]
        self.precip_feature_df = self.complete_df.loc[
            :, self.precip_features[0] : self.precip_features[1]
        ]
        self.change_feature_df = self.complete_df.loc[
            :, self.change_features
        ].to_frame()
        self.config_feature_df = self.complete_df.loc[
            :, self.config_features[0] : self.config_features[1]
        ]

        self.dep_rway_feature_df = self.complete_df.loc[
            :, self.dep_rway_features[0] : self.dep_rway_features[1]
        ]
        self.arr_rway_feature_df = self.complete_df.loc[
            :, self.arr_rway_features[0] : self.arr_rway_features[1]
        ]
        self.used_rway_feature_df = self.complete_df.loc[
            :, self.used_rway_features[0] : self.used_rway_features[1]
        ]
        self.rway_feature_df = pd.concat(
            [
                self.dep_rway_feature_df,
                self.arr_rway_feature_df,
                self.used_rway_feature_df,
            ],
            axis=1,
        )

        self.feature_convolution()

        if self.temperature_lookback == -1:
            self.complete_convolved_feature_df = self.complete_convolved_feature_df.drop(
                columns=self.complete_columns[
                    self.complete_columns.index(
                        self.temperature_features[0]
                    ) : self.complete_columns.index(self.temperature_features[1])
                    + 1
                ]
            )

        self.train_mask = self.complete_df["train_set"]
        self.val_mask = self.complete_df["val_set"]
        self.test_mask = self.complete_df["test_set"]
        self.train_features_df = self.complete_convolved_feature_df[self.train_mask]
        self.val_features_df = self.complete_convolved_feature_df[self.val_mask]
        self.test_features_df = self.complete_convolved_feature_df[self.test_mask]

        self.config_train = self.config_feature_df[self.train_mask].to_numpy()
        self.config_val = self.config_feature_df[self.val_mask].to_numpy()
        self.config_test = self.config_feature_df[self.test_mask].to_numpy()

        self.wind_train = self.wind_feature_df[self.train_mask].to_numpy()
        self.wind_val = self.wind_feature_df[self.val_mask].to_numpy()
        self.wind_test = self.wind_feature_df[self.test_mask].to_numpy()

        self.time_train = self.time_feature_df[self.train_mask].to_numpy()
        self.time_val = self.time_feature_df[self.val_mask].to_numpy()
        self.time_test = self.time_feature_df[self.test_mask].to_numpy()

        # HERE CREATE X ROW THAT WE NEED, THIS WILL BE A SLICE FROM self.complete_convolved_feature_df, probably the last row?

    def label_generator(self):
        # THIS IS NOT NEEDED IN PREDICTION PREPARATOR
        return


@dataclass
class BinaryConfigDataPreparator(DataPreparator):
    lookahead: int = 1
    configuration: str = ""

    def label_generator(self):

        labels = (
            self.config_feature_df[self.configuration]
            .astype(int)
            .shift(-self.lookahead)
            .fillna(0)
        )  # (configs == self.configuration).astype(int)

        return labels

    def describe_parameters(self):
        result = super().describe_parameters()
        result["lookahead"] = self.lookahead
        result["configuration"] = self.configuration
        return result


@dataclass
class BinaryRwayDataPreparator(DataPreparator):
    lookahead: int = 1
    rway: str = ""
    operation: str = "used"

    def label_generator(self):
        self.reference = f"{self.rway}_{self.operation}"
        labels = (
            self.rway_feature_df[self.reference]
            .astype(int)
            .shift(-self.lookahead)
            .fillna(0)
        )
        return labels

    def describe_parameters(self):
        result = super().describe_parameters()
        result["lookahead"] = self.lookahead
        result["reference"] = self.reference
        return result


@dataclass
class SingleConfigDataPreparator(DataPreparator):
    lookahead: int = 1

    def label_generator(self):
        # for each sample in complete_df get a corresponding label - the configuration for the corresponding lookahead - i.e. new column
        labels = self.configuration_series.shift(-self.lookahead).fillna("other")
        return labels

    def describe_parameters(self):
        result = super().describe_parameters()
        result["lookahead"] = self.lookahead
        return result


@dataclass
class AggregateConfigDataPreparator(DataPreparator):
    # should use 1-3,4-6,7-9,10-12 ..... 1-6,7-12 or 1-12
    lookahead_low_idx: int = 1
    lookahead_high_idx: int = 3
    decide_based_on: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.decide_based_on == 0:
            self.decide_based_on = int(
                (self.lookahead_high_idx - self.lookahead_low_idx) / 2
            )

    def label_generator(self):

        labels_df = pd.DataFrame()
        for i in range(self.lookahead_low_idx, self.lookahead_high_idx + 1):
            config_shifted = self.configuration_series.shift(-i).fillna("other")
            labels_df[f"shifted_{i}"] = config_shifted

        labels_modes = labels_df.mode(axis=1)

        labels = labels_modes[0]

        multiple_modes_mask = labels_modes[1].notna()

        labels = labels.mask(
            multiple_modes_mask, other=labels_df.iloc[:, self.decide_based_on]
        )

        return labels

    def describe_parameters(self):
        result = super().describe_parameters()
        result["lookahead_low_idx"] = self.lookahead_low_idx
        result["lookahead_high_idx"] = self.lookahead_high_idx
        result["decide_based_on"] = self.decide_based_on
        return result


@dataclass
class ChangeDataPreparator(DataPreparator):
    lookahead: int = 1

    def label_generator(self):

        labels = (
            (self.configuration_series != self.configuration_series.shift(1))
            .astype(int)
            .shift(-self.lookahead)
            .fillna(0)
        )
        return labels

    def describe_parameters(self):
        result = super().describe_parameters()
        result["lookahead"] = self.lookahead
        return result


@dataclass
class AggregateChangeDataPreparator(DataPreparator):
    lookahead_low_idx: int = 1
    lookahead_high_idx: int = 4

    def label_generator(self):

        labels = (
            (
                self.configuration_series.shift(self.lookahead_low_idx)
                != self.configuration_series.shift(self.lookahead_high_idx)
            )
            .astype(int)
            .fillna(0)
        )
        return labels

    def describe_parameters(self):
        result = super().describe_parameters()
        result["lookahead_low_idx"] = self.lookahead_low_idx
        result["lookahead_high_idx"] = self.lookahead_high_idx
        return result
