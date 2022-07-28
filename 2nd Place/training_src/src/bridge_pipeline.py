import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import json

from src.const import *
from src.submodel_preparators import PredictionDataPreparator

SUBMODELS = {}
SUBMODEL_DESCS = {}
for airport in AIRPORTS:
    airport_submodels = {}
    airport_model_desc = {}
    k = 1
    Flag = True
    while Flag:
        try:
            model_id = f"submodel_{k}"

            submodel = XGBClassifier()

            submodel.load_model(f"{SUBMODEL_DIR}/{airport}/{model_id}")

            with open(f"{SUBMODEL_DIR}/{airport}/{model_id}_description.json") as d:
                preparator_description = json.load(d)

            airport_submodels[k] = submodel
            airport_model_desc[k] = preparator_description

            k += 1
        except:
            Flag = False

    SUBMODELS[airport] = airport_submodels
    SUBMODEL_DESCS[airport] = airport_model_desc


def return_model_ouput(airport, k, preparator, dataset: str = "all"):

    submodel = SUBMODELS[airport][k]

    if dataset == "train":
        df = preparator.train_features_df
    elif dataset == "val":
        df = preparator.val_features_df
    elif dataset == "test":
        df = preparator.test_features_df
    elif dataset == "all":
        df = preparator.complete_convolved_feature_df
    else:
        print("GIVE ME A VALID DATASET OPTION")

    predictions = submodel.predict_proba(df)

    return predictions


def topk_scaled_prob(proba, k):
    scaled_proba = np.zeros(proba.shape)

    N = proba.shape[0]

    for i in range(N):
        ind = (-proba[i, :]).argsort()[:k]
        scaled_proba[i, ind] = proba[i, ind] / sum(proba[i, ind])

    return scaled_proba


def predict_submodels(df, airport, dataset: str = "all"):

    submodel_preds = []

    num_submodels = len(SUBMODELS[airport].keys())

    preparator_description = SUBMODEL_DESCS[airport][1]

    preparator = PredictionDataPreparator(
        complete_df=df,
        airport=airport,
        rate_lookback=preparator_description["rate_lookback"],
        proj_rate_lookback=preparator_description["proj_rate_lookback"],
        temperature_lookback=preparator_description["temperature_lookback"],
        visibility_lookback=preparator_description["visibility_lookback"],
        wind_lookback=preparator_description["wind_lookback"],
        precip_lookback=preparator_description["precip_lookback"],
        rway_lookback=preparator_description["rway_lookback"],
        changes_lookback=preparator_description["changes_lookback"],
        config_lookback=preparator_description["config_lookback"],
    )

    for k in range(1, num_submodels + 1):

        submodel_pred = return_model_ouput(airport, k, preparator, dataset=dataset)

        N, P = submodel_pred.shape

        if P > 2:
            submodel_pred = topk_scaled_prob(submodel_pred, 3)
            for p in range(P):
                submodel_preds.append(submodel_pred[:, p])
        else:
            submodel_preds.append(submodel_pred[:, 0])

    submodel_preds = np.stack(submodel_preds, axis=1)

    if dataset == "train":
        config = preparator.config_train
        wind = preparator.wind_train
        time = preparator.time_train
    elif dataset == "val":
        config = preparator.config_val
        wind = preparator.wind_val
        time = preparator.time_val
    elif dataset == "test":
        config = preparator.config_test
        wind = preparator.wind_test
        time = preparator.time_test
    elif dataset == "all":
        config = preparator.config_feature_df.to_numpy()
        wind = preparator.wind_feature_df.to_numpy()
        time = preparator.time_feature_df.to_numpy()

    else:
        print("GIVE ME A VALID DATASET OPTION")

    X_pred = np.hstack((submodel_preds, wind, time, config))

    X_pred = X_pred.astype("float32")

    return X_pred


def run_bridge_pipeline():

    for airport in AIRPORTS:

        df = pd.read_csv(
            f"{DATA_DIR}/{airport}/{airport}_processed_data.csv", index_col=0
        )

        train_submodel_preds = predict_submodels(df, airport, "train")
        val_submodel_preds = predict_submodels(df, airport, "val")
        test_df_submodel_preds = predict_submodels(df, airport, "test")

        np.savetxt(
            f"{DATA_DIR}/{airport}/{airport}_train_data.csv",
            train_submodel_preds,
            delimiter=",",
        )
        np.savetxt(
            f"{DATA_DIR}/{airport}/{airport}_val_data.csv",
            val_submodel_preds,
            delimiter=",",
        )
        np.savetxt(
            f"{DATA_DIR}/{airport}/{airport}_test_data.csv",
            test_df_submodel_preds,
            delimiter=",",
        )

        print("\n" + 100 * "-" + "\n")
        print("Finished Airport: " + airport)
        print("\n" + 100 * "-" + "\n")


if __name__ == "__main__":

    run_bridge_pipeline()
