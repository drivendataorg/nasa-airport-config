from pathlib import Path
import time
import numpy as np
import pandas as pd
from loguru import logger
import typer
import pickle
from xgboost import XGBClassifier
import subprocess
from preprocess import make_all_predictors

# Assigning data directory
root_directory = Path(__file__).parents[1]
processed_directory = root_directory / "processed_tables"
model_directory = root_directory / "models"
model_directory.mkdir(exist_ok=True, parents=True)
prediction_directory = root_directory / "prediction"
prediction_directory.mkdir(exist_ok=True, parents=True)
data_path = root_directory / "data"

# Read preprocessed table (it runs preprocess script in case it did not find
# train_labels file in the "processed_tables" folder)
if (processed_directory / "train_labels").exists():
    logger.info("Loading existing train labels")
    with (processed_directory / "train_labels").open("rb") as fp:
        train_labels_airports = pickle.load(fp)
else:
    subprocess.call(["python", str(root_directory / "src" / "preprocess.py")])
    with (processed_directory / "train_labels").open("rb") as fp:
        train_labels_airports = pickle.load(fp)


# Training XGBoost models for each airport using train_labels DataFrame.
# If there are pretrained models in “model” folder, the code only loads the
# pretrained models.
if (model_directory / "xgb_models.pkl").exists():
    with (model_directory / "xgb_models.pkl").open("rb") as xgb_models:
        model_list = pickle.load(xgb_models)

else:
    model_list = []
    for idx, predictors in enumerate(train_labels_airports):
        start = time.time()

        all_predictors = predictors.reset_index(drop=True)
        X = all_predictors.iloc[:, :-1].values
        y = all_predictors.iloc[:, -1].values
        model = XGBClassifier(
            max_depth=5,
            learning_rate=0.02,
            objective="multi:softprob",
            eval_metric=["error", "mlogloss"],
            n_estimators=300,
            min_child_weight=1,
            reg_alpha=0.01,
            gamma=0,
            subsample=0.7,
            colsample_bytree=0.7,
            tree_method="hist",
            use_label_encoder=True,
        )
        model.fit(X, y)
        model_list.append(model)
        print(f"Airport {idx} completed")
        logger.info("Airport {} completed", idx)
    with (model_directory / "xgb_models.pkl").open("wb") as xgb_models:
        pickle.dump(model_list, xgb_models)


def main(prediction_time):
    """
    We preprocess trained data using make_all_predictors function and then we
    use the trained XGBoost models for predicting configuration probability of
    each timestamp.

    args:
        prediction_time: The prediction time provided in the datetime format.
    return:
        Save the prediction table as csv.
    """
    logger.info("Computing my predictions for {}", prediction_time)
    open_test_labels = pd.read_csv(
        data_path / "partial_submission_format.csv", parse_dates=["timestamp"]
    )

    grouped_airports_test = open_test_labels.groupby(["airport"], sort=False)
    test_features = []
    for key_airport, grouped_airport in grouped_airports_test:
        test_features.append(
            make_all_predictors(grouped_airport, key_airport, ii=None, test=True)
        )
    # Predicting testing data configurations
    predicted_test_prob = []
    for idx, test_feature in enumerate(test_features):
        test_data = test_feature.reset_index(drop=True)
        test_probs = model_list[idx].predict_proba(test_data)

        # Fix missing labels issue
        all_labels_with_airport = list(test_features[idx].columns[:-25])
        all_labels = [x[5:] for x in all_labels_with_airport]
        model_labels = list(model_list[idx].classes_)
        difference = list(sorted(set(all_labels) - set(model_labels)))

        if difference:
            for item in difference:
                item_idx = all_labels.index(item)
                print(f"missing label at {item_idx}")
                test_probs = np.insert(test_probs, item_idx, 0, axis=1)

        predicted_test_prob.append(test_probs.ravel())

    logger.info("saving predictions {}", prediction_time)
    # Saving the predictions
    predicted_test_prob_final = predicted_test_prob
    all_probs_raveled = np.concatenate(predicted_test_prob_final, axis=0)
    final_submission = pd.read_csv(data_path / "partial_submission_format.csv")
    final_submission["active"] = all_probs_raveled
    final_submission.to_csv(prediction_directory / "prediction.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
