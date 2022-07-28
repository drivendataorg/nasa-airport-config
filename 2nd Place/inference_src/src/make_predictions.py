import numpy as np
import pandas as pd
import json
import time

from src.const import *
from src.data_processing import process_airport
from src.submodel_predictions import predict_submodels


# Function that takes prediction df and match submission format exactly
def match_submission_format(df, SUBMISSION_FORMAT):

    sub_format = pd.read_csv(SUBMISSION_FORMAT)

    df.set_index(['airport', 'timestamp', 'lookahead', 'config'], inplace=True)
    sub_format.set_index(['airport', 'timestamp', 'lookahead', 'config'], inplace=True)

    pred_active = df.loc[sub_format.index]["active"].to_numpy()

    final_sub = pd.read_csv(SUBMISSION_FORMAT)
    final_sub.active = pred_active

    return final_sub


# Function that makes predictions from submission format and date range
def make_predictions(SUBMISSION_FORMAT, date_range):

    # Prediction df is initizilized
    pred_df = pd.DataFrame(columns=["airport", "timestamp", "lookahead", "config", "active"])

    dp_time = 0
    sub_time = 0
    final_time = 0
    for airport in AIRPORTS:

        # Raw Data Loaded and Processed for each Airport
        dp_start = time.time()
        df = process_airport(airport,date_range)
        dp_end= time.time()

        dp_time = (dp_start-dp_end)/60 + dp_time

        pred_time_stamps = np.array([pd.to_datetime(df.index[-1]).strftime(DATETIME_FORMAT)])

        # Submodel predictions are made from processed data
        sub_start = time.time()
        X_pred = predict_submodels(df, airport)
        sub_end = time.time()

        sub_time = (sub_end-sub_start)/60 + sub_time

        print("\n"+100*"-"+"\n")
        print("Finished Submodel Predictions for Airport: " + airport)
        print("\n"+100*"-"+"\n")

        # Labels and values used to update prediction df
        labels = CONFIGS[airport]
        inv_labels = {v: airport+":"+k for k, v in labels.items()}

        N = len(pred_time_stamps)
        C = len(labels.keys())
        
        label_array = []
        for i in range(C):
            label_array.append(inv_labels[i])
        label_array = np.array(label_array)

        label_array = label_array.reshape(-1,1)
        pred_time_stamps = pred_time_stamps.reshape(-1,1)

        label_matrix = np.transpose(np.repeat(label_array, N ,axis=1))
        time_matrix = np.repeat(pred_time_stamps, C ,axis=1)

        for i in range(1,13):
            
            final_start = time.time()
            
            model = MODELS[airport][i]
            
            # Final Model Predictions are made for each airport/lookahead
            pred = model.predict(X_pred)

            config_array = label_matrix.flatten()
            time_array = time_matrix.flatten()
            airport_array = np.full(time_array.shape, airport)
            lookahead_array = np.full(time_array.shape, int(i*30))
            pred_array = pred.flatten()

            # Prediction df is updated with new predictions
            pred_df_new = pd.DataFrame({"airport":airport_array, "timestamp":time_array, "lookahead":lookahead_array, "config":config_array, "active":pred_array})
            pred_df = pd.concat([pred_df, pred_df_new])

            final_end = time.time()

            final_time = (final_end-final_start)/60 + final_time


    # Raw prediction df is made to match the submission format exactly
    pred_df = match_submission_format(pred_df,SUBMISSION_FORMAT)

    return pred_df, dp_time, sub_time, final_time




if __name__ == "__main__":

    make_predictions()