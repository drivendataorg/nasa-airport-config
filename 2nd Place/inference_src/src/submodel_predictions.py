import pandas as pd
import numpy as np

from src.const import *
from src.submodel_preparators import PredictionDataPreparator

def return_model_ouput(df, airport, k, preparator):

    submodel = SUBMODELS[airport][k]

    predictions = submodel.predict_proba(preparator.complete_convolved_feature_df)

    return predictions


def topk_scaled_prob(proba, k):
    scaled_proba = np.zeros(proba.shape)

    N = proba.shape[0]

    for i in range(N):
        ind = (-proba[i,:]).argsort()[:k]
        scaled_proba[i,ind] = proba[i,ind]/sum(proba[i,ind])

    return scaled_proba


def predict_submodels(df, airport):

    submodel_preds = []

    num_submodels = len(SUBMODELS[airport].keys())

    preparator_description = SUBMODEL_DESCS[airport][1]
   
    preparator = PredictionDataPreparator(complete_df=df,
                                            airport=airport,
                                            rate_lookback=preparator_description['rate_lookback'],
                                            proj_rate_lookback=preparator_description['proj_rate_lookback'],
                                            temperature_lookback=preparator_description['temperature_lookback'],
                                            visibility_lookback=preparator_description['visibility_lookback'],
                                            wind_lookback=preparator_description['wind_lookback'],
                                            precip_lookback=preparator_description['precip_lookback'],
                                            rway_lookback=preparator_description['rway_lookback'],
                                            changes_lookback=preparator_description['changes_lookback'],
                                            config_lookback=preparator_description['config_lookback'])
    
    for k in range(1,num_submodels+1):

        submodel_pred = return_model_ouput(df, airport, k, preparator)

        N, P = submodel_pred.shape

        if P > 2:
            submodel_pred = topk_scaled_prob(submodel_pred, 3)
            for p in range(P):
                submodel_preds.append(submodel_pred[:,p])
        else:
            submodel_preds.append(submodel_pred[:,0])
    
    submodel_preds = np.stack(submodel_preds,axis=1)

    config = preparator.config_feature_df.to_numpy()
    wind = preparator.wind_feature_df.to_numpy()
    time = preparator.time_feature_df.to_numpy()

    X_pred = np.hstack((submodel_preds,wind,time,config))

    X_pred = X_pred.astype('float32')
    
    return X_pred


if __name__ == "__main__":

    for airport in AIRPORTS:

        df = pd.read_csv(f"{DATA_DIR}/{airport}/{airport}_processed_data.csv",index_col=0)
        
        submodel_pred_df = predict_submodels(df, airport)
        submodel_pred_df.to_csv(f"{DATA_DIR}/{airport}/{airport}_submodel_outputs.csv")

        print("\n"+100*"-"+"\n")
        print("Finished Airport: " + airport)
        print("\n"+100*"-"+"\n")