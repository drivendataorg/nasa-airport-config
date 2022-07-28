from src.GeneralUtilities import *

def CheckPredictions(prediction_path, submission_format_path):
    """
    Checks the predictions for a single timepoint to make sure the expected columns are present,
    all of the timestamps are equal to the expected timestamp, and the probabiilties for
    configurations at a single airport, timestamp, and lookahead sum to 1.
    """

    DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
    
    columns = [
        "airport",
        "timestamp",
        "lookahead",
        "config",
    ]

    if not prediction_path.exists(): return False

    prediction = pd.read_csv(prediction_path)
    submission_format = pd.read_csv(submission_format_path)
    prediction_time = pd.to_datetime(submission_format["timestamp"].max())
    

    if not submission_format.columns.equals(prediction.columns): return False

    if not submission_format[columns].equals(prediction[columns]): return False

    if not (prediction.timestamp == prediction_time.strftime(DATETIME_FORMAT)).all(): return False

    airport_config_probability_sums = prediction.groupby(
        ["airport", "timestamp", "lookahead"]
    ).active.sum()
    
    if not np.allclose(airport_config_probability_sums, 1): return False
    
    return True