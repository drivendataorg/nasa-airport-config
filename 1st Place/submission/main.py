from src.GeneralUtilities import *
from src.CreateMaster import CreateMaster
from src.GeneratePredictions import GeneratePredictions, BuildDummyPrediction
from src.CheckPredictions import CheckPredictions

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

feature_directory = Path("/codeexecution/data")
prediction_path = Path("/codeexecution/prediction.csv")

def main(prediction_time: datetime):
    """
    Main function that sequentially calls the data pipeline, feature extraction and prediction routines
    to retrieve and store a table at the airport-timestamp-lookahead-configuration level specifying
    the probability of each configuration being active lookahead minutes ahead of time in each airport
    and timestamp
    
    :param datetime prediction_time: Timestamp for which predictions will be returned
    
    :return None: Predictions are stored as a csv in prediction_path and no object is returned
    """
    
    logger.info("################### Started at time {}", time.ctime())
    
    sub_format_path = feature_directory / "partial_submission_format.csv"
    
    try:
        # Load the format of the desired output for the specified prediction_time
        to_submit = pd.read_csv(sub_format_path, parse_dates=["timestamp"])
        to_submit.drop(columns = "active", inplace = True)
        prediction_datetime = to_submit["timestamp"].max()
        airports = to_submit["airport"].unique()

        # Create the master table at an airport-timestamp level
        logger.info("Creating master table for {}", time.ctime())
        master_table = CreateMaster(data_path=feature_directory, 
                                    airports=airports,
                                    start_time=prediction_datetime-pd.Timedelta("48h"), 
                                    end_time=prediction_datetime,
                                    enlarged=False
                                   )

        # Retrieve predictions for each airport-configuration in a given timestamp
        predictions = GeneratePredictions(to_submit=to_submit,
                                          models_path="./models",
                                          master_table=master_table)

        # Store predictions in prediction_path
        predictions.to_csv(prediction_path, date_format=DATETIME_FORMAT, index=False)
        logger.info("Saved predictions for {}", prediction_datetime)
        
    except:
        BuildDummyPrediction(sub_format_path=sub_format_path, prediction_path=prediction_path)
        logger.info("Saved dummy predictions for {}", prediction_time)
    
    checks = CheckPredictions(prediction_path=prediction_path, submission_format_path=sub_format_path)
    logger.info("After performing checks the result is {}", checks)
    
    if not checks:
        BuildDummyPrediction(sub_format_path=sub_format_path, prediction_path=prediction_path)
        
    logger.info("################### Finished at time {}", time.ctime())

if __name__ == "__main__":
    typer.run(main)
