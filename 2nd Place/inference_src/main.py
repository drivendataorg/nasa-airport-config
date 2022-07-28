import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from datetime import timedelta
from pathlib import Path
import time
import sys
import os
from contextlib import contextmanager
from loguru import logger
import pandas as pd
import typer

from src.const import *
from src.make_predictions import make_predictions

logger.remove()
logger.add(sys.stdout, level="INFO")

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

try:
    os.mkdir(PROCESSED_DATA)

except:
    None

count = 1
def main(prediction_time: datetime):

    start = time.time()

    logger.info("Processing {}", prediction_time)
    logger.debug("Copy partial submission format to prediction.")

    with suppress_stdout():

        date_range = (pd.date_range(start=prediction_time-timedelta(hours=6), end=prediction_time, freq='30T')).to_frame(name = 'timestamp')

        # Make Predictions
        pred_df, dp_time, sub_time, final_time = make_predictions(SUBMISSION_FORMAT, date_range)
        pred_df.to_csv(SUBMISSION_FILE, index=False)

    end = time.time()

    total_time = (end - start)/60
    print("\n"+100*"="+"\n")
    print(f"Data Processing Time: {dp_time} min \n")
    print(f"Submodel Time: {sub_time} min \n")
    print(f"Final Model Time: {final_time} min \n")
    print(f"Total Time: {total_time} min \n")
    print("\n"+100*"="+"\n")

    print("\n" + 100*"*" + "\n")
    print(f"Finished Prediction Hour: {prediction_time.strftime(DATETIME_FORMAT)}")
    print("\n" + 100*"*" + "\n")
    

if __name__ == "__main__":
    typer.run(main)
    
