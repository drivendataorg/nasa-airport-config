The steps to utulize the package is as followed:

## Step 0: setting up the computer

1. Ensure python3 is installed on command line.
2. Install all the dependencies in the competition runtime [environment-cpu.yml](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/environment-cpu.yml)
  - ensure the versions match exactly (packages such as tqdm and typer may not be backwards compatible or have hot fixes that are undesirable)
3. Download the training data into the `data` folder.
4. Place your `partial_submission_format.csv` in the data folder for the predictions you would like to make during testing.

## Step 1: setting up the models

Reproduce the data from the raw data provided by the competition.

1. `cd LR_training`
2. Run `python3 usage_preprocess_train.py`. This preprocesses raw landing and take-off data into a more histogram like data-structure. (the output of this program is included in the folder)
3. Run `python3 LR_preprocessing.py` to complete the processing. (the output of this program is included in the folder)
4. Run `python3 train_weather_temporal_p.py` (the output of this program is included in the folder)

## Step 2a: creating the submission files
1. Copy and paste all generated files (`*.pkl`) from `LR_training` to `submission_src`.
2. Follow the direction as specified in the competition runtime [README](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/README.md).
  - Run `make pull`
  - Run `make pack-submission`
  - (to test) run `make test-submission`
