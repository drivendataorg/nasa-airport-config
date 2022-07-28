## Summary

In our solution we only used the past airport configurations and training labels data. We preprocessed the data and for each data point. We extracted information about distribution of the past configurations, current and 10 last configurations, and the duration that each past configuration was active. Also, we considered current date and time (hour, day, week, month) and the lookahead as predictors. In preprocessing step, we used some of the [benchmark code](https://www.drivendata.co/blog/airport-configuration-benchmark/) functions like `censor_data` and `make_config_dist` for selecting part of the data that we were allowed to use and for creating distribution of past configurations, respectively. For more information about these functions, please refer to the benchmark code. At the end of preprocessing step, we created a DataFrame (`train_labels`) as an input for the machine learning algorithms.

In the main code, we trained XGBoost models for each airport. We then preprocessed the test data features and use pretrained XGBoost models for predicting probability of each configuration. 

## Setup

For required packages and their version please see [`environment-gpu.yml`](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/environment-gpu.yml) in the runtime repository.

## Hardware

The codes were run on the competition container runtime.

- Number of CPUs: 6
- Processor: Xeon E5-2690
- Memory: 56 GB
- GPU: Tesla K80
- Training and Inference time: ~6 hours ( Significant percentage of this duration was consumed by preprocessing step)

## Run Training and Inference

To run training and inference from the command line, just use the same command mentioned on the [Code submission format](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened/page/442/) section of the competition guide (e.g., `python main.py 2021-10-20T10:00:00`). Put the training data `train_data` folder and testing data in the `data` folder.

Run the preprocessing script to generate :

```bash
python src/preprocess.py
```

The code generates submission file and saves it in the `prediction` folder.
