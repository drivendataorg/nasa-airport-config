import pandas as pd
from sklearn import datasets
from sklearn import linear_model
import datetime as dt
import numpy as np
import math


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


submission_format = pd.read_csv(
    "../data/partial_submission_format.csv", parse_dates=["timestamp"]
)
# open_training_labels = pd.read_csv("../data/open_train_labels.csv.bz2", parse_dates=["timestamp"], compression = "bz2")
airports = submission_format["airport"].unique().tolist()
start_time = submission_format["timestamp"].min()
max_labels = 0
needed_reports = dict()

for air in airports:
    possible_labels = submission_format[submission_format["airport"] == air][
        "config"
    ].unique()
    if max_labels < len(possible_labels.tolist()):
        max_labels = len(possible_labels.tolist())
    possible_config = possible_labels.copy()
    counter = 0
    for i in possible_labels:
        possible_config[counter] = i.split(":")[1]
        counter += 1
    cur_config = pd.read_csv(
        f"../data/{air}/{air}_airport_config.csv", parse_dates=["timestamp"]
    )
    arrivals = pd.read_csv(f"./{air}_landing_dist.csv")
    departures = pd.read_csv(f"./{air}_take_off_dist.csv")
    dpTimeStamps = []
    arTimeStamps = []
    for i, r in arrivals.iterrows():
        arTimeStamps.append(start_time + dt.timedelta(minutes=i * 30))
    for i, r in departures.iterrows():
        dpTimeStamps.append(start_time + dt.timedelta(minutes=i * 30))
    arrivals["timestamp"] = arTimeStamps
    departures["timestamp"] = dpTimeStamps

    weather_file = f"../data/{air}/{air}_lamp.csv"
    weather_data = pd.read_csv(
        weather_file, parse_dates=["timestamp", "forecast_timestamp"]
    )

    weather_data["cloud"] = (
        0 * (weather_data["cloud"] == "CL")
        + 1 * (weather_data["cloud"] == "FW")
        + 5 * (weather_data["cloud"] == "SC")
        + 9 * (weather_data["cloud"] == "BK")
        + 10 * (weather_data["cloud"] == "OV")
    )
    weather_data["lightning_prob"] = (
        1 * (weather_data["lightning_prob"] == "L")
        + 2 * (weather_data["lightning_prob"] == "M")
        + 3 * (weather_data["lightning_prob"] == "H")
    )
    weather_data["precip"] = (weather_data["precip"]) * 1
    weather_data["wind_direction_cos"] = np.cos(
        weather_data["wind_direction"] * 10 * np.pi / 360
    )
    weather_data["wind_direction_sin"] = np.sin(
        weather_data["wind_direction"] * 10 * np.pi / 360
    )
    weather_data = weather_data.drop("wind_direction", 1)
    pd.DataFrame(possible_labels).to_csv(f"{air}_possibel_config")
    needed_reports[air] = (
        weather_data,
        departures,
        arrivals,
        cur_config,
        possible_config,
        possible_labels,
    )
    print("Preprocess stage 2: " + str(air))

opened = submission_format.copy()
opened = opened.drop("config", axis=1).drop("active", axis=1)
opened = opened.drop_duplicates()
training_file = pd.DataFrame(
    {
        "airport": [],
        "temperature": [],
        "wind_speed": [],
        "wind_gust": [],
        "cloud_ceiling": [],
        "visibility": [],
        "cloud": [],
        "lightning_prob": [],
        "precip": [],
        "wind_direction_cos": [],
        "wind_direction_sin": [],
        "depart1": [],
        "deaprt2": [],
        "depart3": [],
        "depart4": [],
        "arrive1": [],
        "arrive2": [],
        "arrive3": [],
        "arrive4": [],
        "lookahead": [],
    }
)
for i in range(max_labels):
    training_file.insert(19 + i, "cur_config_hot" + str(i), [])

training_file.to_csv("testing_data.csv")
counter = 0

for i, r in opened.iterrows():

    timestamp = r["timestamp"]
    lookahead = r["lookahead"]
    # Remove for "testing" code
    mask = needed_reports[r["airport"]][0]["timestamp"] <= timestamp
    masked_weather_data = needed_reports[r["airport"]][0][mask]

    # Get weather data
    latest_weather_intercept = masked_weather_data[
        masked_weather_data["timestamp"] == masked_weather_data["timestamp"].max()
    ]

    try:
        get_nearest = latest_weather_intercept[
            latest_weather_intercept["forecast_timestamp"]
            == nearest(
                latest_weather_intercept["forecast_timestamp"],
                timestamp + dt.timedelta(minutes=lookahead),
            )
        ]
        get_nearest = get_nearest.drop("timestamp", axis=1).drop(
            "forecast_timestamp", axis=1
        )
        weather_features = get_nearest.values.tolist()[0]
    except:
        weather_features = [np.nan] * 10
        # print("weather excepted")

    # Get latest take-off and landing estimates
    try:
        departure_reports = needed_reports[r["airport"]][1]
        mask = departure_reports["timestamp"] <= timestamp
        masked_departure = departure_reports[mask]
        latest_departure_report = masked_departure[
            masked_departure["timestamp"] == masked_departure["timestamp"].max()
        ]
        departure_features = latest_departure_report.values.tolist()[0][
            int(lookahead / 30) : int(lookahead / 30) + 4
        ]
    except:
        departure_features = [0] * 4
        # print("departure excepted")
    try:
        arrival_reports = needed_reports[r["airport"]][2]
        mask = arrival_reports["timestamp"] <= timestamp
        masked_arrival = arrival_reports[mask]
        latest_arrival_report = masked_arrival[
            masked_arrival["timestamp"] == masked_arrival["timestamp"].max()
        ]
        arrival_features = latest_arrival_report.values.tolist()[0][
            int(lookahead / 30) : int(lookahead / 30) + 4
        ]
    except:
        arrival_features = [0] * 4
        # print("departure excepted")
    # Get current configiration
    try:
        cur_config = needed_reports[r["airport"]][3]
        mask = cur_config["timestamp"] <= timestamp
        masked_cur_config = cur_config[mask]
        latest_cur_config = masked_cur_config[
            masked_cur_config["timestamp"] == masked_cur_config["timestamp"].max()
        ]
        cur_config_feature = [0] * max_labels
        airport_report_4 = needed_reports[r["airport"]]
        latest_print_config = latest_cur_config.values.tolist()
        cur_config_index = np.where(
            needed_reports[r["airport"]][4] == latest_cur_config.values.tolist()[0][1]
        )
        cur_config_index = (
            cur_config_index[0][0]
            if len(cur_config_index[0]) > 0
            else needed_reports[r["airport"]][5].size - 1
        )
        cur_config_feature[cur_config_index] = 1
    except:
        cur_config_feature = [0] * max_labels
        # print("cur config excepted wtf?")

    total_features = (
        [r["airport"]]
        + weather_features
        + departure_features
        + arrival_features
        + cur_config_feature
        + [r["lookahead"]]
    )
    training_file.loc[len(training_file)] = total_features
    if counter % 5000 == 0:
        print(str(counter) + " out of " + str(opened.shape[0]))
        training_file.to_csv("testing_data.csv", mode="a", index=True, header=False)
        training_file = pd.DataFrame(
            {
                "airport": [],
                "temperature": [],
                "wind_speed": [],
                "wind_gust": [],
                "cloud_ceiling": [],
                "visibility": [],
                "cloud": [],
                "lightning_prob": [],
                "precip": [],
                "wind_direction_cos": [],
                "wind_direction_sin": [],
                "depart1": [],
                "deaprt2": [],
                "depart3": [],
                "depart4": [],
                "arrive1": [],
                "arrive2": [],
                "arrive3": [],
                "arrive4": [],
                "lookahead": [],
            }
        )
        for k in range(max_labels):
            training_file.insert(19 + k, "cur_config_hot" + str(k), [])
    counter += 1
training_file.to_csv("testing_data.csv", mode="a", index=True, header=False)
