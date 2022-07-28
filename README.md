[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/airportconfig-tile.jpg)

# Run-way Functions: Predict Reconfigurations at US Airports

## Goal of the Competition

Coordinating our nation’s airways is the role of the National Airspace System (NAS). The NAS is arguably the most complex transportation system in the world. Operational changes can save or cost airlines, taxpayers, consumers, and the economy at large thousands to millions of dollars on a regular basis. It is critical that decisions to change procedures are done with as much lead time and certainty as possible. The NAS is investing in new ways to bring vast amounts of data together with state-of-the-art machine learning to improve air travel for everyone.

An important part of this equation is airport configuration, the combination of runways used for arrivals and departures and the flow direction on those runways. For example, one configuration may use a set of runways in a north-to-south flow (or just "south flow") while another uses south-to-north flow ("north flow"). Air traffic officials may change an airport configuration depending on weather, traffic, or other inputs.

The goal of this challenge was to automatically predict airport configuration changes from real-time data sources including air traffic and weather. Better algorithms for predicting future airport configurations can support critical decisions, reduce costs, conserve energy, and mitigate delays across the national airspace network.

## What's in this Repository

This repository contains code from winning competitors in the [Run-way Functions: Predict Reconfigurations at US Airports](https://www.drivendata.org/competitions/89/competition-nasa-airport-configuration/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Score   | Summary of Model
----- | ------------ | ------- | ---------------- |
1   | Stuytown2      | 0.07385 | Train a CatBoost model (one for each airport and lookahead) to predict the active configuration from air traffic and weather features sampled at 15 minute intervals.
2   | MIT AeroAstro  | 0.08956 | First, XGBoost submodels predict intermediate features such as whether a specific runway would be used, whether a change will occur in a time period. Finally, a dense neural network learned to compute the output probability for each airport configuration and lookahead.
3   | AZ--KA         | 0.09122 | Predicts future airport configurations using an XGBoost model (one for each airport) trained on past airport configuration features such as the distribution of past configurations and the duration that each past configuration was active.
4   | TeamPSU        | 0.10583 | For each airport, train logistic regression to predict the active configuration using a combination of air traffic and weather features.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post coming soon!**
