<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img alt="Logo" src="images/plane.jpg" width="800" height="360">
  <h3 align="center"> NASA Run-way competition: Predict Reconfigurations at US Airports </h3>

  <p align="center">
    Stuytown team repository to develop the models and serve the prediction functionalities for the 2022 NASA Runway prediction challenge 
    <br />
    <a href="https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened/page/440/"><strong>Visit competition site »</strong></a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
## Table of contents
  <ol>
    <li> <a href="#context-of-the-challenge">Context of the challenge</a> </li>
    <li><a href="#approach-of-our-solution">Approach of our solution</a></li>
    <li><a href="#results-obtained">Results obtained</a></li>
    <li><a href="#repository-structure">Repository structure</a></li>
    <li><a href="#usage">Usage</a></li>
     <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#execution">Execution</a></li>
      </ul>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>



<!-- CONTEXT -->
## Context of the challenge

<div align="justify">

Every year millions of flights arrive and depart from US airports. Intelligent scheduling and system control is a very pressing problem. Knowing in advance or accurately forecasting which runways will be available for departure and arrival is one of the key inputs to arrange optimized schedules. With the surge of Big Data and Advanced Analytics techniques most of these forecasts are based on Machine Learning methods.

In a [2022 analytics competition](https://www.drivendata.org/competitions/89/competition-nasa-airport-configuration/), NASA partnered with DrivenData to facilitate real-world data to develop these types of models and suggest alternative approaches to the [existing research](https://aviationsystems.arc.nasa.gov/publications/2021/20210017593_Khater_Aviation2021_paper.pdf). 

More precisely, participants in this challenge were given the task to build an Advanced Analytics model capable of forecasting the likelihood of multiple configurations taking place for 10 US airports at 12 different time horizons ranging from 30 minutes to 6 hours ahead of time. Each configuration contains information about which runways are open for departure and arrivals respectively and in which direction they can be utilized. 

A graphical representation of the desired output by the model can be observed below as a grid crossing airports, configurations and lookahead periods:


<img src="images/predictions_format.JPG" alt="Logo" width="450" height="360">

The metric with which the above forecasts are evaluated is with the aggregated logloss across airports and lookahead periods.


<!-- APPROACH -->
## Approach of our solution

With this context in mind, our solution has been to follow a 4-step approach as depicted in the exhibit below:

<img src="images/approach.JPG" alt="Logo" width="700" height="260">

As one can note, our approach relies on building one model for every combination of airport-lookahead period to predict the likelihood of each possible configuration in the prescribed airport and lookahead horizon. This totals 120 models.

<!-- RESULTS -->
## Results obtained

The results obtained can be split in three different metrics depending on the time period where our models have been evaluated. The open leaderboard is where participants were able to submit the predictions during the development stage, private leaderboard performance was a held out set used to test whether the functionalities in the runtime environment worked correctly, and finally, the held out validation set of May 2022 is where the final ranking was decided.

- Open leaderboard performance: 0.0615 (Rank 5)
- Private leaderboard performance: 0.1043 (Rank 3)
- Held out validation set performance: 0.0739 (Rank 1)

As one can note, based on the evaluation metric with which the final leaderboard was determined, ours was the winning solution. Below one can see the top 4 teams:

<img src="images/ranking.JPG" alt="Logo" width="430" height="140">

<!-- REPOSITORY -->
## Repository structure

The approach and results discussed earlier are structured in two main subrepositories to tackle separate tasks. There is a *training* task under `/code` and a *prediction* task under `/submission`. The *training* task consists of a pipeline which loads the raw data, creates a master table and generates 120 models, one for each combination of airport-lookahead. And the *prediction* task is built to run on DrivenData's runtime environment to retrieve live forecasts for new and unseen data. The skeleton of the repository can be found below:

```
NASA-runways
│   README.md
│
└───code
│   │   main.py
│   └───src
│       │   CheckPredictions.py
│       │   CreateMaster.py
│       │   ExtractFeatures.py
|       |   GeneralUtilities.py
|       |   GeneratePredictions.py
│
└───sumbission
│   │   main.py
│   └───src
│   └───models
└───data
    │   katl
    │   kclt
    │   kden
    │   kdfw
    │   kjfk
    │   kmem
    │   kmia
    │   kord
    │   kphx
    │   ksea
    │   open_submission_format.csv
    │   master_table.parquet
```

<!-- USAGE -->
## Usage

Having an understanding of the approach, results and repository structure, we will cover the usage of the developed code in this fifth section. First studying the necessary requirements and then specifying the execution steps. 

### Prerequisites

In order to execute the routines mentioned above two sets of requirements are needed:

- To execute *main.py* in /submission: Replicate the docker image of the [runtime DrivenData repository](https://github.com/drivendataorg/nasa-airport-config-runtime)
- To execute *main.py* in /code: Use Python 3.8 and have the packages from [*requirements.txt*](https://github.com/alsaco33/NASA-runways/blob/main/requirements.txt) installed.

```bash
conda create -n airport-config python=3.8
conda activate airport-config
pip install -r requirements.txt
```

### Execution

Once the previous requirements are met, the execution of the data and modelling pipeline can be triggered as displayed in the snippet below. The user specifies the data path (where the raw data lives in `csv.bz2` files with one directory per airport), the models path (where the models will be stored after training) and a binary flag (indicating whether we want the master table to be created from scratch or read).

```python
$ python3 main.py data_path models_path build_master
```

So a sample execution could be accomplished by running:

```python
$ python3 main.py "C:/Users/.../data_folder/" "C:/Users/.../models_folder/" True
```

Executing the above will trigger the following sequence of 10 steps automatically:
  1. Load raw data and store it in a dictionary that maps airport + key -> DataFrame
  2. Create cross join of all the dates between start_time and end_time at a 15min frequency (master table at the airport-timestamp level)
  3. Extract features from past periods for the selected data blocks and append them to the master table
  4. Adjust master table in order not to have errors in edge cases in prediction time
  5. Add targets to the master table - future information indicating the configuration at each lookahead period
  6. Store the master table as a parquet file in the *data_path* specified (130Mb)
  7. Define the train / test split based on the *open_submission* file in the *data_path*
  8. Choose the appropriate modelling parameters based on a grid for each airport-lookahead combination
  9. Build a Catboost classifier model for each airport and lookahead period with the chosen parameters (a total of 120)
  10. Save the models in the *models_path* specified

This process is memory and time intensive. The data pipeline (loading the raw data and creating the master table) requires 31Gb in RAM at its peak and a total runtime of 4h while the modelling process that reproduces the submitted models was executed enabling the GPU option in code/main.py in line 79:

```python
task_type='GPU'
```

And in order to run the submission *main.py* script in DrivenData's [runtime environment](https://github.com/drivendataorg/nasa-airport-config-runtime) with a predefined *prediction_time* as it was done in the evaluation phase of the competition:

```python
main(prediction_time)
```

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

As a closing remark, we would like to place on record our deepest sense of gratitude towards the NASA and DrivenData teams for organizing and hosting this competition respectively. Their continuous support and encouragement have been invaluable to develop the final product described earlier.

</div>
