# Practical Machine Learning and Deep Learning - Assignment 2 - Movie Recommender System

## Task description

A recommender system is a type of information filtering system that suggests items or content to users based on their interests, preferences, or past behavior. These systems are commonly used in various domains, such as e-commerce, entertainment, social media, and online content platforms.

Your assignment is to create a recommender system of movies for users:
* Your system should suggest some movies to the user based on user's gemographic information(age, gender, occupation, zip code) and favorite movies (list of movie ids).
* Solve this task using a machine learning model. You may consider only one model: it will be enough.
* Create a benchmark that would evaluate the quality of recommendations of your model. Look for commonly used metrics to evaluate a recommender system and use at least one metric.
* Make a single report decribing data exploration, solution implementation, training process, and evaluation on the benchmark.
* Explicitly state the benchmark scores of your systems.

## Usage

First, you need to install all dependencies. In order to do this, you can run the following command:
```bash
pip install -r requirements.txt
```

Evalution is done using `evaluate.py` script.
```bash
python ./benchmark/evaluate.py --user_id <DESIRED USER ID> -k <TOP K RECOMMENDATIONS> [--use_checkpoints]
```
If `--use_checkpoints` flag is set, then the checkpoints from `models/wide_deep` will be used for inference. Otherwise, the model will be continue training from a checkpoint or start training from a scratch.

All generated responses are stored in `benchmark/results`.

