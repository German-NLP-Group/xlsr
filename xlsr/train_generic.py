"""
This is the training code for automated hyperparameter search with Optuna
for the stsb dataset with two languages and crossings.

No XNLI data is used.
"""

import logging
import math
import os
import random

import numpy as np
import optuna
import yaml
from datasets import load_dataset
from hpoflow import SignificanceRepeatedTrainingPruner
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from torch.utils.data import DataLoader


# init root logger and logger
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.addHandler(logging.StreamHandler())
_logger = logging.getLogger(__name__)

# read and set config from yaml
with open("train_config.yaml", "r") as train_config_file:
    train_config = yaml.safe_load(train_config_file)
_logger.info("train_config: %s", train_config)
# study_name = "stsb_no_cv_remote_02"
study_name = train_config["study_name"]
# model_name = "xlm-r-distilroberta-base-paraphrase-v1"
model_name = train_config["model_name"]
# storage="sqlite:///optuna.db"
# optuna_storage="postgresql://optuna:{OPTUNA_PASS}@{OPTUNA_HOSTNAME}/optuna?sslmode=verify-full&sslrootcert=./root.crt"
optuna_storage = train_config["optuna_storage"]

max_folds = 10

OPTUNA_PASS = os.getenv("OPTUNA_PASS")
OPTUNA_HOSTNAME = os.getenv("OPTUNA_HOSTNAME")


def load_all_stsb_data(languages, split: str):
    data_per_language = []
    for language in languages:
        stsb_data = list(load_dataset("stsb_multi_mt", name=language, split=split))
        data_per_language.append(stsb_data)
    return data_per_language


def add_cross_data(data_per_language, crossings):
    result = data_per_language.copy()
    for crossing in crossings:
        language_data = []
        for data_left, data_right in zip(
            data_per_language[crossing[0]], data_per_language[crossing[1]]
        ):
            sentence1 = data_left["sentence1"]
            sentence2 = data_right["sentence2"]
            assert data_left["similarity_score"] == data_right["similarity_score"]
            data_new = {
                "sentence1": sentence1,
                "sentence2": sentence2,
                "similarity_score": data_right["similarity_score"],
            }
            language_data.append(data_new)
        result.append(language_data)
    return result


def load_all_data(languages, crossings, split: str):
    all_stsb_data = load_all_stsb_data(languages, split)
    all_data_stsb_cross_data = add_cross_data(all_stsb_data, crossings)
    return all_data_stsb_cross_data


def to_input_example(language_list):
    result = []
    for dataset in language_list:
        result.append(
            InputExample(
                texts=[dataset["sentence1"], dataset["sentence2"]],
                label=(dataset["similarity_score"] / 5),
            )
        )
    return result


def fit_model(trial, train_fold, val_fold, fold_index):
    _logger.info("######################")
    _logger.info("start of fold_index: %s", fold_index)
    _logger.info("len(train_fold): %s", len(train_fold))
    _logger.info("len(val_fold): %s", len(val_fold))

    batch_size = trial.suggest_int("train_batch_size", 4, 80)
    num_epochs = trial.suggest_int("num_epochs", 1, 8)
    lr = trial.suggest_float("lr", 2e-6, 2.5e-4)
    eps = trial.suggest_float("eps", 1e-7, 1e-4)
    weight_decay = trial.suggest_float("weight_decay", 0.0005, 0.1)
    warmup_steps_mul = trial.suggest_float("warmup_steps_mul", 0.1, 0.7)

    model = SentenceTransformer(model_name)

    # create train dataloader
    train_dataloader = DataLoader(train_fold, shuffle=True, batch_size=batch_size)

    # define loss
    train_loss = losses.CosineSimilarityLoss(model=model)

    warmup_steps = math.ceil(len(train_fold) * num_epochs / batch_size * warmup_steps_mul)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr, "eps": eps},  # , "correct_bias": False},
        weight_decay=weight_decay,
    )

    # evaluate the model
    val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_fold, name="val_set", main_similarity=SimilarityFunction.COSINE
    )
    result = val_evaluator(model)

    _logger.info("######################################################")
    _logger.info("test result: %s", result)
    _logger.info("######################################################")

    if math.isnan(result):
        result = 0.0

    return result


def train(trial):
    languages = ["en", "de"]
    crossings = [[0, 1], [1, 0]]

    # load all train data incl. crossings
    all_data_stsb_cross_data_train = load_all_data(languages, crossings, split="train")
    assert len(all_data_stsb_cross_data_train) == 4
    assert len(all_data_stsb_cross_data_train[0]) == 5749
    all_data_stsb_cross_data_train = [
        item for sublist in all_data_stsb_cross_data_train for item in sublist
    ]
    assert len(all_data_stsb_cross_data_train) == 4 * 5749

    # load all dev data incl. crossings
    all_data_stsb_cross_data_dev = load_all_data(languages, crossings, split="dev")
    assert len(all_data_stsb_cross_data_dev) == 4
    assert len(all_data_stsb_cross_data_dev[0]) == 1500
    all_data_stsb_cross_data_dev = [
        item for sublist in all_data_stsb_cross_data_dev for item in sublist
    ]
    assert len(all_data_stsb_cross_data_dev) == 4 * 1500

    results = []

    for fold_index in range(max_folds):
        train_fold = all_data_stsb_cross_data_train
        val_fold = all_data_stsb_cross_data_dev

        assert len(train_fold) + len(val_fold) == (5749 + 1500) * 4

        # shuffle with seed
        random.Random(fold_index).shuffle(train_fold)

        # convert to sentence_transformers datasets
        train_fold = to_input_example(train_fold)
        val_fold = to_input_example(val_fold)

        # fit the model
        result = fit_model(trial, train_fold, val_fold, fold_index)

        results.append(result)
        trial.set_user_attr("results", str(results))
        mean_result = np.mean(results)

        # hard pruning
        if mean_result < 0.1:
            _logger.info("### HARD PRUNING")
            break

        trial.report(result, fold_index)
        if trial.should_prune():
            _logger.info("### PRUNING")
            break

    return mean_result


def ex_wrapper(trial):
    try:
        return train(trial)
    except Exception as e:
        _logger.warning("Exception in trial!", exc_info=True)  # TODO: add more info from trial
        trial.set_user_attr("exception", str(e))
        return float("nan")


if __name__ == "__main__":
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True),
        study_name=study_name,
        # storage="sqlite:///optuna.db",
        # storage=f"postgresql://optuna:{OPTUNA_PASS}@{OPTUNA_HOSTNAME}/optuna?sslmode=verify-full&sslrootcert=./root.crt",
        storage=optuna_storage.format(OPTUNA_PASS=OPTUNA_PASS, OPTUNA_HOSTNAME=OPTUNA_HOSTNAME),
        load_if_exists=True,
        direction="maximize",
        pruner=SignificanceRepeatedTrainingPruner(
            alpha=0.4,
            n_warmup_steps=4,
        ),
    )

    study.optimize(ex_wrapper)
