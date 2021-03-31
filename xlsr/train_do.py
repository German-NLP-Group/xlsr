"""
This is the training code for automated hyperparameter search with Optuna
for the stsb dataset with two languages and crossings.
"""

import itertools
import random
import math
import logging
import numpy as np
from sklearn.model_selection import KFold
import optuna
from torch.utils.data import DataLoader
from mltb.optuna import SignificanceRepeatedTrainingPruner
from datasets import load_dataset
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    SentencesDataset,
    losses,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)


# init root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(logging.StreamHandler())


study_name = "xlsr_de_en_cross_stsb_08_do_01"
model_name = "xlm-r-distilroberta-base-paraphrase-v1"
max_folds = 3


def load_all_stsb_data(languages):
    data_per_language = []
    for language in languages:
        stsb_data_train = list(
            load_dataset("stsb_multi_mt", name=language, split="train")
        )
        stsb_data_dev = list(load_dataset("stsb_multi_mt", name=language, split="dev"))
        stsb_data = stsb_data_train + stsb_data_dev
        assert len(stsb_data) == 5749 + 1500
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


def load_all_data(languages, crossings):
    all_stsb_data = load_all_stsb_data(languages)
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
    print("######################")
    print("start of fold_index:", fold_index)
    print("len(train_fold)", len(train_fold))
    print("len(val_fold)", len(val_fold))

    batch_size = trial.suggest_int("train_batch_size", 4, 50)
    num_epochs = trial.suggest_int("num_epochs", 1, 3)
    lr = trial.suggest_uniform("lr", 2e-6, 2e-4)
    eps = trial.suggest_uniform("eps", 1e-7, 1e-5)
    weight_decay = trial.suggest_uniform("weight_decay", 0.001, 0.1)
    warmup_steps_mul = trial.suggest_uniform("warmup_steps_mul", 0.1, 0.5)
    attention_probs_dropout_prob = trial.suggest_uniform("attention_probs_dropout_prob", 0.1, 0.3)
    hidden_dropout_prob = trial.suggest_uniform("hidden_dropout_prob", 0.1, 0.3)
    

    print("batch_size:", batch_size)
    print("num_epochs:", num_epochs)
    print("lr:", lr)
    print("eps:", eps)
    print("weight_decay:", weight_decay)
    print("warmup_steps_mul:", warmup_steps_mul)

    model = SentenceTransformer(model_name)
    
    # change dropout of the model
    model._modules['0'].auto_model.base_model.config.attention_probs_dropout_prob = attention_probs_dropout_prob
    model._modules['0'].auto_model.base_model.config.hidden_dropout_prob = hidden_dropout_prob


    # create train dataloader
    # train_sentece_dataset = SentencesDataset(train_fold, model=model) # this is deprecated
    train_dataloader = DataLoader(train_fold, shuffle=True, batch_size=batch_size)

    # define loss
    train_loss = losses.CosineSimilarityLoss(model=model)

    warmup_steps = math.ceil(
        len(train_fold) * num_epochs / batch_size * warmup_steps_mul
    )

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr, "eps": eps, "correct_bias": False},
        weight_decay=weight_decay,
    )

    # evaluate the model
    val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_fold, name="val_set", main_similarity=SimilarityFunction.COSINE
    )
    result = val_evaluator(model)

    print("######################################################")
    print("test result:", result)
    print("######################################################")

    if math.isnan(result):
        result = 0.0

    return result


def train(trial):
    languages = ["en", "de"]
    crossings = [[0, 1], [1, 0]]

    # load all data incl. crossings
    all_data_stsb_cross_data = load_all_data(languages, crossings)
    assert len(all_data_stsb_cross_data) == 4
    assert len(all_data_stsb_cross_data[0]) == 5749 + 1500

    xval_indexes = np.arange(len(all_data_stsb_cross_data[0]))

    results = []

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold_index, (train_indexes, val_indexes) in enumerate(kf.split(xval_indexes)):
        train_fold = []
        val_fold = []

        # interate all languages and the crossings
        # split to train and val
        for stsb_lang_data in all_data_stsb_cross_data:
            stsb_lang_data_array = np.array(stsb_lang_data)
            train_fold.extend(stsb_lang_data_array[train_indexes])
            val_fold.extend(stsb_lang_data_array[val_indexes])

        assert len(train_fold) + len(val_fold) == (5749 + 1500) * 4

        # shuffle with seed
        random.Random(42).shuffle(train_fold)
        random.Random(42).shuffle(val_fold)

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
            print("### HARD PRUNING")
            return mean_result

        trial.report(result, fold_index)
        if trial.should_prune():
            print("### PRUNING")
            break

    return mean_result


def ex_wrapper(trial):
    try:
        return train(trial)
    except Exception as e:
        print(e)
        trial.set_user_attr("exception", str(e))
        return float("nan")


if __name__ == "__main__":
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True),
        study_name=study_name,
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        direction="maximize",
        pruner=SignificanceRepeatedTrainingPruner(
            alpha=0.3,
            n_warmup_steps=max_folds,
        ),
    )

    study.optimize(ex_wrapper)
