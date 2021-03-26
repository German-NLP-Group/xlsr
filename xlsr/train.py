import itertools
import random
import math
import numpy as np
from sklearn.model_selection import KFold
import optuna
from torch.utils.data import DataLoader
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

from load_data import load_stsb_train_dev_data, load_xnli_data


study_name = "stsb_xnli_de_en_cross_01"
model_name = "xlm-r-distilroberta-base-paraphrase-v1"
max_folds = 4


def load_all_stsb_data(languages):
    data_per_language = []
    for language in languages:
        stsb_data = load_stsb_train_dev_data(language)
        assert len(stsb_data) == 5749 + 1500
        data_per_language.append(stsb_data)
    return data_per_language


def load_all_xnli_data(languages, label_map):
    data_per_language = []
    for language in languages:
        xnli_data = load_xnli_data(language, label_map)
        assert len(xnli_data) == 7500
        data_per_language.append(xnli_data)
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


def load_all_data(languages, label_map, crossings):
    all_stsb_data = load_all_stsb_data(languages)
    all_xnli_data = load_all_xnli_data(languages, label_map)
    all_data_stsb_cross_data = add_cross_data(all_stsb_data, crossings)
    all_data_xnli_cross_data = add_cross_data(all_xnli_data, crossings)
    return all_data_stsb_cross_data, all_data_xnli_cross_data


def to_input_example(language_list):
    result = []
    for dataset in language_list:
        result.append(
            InputExample(
                texts=[dataset["sentence1"], dataset["sentence2"]],
                label=dataset["similarity_score"],
            )
        )
    return result


def fit_model(trial, train_fold, val_fold, fold_index):
    print("######################")
    print("start of fold_index:", fold_index)
    print("len(train_fold)", len(train_fold))
    print("len(val_fold)", len(val_fold))

    batch_size = trial.suggest_int("batch_size", 4, 50)
    num_epochs = trial.suggest_int("num_epochs", 1, 3)
    lr = trial.suggest_uniform("lr", 2e-6, 2e-4)
    eps = trial.suggest_uniform("eps", 1e-7, 1e-5)
    weight_decay = trial.suggest_uniform("weight_decay", 0.001, 0.1)
    warmup_steps_mul = trial.suggest_uniform("warmup_steps_mul", 0.1, 0.5)

    model = SentenceTransformer(model_name)

    # create train dataloader
    train_sentece_dataset = SentencesDataset(train_fold, model=model)
    train_dataloader = DataLoader(
        train_sentece_dataset, shuffle=True, batch_size=batch_size
    )

    # define loss
    train_loss = losses.CosineSimilarityLoss(model=model)

    warmup_steps = math.ceil(
        len(train_sentece_dataset) * num_epochs / batch_size * warmup_steps_mul
    )
    print("warmup_steps:", warmup_steps)

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
    label_map = {
        "contradiction": -1.0,
        "entailment": 1.0,
        "neutral": 0.0,
    }
    crossings = [[0, 1], [1, 0]]

    # load all data incl. crossings
    all_data_stsb_cross_data, all_data_xnli_cross_data = load_all_data(
        languages, label_map, crossings
    )
    assert len(all_data_stsb_cross_data) == 4
    assert len(all_data_stsb_cross_data[0]) == 5749 + 1500
    assert len(all_data_xnli_cross_data) == 4
    assert len(all_data_xnli_cross_data[0]) == 7500

    # xnli data can be flattened here since we do no xvalidation on it
    # we only mix it into the train set
    all_data_xnli_cross_data = list(
        itertools.chain.from_iterable(all_data_xnli_cross_data)
    )
    assert len(all_data_xnli_cross_data) == 7500 * 4

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

        # add xnli to train data
        train_fold.extend(all_data_xnli_cross_data)

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

        if (mean_result < 0.1) or (fold_index >= max_folds - 1):
            return mean_result


def ex_wrapper(trial):
    try:
        train(trial)
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
    )

    study.optimize(ex_wrapper)
