"""
This is the pure training code for model generation
from the stsb dataset with two languages and crossings.
"""

import itertools
import random
import math
import logging
import json
import os
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
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

langs = ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ru", "zh"]
lang_combinations = list(itertools.combinations(langs, 2))
print(lang_combinations)
print(len(lang_combinations))

lang_combinations = [lc for lc in lang_combinations if ("de" in lc or "en" in lc)]
print(lang_combinations)
print(len(lang_combinations))

model_name = "xlm-r-distilroberta-base-paraphrase-v1"
base_output_path = "./models"


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


def load_test_stsb_data(languages):
    data_per_language = []
    for language in languages:
        stsb_data_test = list(
            load_dataset("stsb_multi_mt", name=language, split="test")
        )
        data_per_language.append(stsb_data_test)
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


def load_test_data(languages, crossings):
    all_stsb_data = load_test_stsb_data(languages)
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


def test_model(model_dir, languages, crossings, params):
    # load all data incl. crossings
    test_data_stsb_cross = load_test_data(languages, crossings)
    assert len(test_data_stsb_cross) == 4
    assert len(test_data_stsb_cross[0]) == 1379

    lang_1_stsb = test_data_stsb_cross[0]
    lang_2_stsb = test_data_stsb_cross[1]
    assert len(lang_1_stsb) == len(lang_2_stsb) == 1379

    # test data of crossings only
    lang_cross_stsb = list(itertools.chain.from_iterable(test_data_stsb_cross[2:]))
    assert len(lang_cross_stsb) == 1379 * 2

    # test data of lang1, lang1 and crossings
    lang_all_stsb = list(itertools.chain.from_iterable(test_data_stsb_cross))
    assert len(lang_all_stsb) == 1379 * 4

    # convert to sentence_transformers datasets
    lang_1_stsb = to_input_example(lang_1_stsb)
    lang_2_stsb = to_input_example(lang_2_stsb)
    lang_cross_stsb = to_input_example(lang_cross_stsb)
    lang_all_stsb = to_input_example(lang_all_stsb)

    # load model from dir
    model = SentenceTransformer(model_dir)

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        lang_1_stsb, name="test_lang_1_stsb", main_similarity=SimilarityFunction.COSINE
    )
    result_lang_1_stsb = test_evaluator(model)

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        lang_2_stsb, name="test_lang_2_stsb", main_similarity=SimilarityFunction.COSINE
    )
    result_lang_2_stsb = test_evaluator(model)

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        lang_cross_stsb,
        name="test_lang_cross_stsb",
        main_similarity=SimilarityFunction.COSINE,
    )
    result_lang_cross_stsb = test_evaluator(model)

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        lang_all_stsb,
        name="test_lang_all_stsb",
        main_similarity=SimilarityFunction.COSINE,
    )
    result_lang_all_stsb = test_evaluator(model)

    params["lang_1_test_result_spearman"] = result_lang_1_stsb
    params["lang_2_test_result_spearman"] = result_lang_2_stsb
    params["lang_cross_test_result_spearman"] = result_lang_cross_stsb
    params["lang_all_test_result_spearman"] = result_lang_all_stsb
    params["languages"] = languages

    print("test_results:", params)

    with open(os.path.join(model_dir, "test_results.json"), "w") as outfile:
        json.dump(params, outfile)


def fit_model(params, languages, train_data):
    print("######################")
    print("start of languages:", languages)

    batch_size = params["train_batch_size"]
    num_epochs = params["num_epochs"]
    lr = params["lr"]
    eps = params["eps"]
    weight_decay = params["weight_decay"]
    warmup_steps_mul = params["warmup_steps_mul"]

    model = SentenceTransformer(model_name)

    # create train dataloader
    # train_sentece_dataset = SentencesDataset(train_fold, model=model) # this is deprecated
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    # define loss
    train_loss = losses.CosineSimilarityLoss(model=model)

    warmup_steps = math.ceil(
        len(train_data) * num_epochs / batch_size * warmup_steps_mul
    )

    output_path = os.path.join(
        base_output_path,
        f"cross-{languages[0]}-{languages[1]}-roberta-sentence-transformer",
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

    # save model
    model.save(output_path)

    return output_path


def train(params, languages):
    crossings = [[0, 1], [1, 0]]

    # load all data incl. crossings
    all_data_stsb_cross_data = load_all_data(languages, crossings)
    assert len(all_data_stsb_cross_data) == 4
    assert len(all_data_stsb_cross_data[0]) == 5749 + 1500

    # flatten
    all_data_stsb = list(itertools.chain.from_iterable(all_data_stsb_cross_data))
    assert len(all_data_stsb) == (5749 + 1500) * 4

    # shuffle with seed
    random.Random(42).shuffle(all_data_stsb)

    # convert to sentence_transformers datasets
    train_data = to_input_example(all_data_stsb)

    # fit the model
    model_dir = fit_model(params, languages, train_data)

    # test the model
    test_model(model_dir, languages, crossings, params)


if __name__ == "__main__":
    params = {
        "eps": 4.462251033010287e-06,
        "lr": 1.026343323298136e-05,
        "num_epochs": 2,
        "train_batch_size": 8,
        "warmup_steps_mul": 0.1609010732760181,
        "weight_decay": 0.04794438776350409,
    }

    for lc in lang_combinations:
        print("### Starting", lc)
        train(params, lc)
