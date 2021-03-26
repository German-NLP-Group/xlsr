import itertools
import numpy as np
from sklearn.model_selection import KFold

from load_data import load_stsb_train_dev_data, load_xnli_data


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


def train():
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

    # xnli data can be flattened here since we do no xvalidation on it
    # we only mix it into the train set
    all_data_xnli_cross_data = list(
        itertools.chain.from_iterable(all_data_xnli_cross_data)
    )

    xval_indexes = np.arange(len(all_data_stsb_cross_data[0]))

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_indexes, val_indexes in kf.split(xval_indexes):
        pass


if __name__ == "__main__":
    label_map = {
        "contradiction": -1.0,
        "entailment": 1.0,
        "neutral": 0.0,
    }

    all_stsb_data = load_all_stsb_data(["en", "de"])
    print("### first example from stsb")
    print(all_stsb_data[0][0])
    print(all_stsb_data[1][0])

    all_xnli_data = load_all_xnli_data(["en", "de"], label_map)
    print("### last example from xnli")
    print(all_xnli_data[0][-1])
    print(all_xnli_data[1][-1])

    all_data_stsb_cross_data = add_cross_data(all_stsb_data, [[0, 1], [1, 0]])
    print("### first example from stsb with crossing")
    print(all_data_stsb_cross_data[0][0])
    print(all_data_stsb_cross_data[1][0])
    print(all_data_stsb_cross_data[2][0])
    print(all_data_stsb_cross_data[3][0])

    all_data_xnli_cross_data = add_cross_data(all_xnli_data, [[0, 1], [1, 0]])
    print("### last example from xnli with crossing")
    print(all_data_xnli_cross_data[0][-1])
    print(all_data_xnli_cross_data[1][-1])
    print(all_data_xnli_cross_data[2][-1])
    print(all_data_xnli_cross_data[3][-1])
