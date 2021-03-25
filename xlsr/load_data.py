
import os
import csv

import pandas as pd


BASE_DATA_PATH = "./data"
BASE_STSB_FILENAME = "stsb-{}-{}.csv"
BASE_XNLI_FILENAME = "xnli-1.0-all-{}.csv"


def load_stsb_data(language, split):
    """Load language file from split.

    Scores are normalized between 0 and 1.
    Result is a list of dict.
    """
    result = []
    with open(
        os.path.join(BASE_DATA_PATH, BASE_STSB_FILENAME.format(language, split)),
        newline="",
        encoding="utf-8",
    ) as csvfile:
        csv_dict_reader = csv.DictReader(
            csvfile,
            dialect="excel",
            fieldnames=["sentence1", "sentence2", "similarity_score"],
        )
        for row in csv_dict_reader:
            assert "sentence1" in row
            assert isinstance(row["sentence1"], str)
            assert len(row["sentence1"].strip()) > 0
            assert len(row["sentence1"].strip()) == len(row["sentence1"])
            assert "sentence2" in row
            assert isinstance(row["sentence2"], str)
            assert len(row["sentence2"].strip()) > 0
            assert len(row["sentence2"].strip()) == len(row["sentence2"])
            assert "similarity_score" in row
            assert isinstance(row["similarity_score"], str)
            assert len(row["similarity_score"].strip()) > 0

            # convert similarity_score from str to float and normalize
            row["similarity_score"] = float(row["similarity_score"]) / 5

            # do more asserts
            assert row["similarity_score"] >= 0.0
            assert row["similarity_score"] <= 1.0

            result.append(row)
    return result


def load_stsb_train_dev_data(language):
    train_data = load_stsb_data(language, "train")
    assert len(train_data) == 5749, len(train_data)
    dev_data = load_stsb_data(language, "dev")
    assert len(dev_data) == 1500, len(dev_data)
    return train_data + dev_data


def load_xnli_data(language, label_map):
    """Load xnli data.

    Args:
        language ([type]): The language to load
        label_map (dict): Map gold_label to similarity_score.

    Returns:
        list of dict: sentence1, sentence2, similarity_scores
    """
    df = pd.read_csv(
        os.path.join(BASE_DATA_PATH, BASE_XNLI_FILENAME.format(language)),
        low_memory=False,
        )
    df.drop("language", axis=1, inplace=True)
    result = df.to_dict('records')
    for r in result:
        assert r["gold_label"] in label_map
        r["similarity_score"] = label_map[r["gold_label"]]
        del r["gold_label"]
    return result


if __name__ == "__main__":
    stsb_data_de = load_stsb_train_dev_data("de")
    assert len(stsb_data_de) == 5749 + 1500

    label_map = {
        "contradiction": -1.0,
        "entailment": 1.0,
        "neutral": 0.0,
    }

    xnli_data_de = load_xnli_data("de", label_map)
    assert len(xnli_data_de) == 7500
