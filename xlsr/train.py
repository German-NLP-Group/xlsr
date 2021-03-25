from load_data import load_stsb_train_dev_data, load_xnli_data


def load_all_data(languages, label_map):
    data_per_language = []
    for language in languages:
        stsb_data = load_stsb_train_dev_data(language)
        assert len(stsb_data) == 5749 + 1500
        xnli_data = load_xnli_data(language, label_map)
        assert len(xnli_data) == 7500
        data_per_language.append(stsb_data + xnli_data)
    return data_per_language


if __name__ == "__main__":
    label_map = {
        "contradiction": -1.0,
        "entailment": 1.0,
        "neutral": 0.0,
    }

    all_data = load_all_data(["en", "de"], label_map)
    print(all_data[0][0])
    print(all_data[1][0])
    print("###")
    print(all_data[0][-1])
    print(all_data[1][-1])
