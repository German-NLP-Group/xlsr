# XLSR - Cross-Lingual Sentence Representations
The goal of this project is to provide models that further increase the performance of [T-Systems-onsite/cross-en-de-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer).

This is ongoing research. If you have ideas or want to drop a comment just
[open up an issue](https://github.com/German-NLP-Group/xlsr/issues/new) here on GitHub.

## Contents
- `train_optuna_stsb.py` - hyperparameter search for stsb data
- `train_optuna_stsb_xlni.py` - hyperparameter search for stsb with xlni data
- `train_optuna_stsb_do.py` - hyperparameter search for stsb data with dropout
- `train_models_2_lang.py` - create and save model with two languages

## Models
- [T-Systems-onsite/cross-en-de-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-es-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-es-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-fr-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-fr-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-it-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-it-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-nl-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-nl-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-pl-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-pl-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-pt-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-pt-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-ru-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-ru-roberta-sentence-transformer)

## Data
- stsb from [STSb Multi MT](https://github.com/PhilipMay/stsb-multi-mt)
  - 5749 test samples per language
  - 1500 dev samples per language
  - the labels are `>= 0.0` and `<= 5.0` we devide them by 5 so they are `>= 0.0` and `<= 1.0`
- XNLI 1.0 from https://github.com/facebookresearch/XNLI
  - 2490 dev samples per language
  - 5010 test samples per language
  - we have to convert the XNLI label to float numberts so they match those of stsb
    - contradiction to -1.0
    - entailment to 1.0
    - neutral to 0.0

## Hyperparameter Search
- we do a 10 fold cross validation over concatenation of stsb-train and stsb-dev (but not XNLI)
- we use Optuna
- if we use XNLI data (dev and test) we concatenate it to the train data of each fold

## Ideas and Variations of the Experiment
Things that can be tested:
- reduce label for entailment to something `< 1.0`
- only use contradiction from XNLI and not entailment and neutral
- test with one language (de or en) and a mix of both
- test with 3 languages
- test with other base models from here https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained_models.md#multi-lingual-models

## Results and Findings
- adding all XNLI data to the train data (stsb) had no positive effect
- adding just XNLI entailment and neutral data to the train data (stsb) had no positive effect
- adding just XNLI entailment to the train data (stsb) had no positive effect
- we tried to add dropout for more regularization - since [sentence-transformers](https://github.com/UKPLab/sentence-transformers/) do not have a prediction head we had to modify the language model itself
  - see discussion here: https://github.com/UKPLab/sentence-transformers/issues/846 
  - we tried to change `attention_probs_dropout_prob` and `hidden_dropout_prob` by adding them as hyperparameters to Optuna
  - `model._modules['0'].auto_model.base_model.config.attention_probs_dropout_prob = attention_probs_dropout_prob`
  - `model._modules["0"].auto_model.base_model.config.hidden_dropout_prob = hidden_dropout_prob`
  - this had no positive effect
  - just changing `hidden_dropout_prob` also had no positive effect
