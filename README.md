# XLSR - Cross-Lingual Sentence Representations
Cross-Lingual Sentence Representations

## Data
- stsb from [STSb Multi MT](https://github.com/PhilipMay/stsb-multi-mt)
  - 5749 test samples per language
  - 1500 dev samples per language
- XNLI 1.0 from https://github.com/facebookresearch/XNLI
  - 2490 dev samples per language
  - 5010 test samples per language

## Hyperparameter Search
- we do a 10 fold cross validation over concatenation of stsb-train and stsb-dev
- during cross validaton we concatate all XNLI data (dev and test) to the train data of each fold
