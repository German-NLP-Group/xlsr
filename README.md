# XLSR - Cross-Lingual Sentence Representations
Cross-Lingual Sentence Representations

This is ongoing research. If you have ideas or want to drop a comment just
[open up an issue](https://github.com/German-NLP-Group/xlsr/issues/new) here on GitHub.

## Data
- stsb from [STSb Multi MT](https://github.com/PhilipMay/stsb-multi-mt)
  - 5749 test samples per language
  - 1500 dev samples per language
- XNLI 1.0 from https://github.com/facebookresearch/XNLI
  - 2490 dev samples per language
  - 5010 test samples per language

## Preprocessing
- the stsb label are `>= 0.0` and `<= 5.0` we devide them by 5 so they are `>= 0.0` and `<= 1.0`
- we have to convert the XNLI label to float numberts so they match those of stsb
  - contradiction to -1.0
  - entailment to 1.0
  - neutral to 0.0

My using the contradiction samples from XNLI we also get negative values. Our hypothesis is that this benefits the model.

## Hyperparameter Search
- we do a 10 fold cross validation over concatenation of stsb-train and stsb-dev
- during cross validaton we concatate all XNLI data (dev and test) to the train data of each fold

## Variations of the Experiment
Things that can be tested:
- reduce label for entailment to something `< 1.0`
- only use contradiction from XNLI and not entailment and neutral
- test with one language (de or en) and a mix of both
- test with 3 languages
- test with other base models from here https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained_models.md#multi-lingual-models
