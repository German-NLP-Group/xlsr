# XLSR - Cross-Lingual Sentence Representations
The goal of this project is to provide models that further increase the performance of [T-Systems-onsite/cross-en-de-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer).

This is ongoing research. If you have ideas or want to drop a comment just
[open up an issue](https://github.com/German-NLP-Group/xlsr/issues/new) here on GitHub.

## Contents
- `train_optuna_stsb.py` - hyperparameter search for stsb data
- `train_optuna_stsb_xlni.py` - hyperparameter search for stsb with xlni data
- `train_optuna_stsb_do.py` - hyperparameter search for stsb data with dropout
- `train_models_2_lang.py` - create and save model with two languages

## HPO results

Results for model `xlm-r-distilroberta-base-paraphrase-v1`:

```
best_trial: FrozenTrial(number=46,
values=[0.8837476694198919],
datetime_start=datetime.datetime(2022, 10, 3, 22, 12, 28, 292113), datetime_complete=datetime.datetime(2022, 10, 4, 1, 29, 27, 369305),
params={'train_batch_size': 39, 'num_epochs': 8, 'lr': 1.8020158665633495e-05, 'eps': 8.911500324187447e-06,
'weight_decay': 0.002062478470459487, 'warmup_steps_mul': 0.6571864241925505},
distributions={'train_batch_size': IntDistribution(high=80, log=False, low=4, step=1),
'num_epochs': IntDistribution(high=8, log=False, low=1, step=1),
'lr': FloatDistribution(high=0.00025, log=False, low=2e-06, step=None),
'eps': FloatDistribution(high=0.0001, log=False, low=1e-07, step=None),
'weight_decay': FloatDistribution(high=0.1, log=False, low=0.0005, step=None),
'warmup_steps_mul': FloatDistribution(high=0.7, log=False, low=0.1, step=None)},
user_attrs={'results': '[0.8840569375039711, 0.8840753186972556, 0.8842396846015963, 0.8835176757217545, 0.883729569821923, 0.8830097995752192, 0.882905232860196, 0.88413090489947, 0.8843150617321305, 0.883496508785404]'}, 
system_attrs={},
intermediate_values={0: 0.8840569375039711, 1: 0.8840753186972556, 2: 0.8842396846015963, 3: 0.8835176757217545, 4: 0.883729569821923, 5: 0.8830097995752192, 6: 0.882905232860196, 7: 0.88413090489947, 8: 0.8843150617321305, 9: 0.883496508785404}, 
trial_id=196, state=TrialState.COMPLETE, value=None)
##############
best_trial.params: {'train_batch_size': 39, 'num_epochs': 8, 'lr': 1.8020158665633495e-05, 'eps': 8.911500324187447e-06,
'weight_decay': 0.002062478470459487, 'warmup_steps_mul': 0.6571864241925505}
```

Results for model `paraphrase-multilingual-mpnet-base-v2`:

```
best_trial: FrozenTrial(number=73, 
values=[0.88467176153687], 
datetime_start=datetime.datetime(2022, 10, 4, 4, 29, 52, 297290), 
datetime_complete=datetime.datetime(2022, 10, 4, 5, 36, 34, 592136), 
params={'eps': 5.46546327020518e-05, 'lr': 1.05036252814912e-05, 'num_epochs': 1, 'train_batch_size': 10, 
'warmup_steps_mul': 0.3635907846059213, 'weight_decay': 0.07964970481146776}, 
distributions={'eps': FloatDistribution(high=0.0001, log=False, low=1e-07, step=None), 
'lr': FloatDistribution(high=0.00025, log=False, low=2e-06, step=None), 
'num_epochs': IntDistribution(high=8, log=False, low=1, step=1), 
'train_batch_size': IntDistribution(high=80, log=False, low=4, step=1), 
'warmup_steps_mul': FloatDistribution(high=0.7, log=False, low=0.1, step=None), 
'weight_decay': FloatDistribution(high=0.1, log=False, low=0.0005, step=None)}, 
user_attrs={'results': '[0.8849791104347544, 0.8842818196805877, 0.8857645655013733, 0.8850333656631684, 0.8849685731807815, 0.883991854102697, 0.8854227534484268, 0.883768652785338, 0.8836889495663515, 0.8848179710052217]'}, 
system_attrs={}, 
intermediate_values={0: 0.8849791104347544, 1: 0.8842818196805877, 2: 0.8857645655013733, 3: 0.8850333656631684, 4: 0.8849685731807815, 5: 0.883991854102697, 6: 0.8854227534484268, 7: 0.883768652785338, 8: 0.8836889495663515, 9: 0.8848179710052217}, 
trial_id=91, state=TrialState.COMPLETE, value=None)
##############
best_trial.params: {'eps': 5.46546327020518e-05, 'lr': 1.05036252814912e-05, 'num_epochs': 1, 'train_batch_size': 10, 
'warmup_steps_mul': 0.3635907846059213, 'weight_decay': 0.07964970481146776}
```

## Models
- [T-Systems-onsite/cross-en-de-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-es-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-es-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-fr-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-fr-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-it-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-it-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-nl-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-nl-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-pl-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-pl-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-pt-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-pt-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-ru-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-ru-roberta-sentence-transformer)
- [T-Systems-onsite/cross-de-zh-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-de-zh-roberta-sentence-transformer)
- [T-Systems-onsite/cross-en-es-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-es-roberta-sentence-transformer)
- [T-Systems-onsite/cross-en-fr-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-fr-roberta-sentence-transformer)
- [T-Systems-onsite/cross-en-it-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-it-roberta-sentence-transformer)
- [T-Systems-onsite/cross-en-nl-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-nl-roberta-sentence-transformer)
- [T-Systems-onsite/cross-en-pl-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-pl-roberta-sentence-transformer)
- [T-Systems-onsite/cross-en-pt-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-pt-roberta-sentence-transformer)
- [T-Systems-onsite/cross-en-ru-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-ru-roberta-sentence-transformer)
- [T-Systems-onsite/cross-en-zh-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-zh-roberta-sentence-transformer)

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
