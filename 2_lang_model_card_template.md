---
language:
- _???_
- _???_
license: mit
tags:
- sentence_embedding
- search
- pytorch
- xlm-roberta
- roberta
- xlm-r-distilroberta-base-paraphrase-v1
- paraphrase
datasets:
- stsb_multi_mt
metrics:
- Spearman’s rank correlation
- cosine similarity
---

# Cross _???_ & _???_ RoBERTa for Sentence Embeddings
This model is intended to [compute sentence (text) embeddings](https://www.sbert.net/examples/applications/computing-embeddings/README.html) for _???_ and _???_ text. These embeddings can then be compared with [cosine-similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to find sentences with a similar semantic meaning. For example this can be useful for [semantic textual similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html), [semantic search](https://www.sbert.net/examples/applications/semantic-search/README.html), or [paraphrase mining](https://www.sbert.net/examples/applications/paraphrase-mining/README.html). To do this you have to use the [Sentence Transformers Python framework](https://github.com/UKPLab/sentence-transformers).

The speciality of this model is that it also works cross-lingually. Regardless of the language, the sentences are translated into very similar vectors according to their semantics. This means that you can, for example, enter a search in one language and find results according to the semantics in both languages. We have found empirically that using a xlm model and _multilingual finetuning with language-crossing_ increases performance compared to using just one language.

> Sentence-BERT (SBERT) is a  modification  of  the  pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. This reduces the effort for finding the most similar pair from 65hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT.

Source: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

This model is fine-tuned from [Philip May](https://may.la/) and open-sourced by [T-Systems-onsite](https://www.t-systems-onsite.de/). Special thanks to [Nils Reimers](https://www.nils-reimers.de/) for your awesome open-source work, the Sentence Transformers, the models and your help on GitHub. The complete training code and more experiments have been released here: [German-NLP-Group/xlsr](https://github.com/German-NLP-Group/xlsr)

## How to use
To use this model install and use the `sentence-transformers` package (see here: <https://github.com/UKPLab/sentence-transformers>).

## Training
The base model is [xlm-roberta-base](https://huggingface.co/xlm-roberta-base). This model has been further trained by [Nils Reimers](https://www.nils-reimers.de/) on a large scale paraphrase dataset for 50+ languages. [Nils Reimers](https://www.nils-reimers.de/) about this [on GitHub](https://github.com/UKPLab/sentence-transformers/issues/509#issuecomment-712243280):

>A paper is upcoming for the paraphrase models.
>
>These models were trained on various datasets with Millions of examples for paraphrases, mainly derived from Wikipedia edit logs, paraphrases mined from Wikipedia and SimpleWiki, paraphrases from news reports, AllNLI-entailment pairs with in-batch-negative loss etc.
>
>In internal tests, they perform much better than the NLI+STSb models as they have see more and broader type of training data. NLI+STSb has the issue that they are rather narrow in their domain and do not contain any domain specific words / sentences (like from chemistry, computer science, math etc.). The paraphrase models has seen plenty of sentences from various domains.
>
>More details with the setup, all the datasets, and a wider evaluation will follow soon.

The resulting model called `xlm-r-distilroberta-base-paraphrase-v1` has been released here: <https://github.com/UKPLab/sentence-transformers/releases/tag/v0.3.8>

Building on this cross language model we fine-tuned it for two languages on the [stsb_multi_mt](https://huggingface.co/datasets/stsb_multi_mt) dataset. Additionally to the training samples from the two laguages we generated crossed samples. We call this _multilingual finetuning with language-crossing_. It doubled the traing-datasize and tests show that it further improves performance.

We did an automatic hyperparameter search with [Optuna](https://github.com/optuna/optuna). Using 10-fold crossvalidation on the deepl.com test and dev dataset we found the following best hyperparameters:
- batch_size = 8
- num_epochs = 2
- lr = 1.026343323298136e-05,
- eps = 4.462251033010287e-06
- weight_decay = 0.04794438776350409
- warmup_steps_proportion = 0.1609010732760181

The final model was trained with these hyperparameters on the combination of the train and dev datasets from both languages and the crossings of them. The testset was left for testing.

# Test Set Evaluation Results
The evaluation has been done on language one, language two, the crossing of both languages and everything. As the metric for evaluation we use the Spearman’s rank correlation between the cosine-similarity of the sentence embeddings and STSbenchmark labels.
- Spearman correlation _???_: _???_
- Spearman correlation _???_: _???_
- Spearman correlation both languages crossed: _???_
- Spearman correlation of everything: _???_

## License - The MIT License
Copyright 2021 Philip May

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
