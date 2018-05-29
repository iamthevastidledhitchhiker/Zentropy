# Three main components used for the project

## aspect-sentiment

Separated into two main components: SVM-based and attention-based. The SVM version can utilize two types of models: support vector classification, and a support vector regression, both using linear kernels. Pre-trained models have been included.

`SVC_SVR_Sentiment.ipynb` contains all of the instructions necessary to classify new sentences.


The attention-based analysis has been included for the sake of completeness, however the code itself does not run in a manner that is particularly helpful, as the manner in which the program processes inputs and produces outputs (found in `attention/analysis/`) was difficult to modify without breaking the program itself. Additionally, due to size constraints, the GloVe embeddings must be downloaded separately. This program uses the `glove.6b.300d.txt` file (approximately 1.04 gb)



## ner

Uses the pickle file in data folder with [pretrained english model](https://github.com/glample/tagger) to produce text file in format in `aspects.txt`:
```
Sentence containing aspect with aspect replaced with aspect_term
Actual aspect term
Positive/Negative (N/A for unseen data)
```

## summarizer

takes input from [cnn-dailymail](https://github.com/becxer/cnn-dailymail/) and produces pickle file with lists of `ids`, `stories`, `summaries`, `summaries_3sent`. The `summaries` being produced by [pointer-generator model](https://github.com/becxer/pointer-generator/)
