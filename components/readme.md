# Three main components used for the project

## aspect-sentiment

probably does something

## ner

Uses the pickle file in data folder with [pretrained english model](https://github.com/glample/tagger) to produce text file in format in `aspects.txt`:
```
Sentence containing aspect with aspect replaced with aspect_term
Actual aspect term
Positive/Negative (N/A for unseen data)
```

## summarizer

takes input from [cnn-dailymail](https://github.com/becxer/cnn-dailymail/) and produces pickle file with lists of `ids`, `stories`, `summaries`, `summaries_3sent`. The `summaries` being produced by [pointer-generator model](https://github.com/becxer/pointer-generator/)


