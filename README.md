# Zentropy

Produces polarity of detected organisations from news articles using a pipelined set of 3 neural nets:

##### 1. Summarization

First summarizes the news article with 3 different types of summarization (no coverage, small coverage, full coverage). Based on [pointer-generator model](https://github.com/becxer/pointer-generator/) with [cnn-dailymail training data](https://github.com/becxer/cnn-dailymail/).

*Requirements:* TensorFlow 1.2.X, Python3.X, Vocabulary File

##### 2. Named Entity Analysis

Detects named entities from summarizations using [tagger model](https://github.com/glample/tagger) with pretrained English model. Only outputs organisation tags.

*Requirements:* Python2.7, NLTK (Python2.7), Theano (Python2.7), Python 3.X

##### 3. Aspect Based Polarity

Extracts polarity of organisations from tagger model.

*Requirements:* TensorFlow 1.7, Python3.X

## Installation

To install the requiements:
*insert pip install instructions*
`pip install tensorflow`;
`pip2.7 install nltk`;
`pip2.7 install theano`

## Usage

For example: Model takes input from data folder in newline seperated form:
```
Article Headline?
Article?
```
Then run `python Zentropy.py`.
