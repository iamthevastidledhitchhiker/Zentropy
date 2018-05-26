# Zentropy

Produces polarity of detected organisations from news articles using a pipelined set of 3 neural nets:

## 1. Summarization

First summarizes the news article with 3 different types of summarization (no coverage, small coverage, full coverage). Based on [pointer-generator model](https://github.com/becxer/pointer-generator/) with [cnn-dailymail training data](https://github.com/becxer/cnn-dailymail/).

*Software Requirements:* TensorFlow 1.2.X, Python3.X.  

*Other requirements:* To train the network from scratch a [CNN/Daily Mail](https://github.com/becxer/cnn-dailymail/) data set is required. It is also possible to obtain a [pre-trained model for TensorFlow 1.2.1](https://drive.google.com/file/d/0B7pQmm-OfDv7ZUhHZm9ZWEZidDg/view), this model does not have coverage trained.

## 2. Named Entity Analysis

Detects named entities from summarizations using [tagger model](https://github.com/glample/tagger) with pretrained English model. Only outputs organisation tags.

*Requirements:* Python2.7, NLTK (Python2.7), Theano (Python2.7), Python 3.X

##3. Aspect Based Polarity

Extracts polarity of organisations from tagger model.

*Requirements:* TensorFlow 1.7, Python3.X

## Installation

To install and run Summarization + Named Entity Recognition you need to do the following:

* Make sure that you have installed software packages for creating Python virtual environment
* Create and activate a Python virtual environment -
	* Using Anaconda: `conda create -n venv python=3.5; source activate venv`
	* Using virtualenv: `cd Zentropy; mkdir venv; virtualenv venv`
* Install Python 3 requirements: `pip3 install -r py3_summarization_requirements.txt` (for GPU usage) or `pip3 install -r py3_summarization_requirements_cpu.txt` if you don't have a GPU
* Install Python 2 requirements: `pip2.7 install -r py2_ner_requirements.txt`
* Download and extract [Stanford CoreNLP Java library](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip) and modify file `stanford_nlp_init.env` to point to the version you downloaded. After that run `source stanford_nlp_init.env`. This library is used for tokenizing the articles into the format that the summarization model requires.
* Project depends on [modified summarization code](https://github.com/arpol/pointer-generator), [CNN/Daily Mail data repository](https://github.com/becxer/cnn-dailymail), [NER model](https://github.com/glample/tagger/) as submodules. To make sure the dependencies are loaded, run the following commands: `git submodule init; git submodule update --recursive`

## Usage

### Training and using individual models

* To train the summarization model from scratch follow the [instructions here](https://github.com/becxer/pointer-generator/)

### Using Summarizationa and NER models together

* The best way to see the Summarization and NER models in action is to run `pipeline_demo.py` (making sure that you are inside the virtual environment containing Tensorflow 1.2.1)


For example: Model takes input from data folder in newline seperated form:
```
Article Headline?
Article?
```
Then run `python Zentropy.py`.
