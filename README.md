Evidence graphs for parsing argumentation structure
===================================================

[![Build Status](https://travis-ci.org/peldszus/evidencegraph.svg?branch=master)](https://travis-ci.org/peldszus/evidencegraph)
[![codecov](https://codecov.io/gh/peldszus/evidencegraph/branch/master/graph/badge.svg)](https://codecov.io/gh/peldszus/evidencegraph)
[![GitHub](https://img.shields.io/github/license/peldszus/evidencegraph)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## About

This repository holds the code of the Evidence Graph model, a model for parsing the argumentation structure of text.

It basically is a re-implementation of the model presented first in [(1)](#references). Most work was done 2016-2017. It was used in the experiments of [(2)](#references), [(3)](#references) and [(4)](#references).


## Prerequisites

This code runs in Python 3.8. It is recommended to install it in a separate virtual environment. Here are installation instructions for an Ubuntu 18.04 linux:

```sh
# basics
sudo apt install python3.8-dev
# for lxml
sudo apt install libxml2-dev libxslt1-dev
# for matplotlib
sudo apt install libpng-dev libfreetype6-dev
# for graph plotting
sudo apt install graphviz
```


## Setup environment

Install all required python libaries in the environment and download the language models required by the spacy library.

    make install-requirements
    make download-spacy-data-de
    make download-spacy-data-en

Furthermore, several microtext corpora required for the experiments can be downloaded with:

    make download-corpora


## Test

Make sure all the tests pass.

    make test


## Run a minimal experiment

Run a (shortened and simplified) minimal experiment, to see that everything is working:

    env/bin/python src/experiments/run_minimal.py --corpus m112en

You should (see last lines of the output) get an average macro F1 of the *base classifiers* similar to:  
  (cc ~= 0.84, ro ~= 0.75, fu ~= 0.74, at ~= 0.70).

Evaluate the results, which have been written to `data/`:

    env/bin/python src/experiments/eval_minimal.py --corpus m112en

You should (see first lines of the output) get an average macro F1 for the *decoded results* similar to:  
  (cc ~= 0.86, ro ~= 0.77, fu ~= 0.75, at ~= 0.71).


## Replicate published results

Adjust run_minimal.py:
* Remove the line `folds = folds[:5]` in order to run all 50 train/test splits.
* In the experimental conditions, set `optimize` to `True` so that the local model's hyperparameters are optimized.
* In the experimental conditions, set `optimize_weights` to `True` so that the global model's hyperparameters are optimized.

For more details, see the actual experiment definitions in `src/experiments`.


## Reusing / extending components of the library

### Use the same features for a new language

Load a spacy nlp for the desired language and pass it together with a connective lexicon to the TextFeatures.

```python
from evidencegraph.features_text import TextFeatures
from evidencegraph.classifiers import EvidenceGraphClassifier

my_features = TextFeatures(
    nlp=spacy.load("klingon"),
    connectives={}, # add a connective lexicon here
    feature_set=TextFeatures.F_SET_ALL_BUT_VECTORS
)
clf = EvidenceGraphClassifier(
    my_features.feature_function_segments,
    my_features.feature_function_segmentpairs
)
```

### Use a custom base classifier

Derive a custom base classifier class (stick to the interface) and pass this class to the EvidenceGraphClassifier.

```python
from evidencegraph.classifiers import BaseClassifier

class MyBaseClassifier(BaseClassifier):
    # do something different here
    pass

clf = EvidenceGraphClassifier(
    my_features.feature_function_segments,
    my_features.feature_function_segmentpairs,
    base_classifier_class=MyBaseClassifier
)
```

### Load a custom corpus

Simply load a folder containing argument graph xml files into a GraphCorpus.

```python
from evidencegraph.corpus import GraphCorpus

corpus = GraphCorpus()
corpus.load("path/to/my/folder")
texts, trees = corpus.segments_trees()
```


## References

1) [Joint prediction in MST-style discourse parsing for argumentation mining](https://aclweb.org/anthology/D/D15/D15-1110.pdf)  
   Andreas Peldszus, Manfred Stede.  
   In: Proceedings of the 2015 Conference on Empirical Methods in Natural Language  Processing (EMNLP), Portugal, Lisbon, September 2015.

2) [Automatic recognition of argumentation structure in short monological texts](https://publishup.uni-potsdam.de/files/42144/diss_peldszus.pdf)  
   Andreas Peldszus.  
   Ph.D. thesis, Universität Potsdam, 2018.

3) [Comparing decoding mechanisms for parsing argumentative structures](https://content.iospress.com/download/argument-and-computation/aac033?id=argument-and-computation%2Faac033)  
   Stergos Afantenos, Andreas Peldszus, Manfred Stede.  
   In: Argument & Computation, Volume 9, Issue 3, 2018, Pages 177-192.

4) [More or less controlled elicitation of argumentative text: Enlarging a microtext corpus via crowdsourcing](http://www.aclweb.org/anthology/W/W18/W18-5218.pdf)  
   Maria Skeppstedt, Andreas Peldszus, Manfred Stede.  
   In: Proceedings of the 5th Workshop on Argument Mining. EMNLP 2018, Belgium, Brussels, November 2018.
