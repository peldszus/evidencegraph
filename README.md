EvidenceGraphs for parsing argumentation structure
==================================================

## Prerequisites

This code runs in Python 2.7. It is recommended to install it in a separate virtual environment. Here are installation instructions for an Ubuntu 18.04 linux:

    # basics
    sudo apt install virtualenv python2.7-dev
    # for lxml
    sudo apt install libxml2-dev libxslt1-dev
    # for matplotlib
    sudo apt install libpng-dev libfreetype6-dev
    # for graph plotting
    sudo apt install graphviz


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

You should (see last lines of the output) get an average macro F1 of the *base classifiers* similar to: (cc ~= 0.84, ro ~= 0.75, fu ~= 0.74, at ~= 0.70).

Evaluate the results, which have been written to `data/`:

    env/bin/python src/experiments/eval_minimal.py --corpus m112en

You should (see first lines of the output) get an average macro F1 for the *decoded results* similar to: (cc ~= 0.86, ro ~= 0.77, fu ~= 0.75, at ~= 0.71).


## Replicate published results

Adjust run_minimal.py:
* Remove the line `folds = folds[:5]` in order to run all 50 train/test splits.
* In the experimental conditions, set `optimize` to `True` so that the local model's hyperparameters are optimized.
* In the experimental conditions, set `optimize_weights` to `True` so that the global model's hyperparameters are optimized.
