# AMCGRec
This is the implementation of our paper "Attentional Meta-path Contrastive Graph Convolutional Networks for Knowledge Concept Recommendation."
This work is inspired by and built on top of the [EDM2021](https://educationaldatamining.org/edm2021/) parper ["Recommending Knowledge Concepts on MOOC Platforms with Meta-path-based Representation Learning."](https://parklize.github.io/publications/EDM2021.pdf)
## Requirement:
Python 3.6

Tensorflow-gpu 1.13.1


## Data set:
Our AMCGRec uses the same dataset as ["Recommending Knowledge Concepts on MOOC Platforms with Meta-path-based Representation Learning."](https://parklize.github.io/publications/EDM2021.pdf) You can get all training data from `./data` in [code](https://github.com/parklize/kgc-rec).

## Quick Start:
1. Download the training data in `./data` from [MOOCIR](https://github.com/parklize/kgc-rec) and put them in the `./data` folder.
2. Run `generate_test_data.py` to get the `test_negative.npy` and put it in the `./data` folder.
3. Run `train.py`.

