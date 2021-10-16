# Pytorch Beam Search

This library implements beam search, greedy search and sampling in a fully vectorized way in PyTorch. This is specially useful for NLP tasks, but can also be used for anything that requires generating a sequence from a sequence model.

## Usage

    from pytorch_beam_search.autoregressive import AutoRegressive

You can find detailed tutorials in the **tutorials** folder.

## Features

### Algorithms

+ **greedy_search** implements the most basic search method, picking the token with maximum likelihood at every step. 
+ **sample** implements sampling from a sequence model, using the learned distribution at every step to build the output token by token.
+ **beam_search** implements beam search, a form of pruned breadth-first search that expands only the best candidates at every step.

### Architectures

+ **Forward** in *forward.py* implements the search algorithms for unsupervised models that learn to predict the next token in a sequence.
  + LSTM  as a simple baseline/sanity check 
  + TransformerEncoder similar to [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) for more advanced tasks.
+ **Seq2Seq** in *seq2seq.py* implements the decoding methods for supervised models with a encoder-decoder architecture that learn how to map sequences to sequences. The implementation features a  and  
  + LSTM  seq2seq unidirectional LSTM model similar to the one in [Cho et al., 2014](https://arxiv.org/abs/1406.1078) as a simple baseline/sanity check
  + Transformer is an a full Transformer for more advanced tasks.


## Installation

Install pytorch_decoding by running:

    install project

## Contribute

- [Issue Tracker](https://github.com/jarobyte91/pytorch_beam_search/issues)
- [Source Code](https://github.com/jarobyte91/pytorch_beam_search)

## Support

If you are having issues, please let me know. You can contact me at jarobyte91@gmail.com

## License

The project is licensed under the BSD license.

