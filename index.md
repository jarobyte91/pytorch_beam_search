# PyTorch Decoding

This library implements beam search, greedy search and sampling in a fully vectorized way in PyTorch. This is specially useful for NLP tasks, but can also be used for anything that requires sequence generation from a sequence model.

## Usage

## Features

* **Autoregressive** in *autoregressive.py* implements the generation methods for unsupervised models that learn to predict the next token in a sequence. The implementation features an LSTM as a simple baseline/sanity check and a Transformer Encoder similar to [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) for more advanced tasks.
* **Seq2Seq** in *seq2seq.py* implements the decoding methods for supervised models with a encoder-decoder architecture that learn how to map sequences to sequences. The implementation features a seq2seq unidirectional LSTM model similar to the one in [Cho et al., 2014](https://arxiv.org/abs/1406.1078) as a simple baseline/sanity check and a full Transformer for more advanced tasks.

## Installation

Install pytorch_decoding by running:

    install project

## Contribute

- Issue Tracker: github.com/$project/$project/issues
- Source Code: github.com/$project/$project

## Support

If you are having issues, please let us know.
We have a mailing list located at: project@google-groups.com

## License

The project is licensed under the BSD license.
