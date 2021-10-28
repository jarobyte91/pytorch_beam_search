# PyTorch Beam Search

This library implements fully vectorized Beam Search, Greedy Search and sampling for sequence models written in PyTorch. This is specially useful for tasks in Natural Language Processing, but can also be used for anything that requires generating a sequence from a sequence model.

## Usage

### A GPT-like character-level language model
    
    from pytorch_beam_search import autoregressive

    # Create vocabulary and examples
    # tokenize the way you need
    corpus = list("abcdefghijklmnopqrstwxyz ")    
    # len(corpus) == 25
    # An Index object represents a mapping from the vocabulary
    # to integers (indices) to feed into the models
    index = autoregressive.Index(corpus)
    n_gram_size = 17    # 16 with an offset of 1 
    n_grams = [corpus[i:n_gram_size + i] for i in range(len(corpus))[:-n_gram_size + 1]]

    # Create tensor
    X = index.text2tensor(n_grams)
    # X.shape == (n_examples, len_examples) == (25 - 17 + 1 = 9, 17)

    # Create and train the model
    model = autoregressive.TransformerEncoder(index)    # just a PyTorch model
    model.fit(X)    # basic method included

    # Generate new predictions
    new_examples = ["new first", "new second"]
    X_new = index.text2tensor(new_examples)
    loss, error_rate = model.evaluate(X_new)    # basic method included
    predictions, log_probabilities = autoregressive.beam_search(model, X_new)
    # every element in predictions is the list of candidates for each example
    output = [index.tensor2text(p) for p in predictions]
    output

### A Transformer character sequence-to-sequence model

    from pytorch_beam_search import seq2seq

    # Create vocabularies
    # Tokenize the way you need
    source = [list("abcdefghijkl"), list("mnopqrstwxyz")]
    target = [list("ABCDEFGHIJKL"), list("MNOPQRSTWXYZ")]
    # An Index object represents a mapping from the vocabulary
    # to integers (indices) to feed into the models
    source_index = seq2seq.Index(source)
    target_index = seq2seq.Index(target)

    # Create tensors
    X = source_index.text2tensor(source)
    Y = target_index.text2tensor(target)
    # X.shape == (n_source_examples, len_source_examples) == (2, 11)
    # Y.shape == (n_target_examples, len_target_examples) == (2, 12)

    # Create and train the model
    model = seq2seq.Transformer(source_index, target_index)    # just a PyTorch model
    model.fit(X, Y, epochs = 100)    # basic method included

    # Generate new predictions
    new_source = [list("new first in"), list("new second in")]
    new_target = [list("new first out"), list("new second out")]
    X_new = source_index.text2tensor(new_source)
    Y_new = target_index.text2tensor(new_target)
    loss, error_rate = model.evaluate(X_new, Y_new)    # basic method included
    predictions, log_probabilities = seq2seq.beam_search(model, X_new) 
    output = [target_index.tensor2text(p) for p in predictions]
    output
    
## Features

### Algorithms

- The **greedy_search** function implements Greedy Search, which simply picks the most likely token at every step. This is the fastest and simplest algorithm, but can work well if the model is properly trained.
- The **sample** function implements sampling from a sequence model, using the learned distribution at every step to build the output token by token. This is very useful to inspect what the model learned.
- The **beam_search** function implements Beam Search, a form of pruned Breadth-First Search that expands a fixed number of the best candidates at every step. This is the slowest algorithm, but usually outperforms Greedy Search.

### Models

- The **autoregressive** module implements the search algorithms and some architectures for unsupervised models that learn to predict the next token in a sequence.
  - **LSTM** is a simple baseline/sanity check.
  - **TransformerEncoder** is a [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)-like model for state-of-the-art performance.
- The **seq2seq** module implements the search algorithms and some architectures for supervised encoder-decoder models that learn how to map sequences to sequences.  
  - **LSTM** is a sequence-to-sequence unidirectional LSTM model similar to the one in [Cho et al., 2014](https://arxiv.org/pdf/1406.1078.pdf), useful as a simple baseline/sanity check.
  - **ReversingLSTM** is a sequence-to-sequence unidirectional LSTM model that reverses the order of the tokens in the input, similar to the one in [Sutskever et al., 2014](https://arxiv.org/pdf/1409.3215.pdf). A bit more complex than LSTM but gives better performance.
  - **Transformer** is a standard [Transformer](https://arxiv.org/pdf/1706.03762.pdf) model for state-of-the-art performance.


## Installation

    pip install pytorch_beam_search

## Contribute

- [Issue Tracker](https://github.com/jarobyte91/pytorch_beam_search/issues)

## Support

If you are having issues, feel free to contact me at jarobyte91@gmail.com

## License

The project is licensed under the MIT License.

