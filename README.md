# Pytorch Beam Search

This library implements fully vectorized Beam Search, Greedy Search and sampling for sequence models written in PyTorch. This is specially useful for tasks in Natural Language Processing, but can also be used for anything that requires generating a sequence from a sequence model.

## Usage

### A GPT-like character-level language model
    
    from pytorch_beam_search import autoregressive
    
    # Create vocabulary and examples
    corpus = list("This is a very long string")    # tokenize the way you need
    vocabulary = autoregressive.Vocabulary(corpus)
    n_gram_size = 17    # 16 with an offset of 1 
    n_grams = [text[i:n_gram_size + i] for i in range(len(text))[:-n_gram_size]]
    
    # Create tensors
    T = vocabulary.text2tensor(n_grams)
    X, Y = T[:, :-1], T[:, 1:]    # examples to predict the next token in a sequence
    
    # Create and train the model
    model = autoregressive.TransformerEncoder(vocabulary)    # just a standard PyTorch sequence model
    model.fit(X, Y)    # basic method included, train however you see fit
    
    # Generate new predictions
    new_examples = ["First", "Second"]
    X_new = vocabulary.text2tensor(new_examples)
    predictions, log_probabilities = autoregressive.beam_search(model, X_new) 
    output_text = vocabulary.tensor2text(predictions)

### A Transformer character sequence-to-sequence model

    from pytorch_beam_search import seq2seq
    
    # Create vocabularies
    source = [list("first in"), list("second in")]    # tokenize the way you need
    target = [list("first out"), list("second out")]    # tokenize the way you need
    in_vocabulary = seq2seq.Vocabulary(source)
    out_vocabulary = seq2seq.Vocabulary(target)
    
    # Create tensors
    X = in_vocabulary.text2tensor(source)
    Y = out_vocabulary.text2tensor(target)
    
    # Create and train the model
    model = seq2seq.Transformer(in_vocabulary, out_vocabulary)    # just a standard PyTorch sequence model
    model.fit(X, Y)    # basic method included, train however you see fit
    
    # Generate new predictions
    new_examples = ["first new", "second new"]
    X_new = in_vocabulary.text2tensor(new_examples)
    predictions, log_probabilities = seq2seq.beam_search(model, X_new) 
    output_text = out_vocabulary.tensor2text(predictions)
    
You can find tutorials for some use cases in the **tutorials** folder.

## Features

### Algorithms

- **greedy_search** implements Greedy Search, which simply picks the most likely token at every step. This is the fastest and simplest algorithm, but can work well if the model is properly trained.
- **sample** implements sampling from a sequence model, using the learned distribution at every step to build the output token by token. This is very useful to inspect what the model learned.
- **beam_search** implements Beam Search, a form of pruned Breadth-First Search that expands a fixed number of the best candidates at every step. This is the slowest algorithm, but usually outperforms Greedy Search.

### Models

- The **autoregressive** module implements the search algorithms and some architectures for unsupervised models that learn to predict the next token in a sequence.
  - LSTM is a simple baseline/sanity check.
  - TransformerEncoder is a [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)-like model for state-of-the-art performance.
- The **seq2seq** module implements the search algorithms and some architectures for supervised encoder-decoder models that learn how to map sequences to sequences.  
  - LSTM is a sequence-to-sequence unidirectional LSTM model similar to the one in [Cho et al., 2014](https://arxiv.org/abs/1406.1078), useful as a simple baseline/sanity check.
  - Transformer is a standard [Transformer](https://arxiv.org/pdf/1706.03762.pdf) model for state-of-the-art performance.


## Installation

    pip install pytorch_beam_search

## Contribute

- [Issue Tracker](https://github.com/jarobyte91/pytorch_beam_search/issues)

## Support

If you are having issues, feel free to contact me at jarobyte91@gmail.com

## License

The project is licensed under the GPL-3.0 License.

