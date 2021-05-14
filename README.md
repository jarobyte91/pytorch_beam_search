# pytorch_decoding

A simple decoding library for models in PyTorch

This library implements beam search, greedy search and sampling in a fully vectorized way in PyTorch. This is specially useful for NLP tasks, but can also be used for anything that requires sequence generation from a sequence model.

The two main classes are found in the main folder:

* **Autoregressive** in *autoregressive.py* implements the generation methods for unsupervised models that learn to predict the next token in a sequence. The implementation features an LSTM as a simple baseline/sanity check and a Transformer Encoder similar to [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) for more advanced tasks.
* **Seq2Seq** in *seq2seq.py* implements the decoding methods for supervised models with a encoder-decoder architecture that learn how to map sequences to sequences. The implementation features a seq2seq unidirectional LSTM model similar to the one in [Cho et al., 2014](https://arxiv.org/abs/1406.1078) as a simple baseline/sanity check and a full Transformer for more advanced tasks.

Both are extensions of **torch.nn.Module**, so they can be extended by writing the **__init__** and **forward** methods as is usual in PyTorch.
In the **examples** folder, you can find several tasks with notebooks that show how to use the classes.

The classes have the following methods:

* **greedy_search** implements the most basic decoding method, picking the token with maximum likelihood at every step.
* **sample** implements sampling from a sequence model, using the learned distribution at every step to build the output token by token.
* **beam_search** implements beam search, a form of pruned breadth-first search that expands only the best candidates at every step, and usually outperforms greedy search.
* **fit** is a generic fitting method that uses Adam and Cross Entropy to train the models.
* **evaluate** is useful to compute loss and step-accuracy for development and test sets.
* **predict** is a generic sequence producing method that unifies the decoding into a single API.
* **text2tensor** and **tensor2text** are utility methods to convert back and forth between lists of strings and tensors using the vocabularies of the models.
* **save_architecture** is a utility method to save all the hyper-parameters into a pickled Python dict for easy loading.

All these methods include a **progress_bar** parameter that allows to track progress in terminal and notebook environments.

Each file also includes a special **load_architecture** utility function that allows easy loading the specifications of the model.
