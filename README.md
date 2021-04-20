# pytorch_beam_search
A simple decoding library for models in PyTorch

This library implements beam search, greedy search and sampling in a fully vectorized way in PyTorch. This is specially useful for NLP tasks, but can also be 
used for anything that requires sequence generation from a sequence model.

The two main classes are found in the main folder:
* **seq2seq** in *seq2seq.py* implements the generation methods for models that are composed of an encoder and a decoder, like the Transformer.
* **autoregressive** in *autoregressive.py* implements the generation methods for models that only have a decoder, like GPT.

In the **examples** folder, you can find several task that show how to use the classes.
