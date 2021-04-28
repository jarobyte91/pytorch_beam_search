# Language Modelling

This is the classical NLP task of training a probabilistic sequence model to assign probabilities to strings of characters.

The task is implemented as follows:

* Two implementations of the task with autoregressive models:
  + An unidirectional LSTM that serves as a sanity check for the task.
  + A GPT-like Transformer decoder.
* Two implementations of the task with Seq2Seq models:
  + A Seq2Seq model composed of two unidirectional LSTM as in the original Seq2Seq model proposed in (Sutskever et al., 2014)[https://arxiv.org/abs/1409.3215] that serves as a sanity check for the task.
  + An implementation with a full Transformer.
  
The folder includes the following:
* The **architectures** file contains the code of the architectures and can be imported directly as is.
* The **data** folder includes an E-Book in .txt format from Project Gutenberg to train the models.
* Each **notebook** shows a simple pipeline about how to use the architecture. 
* The **checkpoints** folder includes one checkpoint for every notebook to test the architectures more quickly.