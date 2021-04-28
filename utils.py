import pickle
import re
import torch
import torch.nn as nn
import seq2seq

def save_architecture(model, path):
    with open(path, "wb") as file:
        pickle.dump(model.architecture, file)

def load_architecture(path):
    with open(path, "rb") as file:
        architecture = pickle.load(file)
    if architecture["model"] == "Seq2Seq RNN":
        model = seq2seq.Seq2SeqRNN(in_vocabulary = architecture["in_vocabulary"], 
                           out_vocabulary = architecture["out_vocabulary"], 
                           in_embedding_dimension = architecture["in_embedding_dimension"],
                           out_embedding_dimension = architecture["out_embedding_dimension"],
                           encoder_hidden_units = architecture["encoder_hidden_units"], 
                           encoder_layers = architecture["encoder_layers"],
                           decoder_hidden_units = architecture["decoder_hidden_units"],
                           decoder_layers = architecture["decoder_layers"],
                           dropout = architecture["dropout"])
    elif architecture["model"] == "Transformer":
        model = seq2seq.Transformer(in_vocabulary = architecture["in_vocabulary"], 
                            out_vocabulary = architecture["out_vocabulary"], 
                            embedding_dimension = architecture["embedding_dimension"], 
                            feedforward_dimension = architecture["feedforward_dimension"], 
                            encoder_layers = architecture["encoder_layers"],
                            decoder_layers = architecture["decoder_layers"],
                            dropout = architecture["dropout"],
                            attention_heads = architecture["attention_heads"])
    else:
        raise Exception(f"Unknown architecture: {architecture['model']}")
    return model
    
def tensor2text(X, vocabulary, separator = "", end = "<END>"):
    return [re.sub(end + ".*", end, separator.join([vocabulary[i] for i in l])) for l in X.tolist()]

def text2tensor(strings, vocabulary):
    return nn.utils.rnn.pad_sequence([torch.tensor([vocabulary[c] for c in l]) 
                                      for l in strings], 
                                     batch_first = True)