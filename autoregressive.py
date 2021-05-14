import torch
import torch.nn as nn
import torch.utils.data as tud
from tqdm.notebook import tqdm
from pprint import pprint
import numpy as np
from timeit import default_timer as timer
import pandas as pd
import re
import pickle

class Autoregressive(nn.Module):        
    def greedy_search(self, 
                      X, 
                      max_predictions = 20,
                      verbose = False):
        with torch.no_grad():
            probabilities = torch.zeros(X.shape[0]).to(next(self.parameters()).device)
            iterator = range(max_predictions)
            if verbose:
                iterator = tqdm(iterator)
            for i in iterator:
                next_probabilities = self.forward(X)[:, -1].log_softmax(-1)
                max_next_probabilities, next_chars = next_probabilities.max(-1)
                next_chars = next_chars.unsqueeze(-1)
                X = torch.cat((X, next_chars), axis = 1)
                probabilities += max_next_probabilities
        return X, probabilities

    def sample(self, 
               X, 
               max_predictions = 20,
               temperature = 1,
               verbose = False):
        with torch.no_grad():
            probabilities = torch.zeros(X.shape[0]).to(next(self.parameters()).device)
            iterator = range(max_predictions)
            if verbose:
                iterator = tqdm(iterator)
            for i in iterator:
                next_probabilities = self.forward(X)[:, -1]
                next_probabilities = (next_probabilities / temperature).softmax(1)
                random = torch.rand((next_probabilities.shape[0], 1)).to(next(self.parameters()).device)
                next_chars = ((next_probabilities.cumsum(1) < random).sum(1, keepdims = True))
                probabilities += torch.gather(input = next_probabilities.log(), dim = 1, index = next_chars).squeeze()
                X = torch.cat((X, next_chars), axis = 1)
            return X, probabilities

    def beam_search(self, 
                    X, 
                    max_predictions = 20,
                    beam_width = 5,
                    batch_size = 100, 
                    verbose = 0):
        with torch.no_grad():
            # The next command can be a memory bottleneck, but can be controlled with the batch 
            # size of the predict method.
            next_probabilities = self.forward(X)[:, -1, :]
            vocabulary_size = next_probabilities.shape[-1]
            probabilities, idx = next_probabilities.squeeze().log_softmax(-1)\
            .topk(k = beam_width, axis = -1)
            X = X.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = -2)
            next_chars = idx.reshape(-1, 1)
            X = torch.cat((X, next_chars), axis = -1)
            # This has to be minus one because we already produced a round
            # of predictions before the for loop.
            predictions_iterator = range(max_predictions - 1)
            if verbose > 0:
                predictions_iterator = tqdm(predictions_iterator)
            for i in predictions_iterator:
                dataset = tud.TensorDataset(X)
                loader = tud.DataLoader(dataset, batch_size = batch_size)
                next_probabilities = []
                iterator = iter(loader)
                if verbose > 1:
                    iterator = tqdm(iterator)
                for (x,) in iterator:
                    next_probabilities.append(self.forward(x)[:, -1, :].log_softmax(-1))
                next_probabilities = torch.cat(next_probabilities, axis = 0)
                next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
                probabilities = probabilities.unsqueeze(-1) + next_probabilities
                probabilities = probabilities.flatten(start_dim = 1)
                probabilities, idx = probabilities.topk(k = beam_width, axis = -1)
                next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
                best_candidates = (idx / vocabulary_size).long()
                best_candidates += torch.arange(X.shape[0] // beam_width, device = X.device).unsqueeze(-1) * beam_width
                X = X[best_candidates].flatten(end_dim = -2)
                X = torch.cat((X, next_chars), axis = 1)
            return X.reshape(-1, beam_width, X.shape[-1]), probabilities
        
    def fit(self, 
            X_train,
            X_dev = None,
            batch_size = 100, 
            epochs = 5, 
            learning_rate = 0.0001, 
            verbose = 0, 
            weight_decay = 0, 
            save_path = None):
        assert X_train.shape[1] > 1
        if X_dev is not None:
            dev = True
        else:
            dev = False
        train_dataset = tud.TensorDataset(X_train)
        train_loader = tud.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        criterion = nn.CrossEntropyLoss(ignore_index = 0)
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate, weight_decay = weight_decay)
        performance = []
        start = timer()
        epochs_iterator = range(1, epochs + 1)
        if verbose > 0:
            epochs_iterator = tqdm(epochs_iterator)
        header_1 = "Epoch | Train                "
        header_2 = "      | Loss     | Error Rate"
        rule = "-" * 29
        if dev:
            header_1 += " | Development          "
            header_2 += " | Loss     | Error Rate"
            rule += "-" * 24
        header_1 += " | Training time"
        header_2 += " |"
        rule += "-" * 16
        print(header_1, header_2, rule, sep = "\n")
        for e in epochs_iterator:
            self.train()
            losses = []
            errors = []
            sizes = []
            train_iterator = train_loader
            if verbose > 1:
                train_iterator = tqdm(train_iterator)
            for (x, ) in train_iterator:
                # compute loss and backpropagate
                probabilities = self.forward(x[:, :-1])
                y = x[:, 1:]
                loss = criterion(probabilities.transpose(1, 2), y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # compute accuracy
                predictions = probabilities.argmax(-1)
                batch_errors = (predictions != y)
                # append the results
                losses.append(loss.item())
                errors.append(batch_errors.sum().item())
                sizes.append(batch_errors.numel())
            train_loss = sum(losses) / len(losses)
            train_error_rate = 100 * sum(errors) / sum(sizes)
            t = timer() - start
            status_string = f"{e:>5} | {train_loss:>8.4f} | {train_error_rate:>10.3f}"
            status = {"epoch":e,
                      "train_loss": train_loss,
                      "train_error_rate": train_error_rate}
            if dev:
                dev_loss, dev_error_rate = self.evaluate(X_dev, 
                                                         batch_size = batch_size, 
                                                         verbose = verbose > 2, 
                                                         criterion = criterion)
                status_string += f" | {dev_loss:>8.4f} | {dev_error_rate:>10.3f}"
                status.update({"dev_loss": dev_loss, "dev_error_rate": dev_error_rate})
            status.update({"training_time": t,
                           "learning_rate": learning_rate,
                           "weight_decay": weight_decay})
            performance.append(status)
            if save_path is not None:  
                if (not dev) or (e < 2) or (dev_loss < min([p["dev_loss"] for p in performance[:-1]])):
                    torch.save(self.state_dict(), save_path)
            status_string += f" | {t:>13.2f}"
            print(status_string)
        return pd.concat((pd.DataFrame(performance), 
                          pd.DataFrame([self.architecture for i in performance])), axis = 1)\
               .drop(columns = ["in_vocabulary", "out_vocabulary"])
    
    def print_architecture(self):
        for k in self.architecture.keys():
            if k == "in_vocabulary":
                print(f"Tokens in the in vocabulary: {len(self.architecture[k]):,}")
            elif k == "out_vocabulary":
                print(f"Tokens in the out vocabulary: {len(self.architecture[k]):,}")
            elif k == "parameters":
                print(f"Trainable parameters: {self.architecture[k]:,}")
            else:
                print(f"{k.replace('_', ' ').capitalize()}: {self.architecture[k]}")
        print()
            
    def evaluate(self, X, criterion, batch_size = 100, verbose = False):
        dataset = tud.TensorDataset(X)
        loader = tud.DataLoader(dataset, batch_size = batch_size)
        self.eval()
        losses = []
        errors = []
        sizes = []
        with torch.no_grad():
            iterator = iter(loader)
            if verbose:
                iterator = tqdm(iterator)
            for (x,) in iterator:
                # compute loss
                probabilities = self.forward(x[:, :-1])
                y = x[:, 1:]
                loss = criterion(probabilities.transpose(1, 2), y)
                # compute accuracy
                predictions = probabilities.argmax(-1)
                batch_errors = (predictions != y)
                # append the results
                losses.append(loss.item())
                errors.append(batch_errors.sum().item())
                sizes.append(batch_errors.numel())
            loss = sum(losses) / len(losses)
            error_rate = 100 * sum(errors) / sum(sizes)
        return loss, error_rate   
    
    def predict(self, 
                X, 
                max_predictions = 20, 
                method = "beam_search",
                main_batch_size = 100,
                main_verbose = False,
                **kwargs):
        self.eval()
        dataset = tud.TensorDataset(X.to(next(self.parameters()).device))
        loader = tud.DataLoader(dataset, batch_size = main_batch_size)
        final_indexes = []
        final_probabilities = []
        iterator = iter(loader)
        if main_verbose:
            iterator = tqdm(iterator)
        if method == "beam_search":
            for x in iterator:
                indexes, probabilities = self.beam_search(X = x[0], 
                                                              max_predictions = max_predictions, 
                                                              **kwargs)
                # In this case, we only return the best candidate for each example
                final_indexes.append(indexes[:, 0, :])
                final_probabilities.append(probabilities)
        elif method == "greedy_search":
            for x in iterator:
                indexes, probabilities = self.greedy_search(X = x[0], 
                                                                max_predictions = max_predictions, 
                                                                **kwargs)
                final_indexes.append(indexes)
                final_probabilities.append(probabilities)
        elif method == "sample":
            for x in iterator:
                indexes, probabilities = self.sample(X = x[0], 
                                                         max_predictions = max_predictions, 
                                                         **kwargs)        
                final_indexes.append(indexes)
        else:
            raise ValueError("Decoding method not implemented")
        final_indexes = torch.cat(final_indexes, axis = 0)
        final_probabilities = torch.cat(final_probabilities, axis = 0)
        return final_indexes, final_probabilities
    
    def save_architecture(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.architecture, file)

    def text2tensor(self, strings, vocabulary = None, device = None):
        if vocabulary is None:
            vocabulary = self.architecture["in_vocabulary"]
        if device is None:
            device = next(self.parameters()).device
        return nn.utils.rnn.pad_sequence([torch.tensor([vocabulary[c] for c in l]) 
                                          for l in strings], 
                                         batch_first = True).to(device)

    def tensor2text(self, X, separator = "", vocabulary = None, end = "<END>"):
        if vocabulary is None:
            vocabulary = self.architecture["out_vocabulary"]
        return [re.sub(end + ".*", end, separator.join([vocabulary[i] for i in l])) for l in X.tolist()] 

def load_architecture(path):
    with open(path, "rb") as file:
        architecture = pickle.load(file)
    name = architecture.pop("model")
    architecture.pop("parameters")
    if name == "Seq2Seq RNN":
        model = Seq2SeqRNN(**architecture)
    elif name == "Transformer":
        model = Transformer(**architecture)
    else:
        raise Exception(f"Unknown architecture: {architecture['model']}")
    return model
        
class LSTM(Autoregressive):
    def __init__(self, 
                 in_vocabulary,
                 out_vocabulary, 
                 embedding_dimension = 8,
                 hidden_units = 64, 
                 layers = 2,
                 dropout = 0.0):
        assert len(in_vocabulary) == len(out_vocabulary)
        super().__init__()
        self.embeddings = nn.Embedding(len(in_vocabulary), embedding_dimension)
        self.rnn = nn.LSTM(input_size = embedding_dimension, 
                           hidden_size = hidden_units, 
                           num_layers = layers,
                           dropout = dropout)
        self.output_layer = nn.Linear(hidden_units, len(out_vocabulary))
        self.architecture = dict(model = "Autoregressive LSTM",
                                 in_vocabulary = in_vocabulary,
                                 out_vocabulary = out_vocabulary,
                                 embedding_dimension = embedding_dimension,
                                 hidden_units = hidden_units, 
                                 layers = layers,
                                 dropout = dropout,
                                 parameters = sum([t.numel() for t in self.parameters()]))
        self.print_architecture()
        
    def forward(self, X):
        X = self.embeddings(X.T)
        rnn, (rnn_last_hidden, rnn_last_memory) = self.rnn(X)
        return self.output_layer(rnn.transpose(0, 1))
    
class TransformerEncoder(Autoregressive):    
    def __init__(self, 
                 in_vocabulary, 
                 out_vocabulary,
                 max_sequence_length = 32,
                 embedding_dimension = 32,
                 feedforward_dimension = 128,
                 layers = 2,
                 attention_heads = 2,
                 activation = "relu",
                 dropout = 0.0):
        super().__init__()
        self.embeddings = nn.Embedding(len(in_vocabulary), embedding_dimension)
        self.positional_embeddings = nn.Embedding(max_sequence_length, embedding_dimension)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model = embedding_dimension, 
                                                            dim_feedforward = feedforward_dimension,
                                                            nhead = attention_heads, 
                                                            activation = activation,
                                                            dropout = dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer = self.transformer_layer,
                                             num_layers = layers)
        self.output_layer = nn.Linear(embedding_dimension, len(out_vocabulary))
        self.architecture = dict(model = "Autoregressive Transformer Encoder",
                                 in_vocabulary = in_vocabulary,
                                 out_vocabulary = out_vocabulary,
                                 max_sequence_length = max_sequence_length,
                                 embedding_dimension = embedding_dimension,
                                 feedforward_dimension = feedforward_dimension,
                                 layers = layers,
                                 attention_heads = attention_heads,
                                 activation = activation,
                                 dropout = dropout,
                                 parameters = sum([t.numel() for t in self.parameters()]))
        self.print_architecture()
        
    def forward(self, X):
        assert X.shape[1] <= self.architecture["max_sequence_length"]
        X = self.embeddings(X)
        X_positional = torch.arange(X.shape[1], device = X.device).repeat((X.shape[0], 1))
        X_positional = self.positional_embeddings(X_positional)
        X = (X + X_positional).transpose(0, 1)
        mask = (torch.triu(torch.ones(X.shape[0], X.shape[0])) == 1).transpose(0, 1).to(X.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        output = self.encoder.forward(src = X, mask = mask).transpose(0, 1)
        return self.output_layer(output)
