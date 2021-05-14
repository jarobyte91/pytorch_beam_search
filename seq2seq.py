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

class Seq2Seq(nn.Module):         
    def greedy_search(self, 
                      X, 
                      max_predictions = 20,
                      verbose = False):
        with torch.no_grad():
            Y = torch.ones(X.shape[0], 1).long().to(next(self.parameters()).device)
            log_probabilities = torch.zeros(X.shape[0]).to(next(self.parameters()).device)
            iterator = range(max_predictions)
            if verbose:
                iterator = tqdm(iterator)
            for i in iterator:
                next_log_probabilities = self.forward(X, Y)[:, -1].log_softmax(-1)
                max_next_log_probabilities, next_chars = next_log_probabilities.max(-1)
                next_chars = next_chars.unsqueeze(-1)
                Y = torch.cat((Y, next_chars), axis = 1)
                log_probabilities += max_next_log_probabilities
        return Y, log_probabilities

    def sample(self, 
               X, 
               max_predictions = 20,
               temperature = 1,
               verbose = False):
        with torch.no_grad():
            Y = torch.ones(X.shape[0], 1).long().to(next(self.parameters()).device)
            log_probabilities = torch.zeros(X.shape[0]).to(next(self.parameters()).device)
            iterator = range(max_predictions)
            if verbose:
                iterator = tqdm(iterator)
            for i in iterator:
                next_log_probabilities = self.forward(X, Y)[:, -1]
                next_probabilities = (next_log_probabilities / temperature).softmax(1)
                random = torch.rand((next_probabilities.shape[0], 1)).to(next(self.parameters()).device)
                next_chars = ((next_probabilities.cumsum(1) < random).sum(1, keepdims = True))
                log_probabilities += torch.gather(input = next_probabilities.log(), dim = 1, index = next_chars).squeeze()
                Y = torch.cat((Y, next_chars), axis = 1)
            return Y, log_probabilities

    def beam_search(self, 
                    X, 
                    max_predictions = 20,
                    beam_width = 5,
                    batch_size = 50, 
                    verbose = 0):
        with torch.no_grad():
            Y = torch.ones(X.shape[0], 1).to(next(self.parameters()).device).long()
            # The next command can be a memory bottleneck, can be controlled with the batch 
            # size of the predict method.
            next_probabilities = self.forward(X, Y)[:, -1, :]
            vocabulary_size = next_probabilities.shape[-1]
            probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1)\
            .topk(k = beam_width, axis = -1)
            Y = Y.repeat((beam_width, 1))
            next_chars = next_chars.reshape(-1, 1)
            Y = torch.cat((Y, next_chars), axis = -1)
            # This has to be minus one because we already produced a round
            # of predictions before the for loop.
            predictions_iterator = range(max_predictions - 1)
            if verbose > 0:
                predictions_iterator = tqdm(predictions_iterator)
            for i in predictions_iterator:
                dataset = tud.TensorDataset(X.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1), Y)
                loader = tud.DataLoader(dataset, batch_size = batch_size)
                next_probabilities = []
                iterator = iter(loader)
                if verbose > 1:
                    iterator = tqdm(iterator)
                for x, y in iterator:
                    next_probabilities.append(self.forward(x, y)[:, -1, :].log_softmax(-1))
                next_probabilities = torch.cat(next_probabilities, axis = 0)
                next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
                probabilities = probabilities.unsqueeze(-1) + next_probabilities
                probabilities = probabilities.flatten(start_dim = 1)
                probabilities, idx = probabilities.topk(k = beam_width, axis = -1)
                next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
                best_candidates = (idx / vocabulary_size).long()
                best_candidates += torch.arange(Y.shape[0] // beam_width, device = X.device).unsqueeze(-1) * beam_width
                Y = Y[best_candidates].flatten(end_dim = -2)
                Y = torch.cat((Y, next_chars), axis = 1)
            return Y.reshape(-1, beam_width, Y.shape[-1]), probabilities
        
    def fit(self, 
            X_train, 
            Y_train, 
            X_dev = None, 
            Y_dev = None, 
            batch_size = 100, 
            epochs = 5, 
            learning_rate = 0.0001, 
            verbose = 0, 
            weight_decay = 0, 
            save_path = None):
        assert X_train.shape[0] == Y_train.shape[0]
        assert (X_dev is None and Y_dev is None) or (X_dev is not None and Y_dev is not None) 
        if (X_dev is not None and Y_dev is not None):
            assert X_dev.shape[0] == Y_dev.shape[0]
            dev = True
        else:
            dev = False
        train_dataset = tud.TensorDataset(X_train, Y_train)
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
            for x, y in train_iterator:
                # compute loss and backpropagate
                log_probabilities = self.forward(x, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]
                loss = criterion(log_probabilities, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # compute accuracy
                predictions = log_probabilities.argmax(1)
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
                                                         Y_dev, 
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
                print(f"Tokens in the input vocabulary: {len(self.architecture[k]):,}")
            elif k == "out_vocabulary":
                print(f"Tokens in the output vocabulary: {len(self.architecture[k]):,}")
            elif k == "parameters":
                print(f"Trainable parameters: {self.architecture[k]:,}")
            else:
                print(f"{k.replace('_', ' ').capitalize()}: {self.architecture[k]}")
        print()
            
    def evaluate(self, X, Y, criterion, batch_size = 100, verbose = False):
        dataset = tud.TensorDataset(X, Y)
        loader = tud.DataLoader(dataset, batch_size = batch_size)
        self.eval()
        losses = []
        errors = []
        sizes = []
        with torch.no_grad():
            iterator = iter(loader)
            if verbose:
                iterator = tqdm(iterator)
            for batch in iterator:
                x, y = batch
                # compute loss
                log_probabilities = self.forward(x, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]
                loss = criterion(log_probabilities, y)
                # compute accuracy
                predictions = log_probabilities.argmax(1)
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
                max_predictions = None, 
                method = "beam_search",
                main_batch_size = 100,
                main_verbose = False,
                **kwargs):
        if max_predictions is None:
            max_predictions = X.shape[1]
        self.eval()
        dataset = tud.TensorDataset(X.to(next(self.parameters()).device))
        loader = tud.DataLoader(dataset, batch_size = main_batch_size)
        final_indexes = []
        final_log_probabilities = []
        iterator = iter(loader)
        if main_verbose:
            iterator = tqdm(iterator)
        if method == "beam_search":
            for x in iterator:
                indexes, log_probabilities = self.beam_search(X = x[0], 
                                                              max_predictions = max_predictions, 
                                                              **kwargs)
                # In this case, we only return the best candidate for each example
                final_indexes.append(indexes[:, 0, :])
                final_log_probabilities.append(log_probabilities)
        elif method == "greedy_search":
            for x in iterator:
                indexes, log_probabilities = self.greedy_search(X = x[0], 
                                                                max_predictions = max_predictions, 
                                                                **kwargs)
                final_indexes.append(indexes)
                final_log_probabilities.append(log_probabilities)
        elif method == "sample":
            for x in iterator:
                indexes, log_probabilities = self.sample(X = x[0], 
                                                         max_predictions = max_predictions, 
                                                         **kwargs)        
                final_indexes.append(indexes)
        else:
            raise ValueError("Decoding method not implemented")
        final_indexes = torch.cat(final_indexes, axis = 0)
        final_log_probabilities = torch.cat(final_log_probabilities, axis = 0)
        return final_indexes, final_log_probabilities
    
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
    if name == "Seq2Seq LSTM":
        model = LSTM(**architecture)
    elif name == "Transformer":
        model = Transformer(**architecture)
    else:
        raise Exception(f"Unknown architecture: {name}")
    return model
        
class LSTM(Seq2Seq):
    def __init__(self, 
                 in_vocabulary, 
                 out_vocabulary, 
                 in_embedding_dimension = 32,
                 out_embedding_dimension = 32,
                 encoder_hidden_units = 128, 
                 encoder_layers = 2,
                 decoder_hidden_units = 128,
                 decoder_layers = 2,
                 dropout = 0.0):
        super().__init__()
        self.in_embeddings = nn.Embedding(len(in_vocabulary), in_embedding_dimension)
        self.out_embeddings = nn.Embedding(len(out_vocabulary), out_embedding_dimension)
        self.encoder_rnn = nn.LSTM(input_size = in_embedding_dimension, 
                                   hidden_size = encoder_hidden_units, 
                                   num_layers = encoder_layers,
                                   dropout = dropout)
        self.decoder_rnn = nn.LSTM(input_size = encoder_layers * encoder_hidden_units + out_embedding_dimension, 
                                   hidden_size = decoder_hidden_units, 
                                   num_layers = decoder_layers,
                                   dropout = dropout)
        self.output_layer = nn.Linear(decoder_hidden_units, len(out_vocabulary))
        self.architecture = dict(model = "Seq2Seq LSTM",
                                 in_vocabulary = in_vocabulary, 
                                 out_vocabulary = out_vocabulary, 
                                 in_embedding_dimension = in_embedding_dimension,
                                 out_embedding_dimension = out_embedding_dimension,
                                 encoder_hidden_units = encoder_hidden_units, 
                                 encoder_layers = encoder_layers,
                                 decoder_hidden_units = decoder_hidden_units,
                                 decoder_layers = decoder_layers,
                                 dropout = dropout,
                                 parameters = sum([t.numel() for t in self.parameters()]))
        self.print_architecture()
        
    def forward(self, X, Y):
        X = self.in_embeddings(X.T)
        encoder, (encoder_last_hidden, encoder_last_memory) = self.encoder_rnn(X)
        encoder_last_hidden = encoder_last_hidden.transpose(0, 1).flatten(start_dim = 1)
        encoder_last_hidden = encoder_last_hidden.repeat((Y.shape[1], 1, 1))
        Y = self.out_embeddings(Y.T)
        Y = torch.cat((Y, encoder_last_hidden), axis = -1)
        decoder, (decoder_last_hidden, decoder_last_memory) = self.decoder_rnn(Y)
        output = self.output_layer(decoder.transpose(0, 1))
        return output        
    
class Transformer(Seq2Seq):    
    def __init__(self, 
                 in_vocabulary, 
                 out_vocabulary,
                 max_sequence_length = 32,
                 embedding_dimension = 64,
                 feedforward_dimension = 256,
                 encoder_layers = 2,
                 decoder_layers = 2,
                 attention_heads = 2,
                 activation = "relu",
                 dropout = 0.0):
        super().__init__()
        self.in_embeddings = nn.Embedding(len(in_vocabulary), embedding_dimension)
        self.out_embeddings = nn.Embedding(len(out_vocabulary), embedding_dimension)
        self.positional_embeddings = nn.Embedding(max_sequence_length, embedding_dimension)
        self.transformer = nn.Transformer(d_model = embedding_dimension, 
                                          dim_feedforward = feedforward_dimension,
                                          nhead = attention_heads, 
                                          num_encoder_layers = encoder_layers, 
                                          num_decoder_layers = decoder_layers,
                                          activation = activation,
                                          dropout = dropout)
        self.output_layer = nn.Linear(embedding_dimension, len(out_vocabulary))
        self.architecture = dict(model = "Transformer",
                                 in_vocabulary = in_vocabulary,
                                 out_vocabulary = out_vocabulary,
                                 max_sequence_length = max_sequence_length,
                                 embedding_dimension = embedding_dimension,
                                 feedforward_dimension = feedforward_dimension,
                                 encoder_layers = encoder_layers,
                                 decoder_layers = decoder_layers,
                                 attention_heads = attention_heads,
                                 activation = activation,
                                 dropout = dropout,
                                 parameters = sum([t.numel() for t in self.parameters()]))
        self.print_architecture()
        
    def forward(self, X, Y):
        X = self.in_embeddings(X)
        X_positional = torch.arange(X.shape[1], device = X.device).repeat((X.shape[0], 1))
        X_positional = self.positional_embeddings(X_positional)
        X = (X + X_positional).transpose(0, 1)
        Y = self.out_embeddings(Y)
        Y_positional = torch.arange(Y.shape[1], device = Y.device).repeat((Y.shape[0], 1))
        Y_positional = self.positional_embeddings(Y_positional)
        Y = (Y + Y_positional).transpose(0, 1)
        mask = self.transformer.generate_square_subsequent_mask(Y.shape[0]).to(Y.device)
        transformer_output = self.transformer.forward(src = X,
                                                      tgt = Y, 
                                                      tgt_mask = mask)
        transformer_output = transformer_output.transpose(0, 1)
        return self.output_layer(transformer_output)