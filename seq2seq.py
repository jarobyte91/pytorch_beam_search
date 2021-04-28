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
    def forward(self, X, Y):
        """
        Since this class implements an encoder-decoder architecture, its children classes should have an encoder 
        method and a decoder method. This is done for efficiency reasons in the decoding methods.
        """
        context = self.encoder(X)
        decoder = self.decoder(Y = Y, context = context)
        return decoder

    def greedy_search(self, 
                      X, 
                      max_predictions = 20):
        with torch.no_grad():
            Y = torch.ones(X.shape[0], 1).long().to(next(self.parameters()).device)
            log_probabilities = torch.zeros(X.shape[0]).to(next(self.parameters()).device)
            for i in range(max_predictions):
                next_log_probabilities = self.forward(X, Y)[:, -1].log_softmax(-1)
                max_next_log_probabilities, next_chars = next_log_probabilities.max(-1)
                next_chars = next_chars.unsqueeze(-1)
                Y = torch.cat((Y, next_chars), axis = 1)
                log_probabilities += max_next_log_probabilities
        return Y, log_probabilities

    def sample(self, 
               X, 
               max_predictions = 20,
               temperature = 1):
        Y = torch.ones(X.shape[0], 1).long().to(next(self.parameters()).device)
        for i in range(max_predictions):
            next_log_probabilities = self.forward(X, Y)[:, -1]
            next_probabilities = (next_log_probabilities / temperature).softmax(1)
            random = torch.rand((next_probabilities.shape[0], 1)).to(next(self.parameters()).device)
            next_chars = ((next_probabilities.cumsum(1) < random).sum(1, keepdims = True))
            Y = torch.cat((Y, next_chars), axis = 1)
        return Y

    def beam_search(self, 
                    X, 
                    max_predictions = 20,
                    beam_width = 5,
                    candidates = 5,
                    batch_size = 50, 
                    verbose = 0):
        with torch.no_grad():
            Y = torch.ones(X.shape[0], 1).to(next(self.parameters()).device).long()
            dataset = tud.TensorDataset(X, Y)
            loader = tud.DataLoader(dataset, batch_size = batch_size)
            next_log_probabilities = []
            iterator = iter(loader)
            if verbose > 1:
                iterator = tqdm(iterator)
            context = []
            for x, y in iterator:
                c = self.encoder(x)
                context.append(c.repeat((candidates, 1, 1, 1)).transpose(0, 1).flatten(end_dim = 1))
                next_log_probabilities.append(self.decoder(Y = y, context = c)[:, -1, :])
            context = torch.cat(context, axis = 0)
            next_log_probabilities = torch.cat(next_log_probabilities, axis = 0)
            log_probabilities, next_chars = next_log_probabilities.squeeze().log_softmax(-1)\
            .topk(k = candidates, axis = -1)
            Y = Y.repeat((candidates, 1))
            next_chars = next_chars.reshape(-1, 1)
            Y = torch.cat((Y, next_chars), axis = -1)
            # This has to be minus one because we already produced a round
            # of predictions before the for loop.
            predictions_iterator = range(max_predictions - 1)
            if verbose > 0:
                predictions_iterator = tqdm(predictions_iterator)
            for i in predictions_iterator:
                dataset = tud.TensorDataset(context, Y)
                loader = tud.DataLoader(dataset, batch_size = batch_size)
                next_log_probabilities = []
                iterator = iter(loader)
                if verbose > 1:
                    iterator = tqdm(iterator)
                for c, y in iterator:
                    next_log_probabilities.append(self.decoder(Y = y, context = c)[:, -1, :])
                next_log_probabilities = torch.cat(next_log_probabilities, axis = 0)
                best_next_log_probabilities, next_chars = next_log_probabilities.log_softmax(-1)\
                .topk(k = beam_width, axis = -1)
                best_next_log_probabilities = best_next_log_probabilities.reshape(X.shape[0], -1)
                next_chars = next_chars.reshape(-1, 1)
                Y = torch.cat((Y.repeat((1, beam_width)).reshape(-1, Y.shape[1]), 
                               next_chars), 
                              axis = -1)
                log_probabilities = log_probabilities\
                                    .repeat(beam_width, 1, 1)\
                                    .permute(1, 2, 0)\
                                    .flatten(start_dim = 1)
                log_probabilities += best_next_log_probabilities
                log_probabilities, best_candidates = log_probabilities.topk(k = candidates, axis = -1)
                fix_indices = candidates * beam_width * torch.arange(X.shape[0], 
                                                                     device = next(self.parameters()).device)\
                .repeat((candidates, 1)).T.flatten()
                Y = torch.index_select(input = Y, 
                                       dim = 0, 
                                       index = fix_indices + best_candidates.flatten())
            return Y.reshape(-1, candidates, max_predictions + 1), log_probabilities
        
    def fit(self, 
            X_train, 
            Y_train, 
            X_dev = None, 
            Y_dev = None, 
            batch_size = 100, 
            epochs = 10, 
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
                    print("save")
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
    
    
class Seq2SeqRNN(Seq2Seq):
    def __init__(self, 
                 in_vocabulary, 
                 out_vocabulary, 
                 in_embedding_dimension = 64,
                 out_embedding_dimension = 64,
                 encoder_hidden_units = 128, 
                 encoder_layers = 1,
                 decoder_hidden_units = 128,
                 decoder_layers = 1,
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
        self.architecture = dict(model = "Seq2Seq RNN",
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
        
    def encoder(self, X):
        X = self.in_embeddings(X.T)
        encoder, (encoder_last_hidden, encoder_last_memory) = self.encoder_rnn(X)
        return encoder_last_hidden.transpose(0, 1)
    
    def decoder(self, Y, context):
        context = context.flatten(start_dim = 1).unsqueeze(1)
        context = context.repeat((1, Y.shape[1], 1)).transpose(0, 1)
        Y = self.out_embeddings(Y.T)
        Y = torch.cat((Y, context), axis = -1)
        decoder, (decoder_last_hidden, decoder_last_memory) = self.decoder_rnn(Y)
        output = self.output_layer(decoder.transpose(0, 1))
        return output        
    
    
class Transformer(Seq2Seq):    
    def __init__(self, 
                 in_vocabulary, 
                 out_vocabulary, 
                 embedding_dimension = 64,
                 feedforward_dimension = 128,
                 encoder_layers = 1,
                 decoder_layers = 1,
                 attention_heads = 2,
                 activation = "relu",
                 dropout = 0.0):
        super().__init__()
        self.in_embeddings = nn.Embedding(len(in_vocabulary), embedding_dimension)
        self.out_embeddings = nn.Embedding(len(out_vocabulary), embedding_dimension)
        self.positional_embeddings = nn.Embedding(350, embedding_dimension)
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
                                 embedding_dimension = embedding_dimension,
                                 feedforward_dimension = feedforward_dimension,
                                 encoder_layers = encoder_layers,
                                 decoder_layers = decoder_layers,
                                 attention_heads = attention_heads,
                                 activation = activation,
                                 dropout = dropout,
                                 parameters = sum([t.numel() for t in self.parameters()]))
        self.print_architecture()
        
    def encoder(self, X):
        X = self.in_embeddings(X)
        X_positional = torch.arange(X.shape[1], device = next(self.parameters()).device).repeat((X.shape[0], 1))
        X_positional = self.positional_embeddings(X_positional)
        X = (X + X_positional).transpose(0, 1)
        encoder_output = self.transformer.encoder(X).transpose(0, 1)
        return encoder_output
    
    def decoder(self, Y, context):
        context = context.transpose(0, 1)
        Y = self.out_embeddings(Y)
        Y_positional = torch.arange(Y.shape[1], device = next(self.parameters()).device).repeat((Y.shape[0], 1))
        Y_positional = self.positional_embeddings(Y_positional)
        Y = (Y + Y_positional).transpose(0, 1)
        mask = self.transformer.generate_square_subsequent_mask(Y.shape[0]).to(next(self.parameters()).device)
        decoder_output = self.transformer.decoder(tgt = Y, 
                                                  memory = context, 
                                                  tgt_mask = mask)
        decoder_output = decoder_output.transpose(0, 1)
        return self.output_layer(decoder_output)
