import torch
import torch.nn as nn
import torch.utils.data as tud
from tqdm.notebook import tqdm
from pprint import pprint
import numpy as np
import re

class Seq2Seq(nn.Module):   
    """
    Since this class implements an encoder-decoder architecture, its children classes should have an encoder method
    and a decoder method.
    """
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
#             print("X", X.shape)
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
                context.append(c.repeat((candidates, 1, 1, 1)).transpose(0, 1).flatten(end_dim = -2))
                next_log_probabilities.append(self.decoder(Y = y, context = c)[:, -1, :])
            context = torch.cat(context, axis = 0)
            next_log_probabilities = torch.cat(next_log_probabilities, axis = 0)
            log_probabilities, next_chars = next_log_probabilities.squeeze().log_softmax(-1)\
            .topk(k = candidates, axis = -1)
            Y = Y.repeat((candidates, 1))
            next_chars = next_chars.reshape(-1, 1)
            Y = torch.cat((Y, next_chars), axis = -1)
#             print("X", X.shape)
            X_repeated = X.repeat((1, candidates)).reshape(-1, X.shape[1])
            dataset = tud.TensorDataset(X_repeated)
            loader = tud.DataLoader(dataset)
#             context_2 = []
#             for x in iter(loader):
#                 context_2.append(self.encoder(x[0]))
#             context_2 = torch.cat(context_2, axis = 0)
#             print("context difference", (context.flatten() - context_2.flatten()).norm())
#             print("X", X.shape)
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
                for x, y in iterator:
#                     print(y)
                    next_log_probabilities.append(self.decoder(Y = y, context = x)[:, -1, :])
                next_log_probabilities = torch.cat(next_log_probabilities, axis = 0)
                best_next_log_probabilities, next_chars = next_log_probabilities.log_softmax(-1)\
                .topk(k = beam_width, axis = -1)
                best_next_log_probabilities = best_next_log_probabilities.reshape(X.shape[0], -1)
                next_chars = next_chars.reshape(-1, 1)
                Y = torch.cat((Y.repeat((1, beam_width)).reshape(-1, Y.shape[1]), 
                               next_chars), 
                              axis = -1)
                log_probabilities = log_probabilities.repeat(beam_width, 1, 1).permute(1, 2, 0).flatten(start_dim = 1)
                log_probabilities += best_next_log_probabilities
                log_probabilities, best_candidates = log_probabilities.topk(k = candidates, axis = -1)
                fix_indices = candidates * beam_width * torch.arange(X.shape[0], 
                                                                     device = next(self.parameters()).device)\
                .repeat((candidates, 1)).T.flatten()
                Y = torch.index_select(input = Y, 
                                       dim = 0, 
                                       index = fix_indices + best_candidates.flatten())
            return Y.reshape(-1, candidates, max_predictions + 1), log_probabilities
        