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
    """
    A generic sequence-to-sequence model. All other sequence-to-sequence models should extend this class 
    with a __init__ and forward methods, in the same way as in normal PyTorch.
    """
    def __init__(self, in_vocabulary, out_vocabulary):
        super().__init__()
        self.in2i = {c:i for i, c in enumerate(sorted(in_vocabulary), 3)}
        self.in2i["<PAD>"] = 0
        self.in2i["<START>"] = 1
        self.in2i["<END>"] = 2
        self.i2in = {i:c for i, c in enumerate(sorted(in_vocabulary), 3)}
        self.i2in[0] = "<PAD>"
        self.i2in[1] = "<START>"
        self.i2in[2] = "<END>"
        self.out2i = {c:i for i, c in enumerate(sorted(out_vocabulary), 3)}
        self.out2i["<PAD>"] = 0
        self.out2i["<START>"] = 1
        self.out2i["<END>"] = 2
        self.i2out = {i:c for i, c in enumerate(sorted(out_vocabulary), 3)}
        self.i2out[0] = "<PAD>"
        self.i2out[1] = "<START>"
        self.i2out[2] = "<END>"
        
    def greedy_search(self, 
                      X, 
                      predictions = 20,
                      progress_bar = True):
        """
        Implements Greedy Search to compute the output with the sequences given in X. The method can compute 
        several outputs in parallel with the first dimension of X.

        Parameters
        ----------    
        X: LongTensor of shape (examples, length)
            The sequences to start the decoding process.

        predictions: int
            The number of tokens to append to X.

        progress_bar: bool
            Shows a tqdm progress bar, useful for tracking progress with large tensors.

        Returns
        -------
        Y: LongTensor of shape (examples, length + predictions)
            The output sequences.

        probabilities: FloatTensor of length examples
            The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
            probability of the next token at every step.
        """
        with torch.no_grad():
            Y = torch.ones(X.shape[0], 1).long().to(next(self.parameters()).device)
            probabilities = torch.zeros(X.shape[0]).to(next(self.parameters()).device)
            iterator = range(predictions)
            if progress_bar:
                iterator = tqdm(iterator)
            for i in iterator:
                next_probabilities = self.forward(X, Y)[:, -1].log_softmax(-1)
                max_next_probabilities, next_chars = next_probabilities.max(-1)
                next_chars = next_chars.unsqueeze(-1)
                Y = torch.cat((Y, next_chars), axis = 1)
                probabilities += max_next_probabilities
        return Y, probabilities

    def sample(self, 
               X, 
               predictions = 20,
               temperature = 1.0,
               progress_bar = True):
        """
        Samples the sequence distribution to compute the output with the sequences given in X. The method can compute 
        several outputs in parallel with the first dimension of X.

        Parameters
        ----------    
        X: LongTensor of shape (examples, length)
            The sequences to start the decoding process.

        predictions: int
            The number of tokens to append to X.

        temperature: positive float
            Parameter to control the freedom of the sampling. Higher values give more freedom.

        progress_bar: bool
            Shows a tqdm progress bar, useful for tracking progress with large tensors.

        Returns
        -------
        Y: LongTensor of shape (examples, length + predictions)
            The output sequences.

        probabilities: FloatTensor of length examples
            The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
            probability of the next token at every step.
        """
        with torch.no_grad():
            Y = torch.ones(X.shape[0], 1).long().to(next(self.parameters()).device)
            probabilities = torch.zeros(X.shape[0]).to(next(self.parameters()).device)
            iterator = range(predictions)
            if progress_bar:
                iterator = tqdm(iterator)
            for i in iterator:
                next_probabilities = self.forward(X, Y)[:, -1]
                next_probabilities = (next_probabilities / temperature).softmax(1)
                random = torch.rand((next_probabilities.shape[0], 1)).to(next(self.parameters()).device)
                next_chars = ((next_probabilities.cumsum(1) < random).sum(1, keepdims = True))
                probabilities += torch.gather(input = next_probabilities.log(), dim = 1, index = next_chars).squeeze()
                Y = torch.cat((Y, next_chars), axis = 1)
            return Y, probabilities

    def beam_search(self, 
                    X, 
                    predictions = 20,
                    beam_width = 5,
                    batch_size = 50, 
                    progress_bar = 1):
        """
        Implements Beam Search to compute the output with the sequences given in X. The method can compute 
        several outputs in parallel with the first dimension of X.

        Parameters
        ----------    
        X: LongTensor of shape (examples, length)
            The sequences to start the decoding process.

        predictions: int
            The number of tokens to append to X.

        beam_width: int
            The number of candidates to keep in the search.
            
        batch_size: int
            The batch size of the inner loop of the method, which relies on the beam width. 

        progress_bar: bool
            Shows a tqdm progress bar, useful for tracking progress with large tensors.

        Returns
        -------
        Y: LongTensor of shape (examples, length + predictions)
            The output sequences.
            
        probabilities: FloatTensor of length examples
            The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
            probability of the next token at every step.
        """
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
            predictions_iterator = range(predictions - 1)
            if progress_bar > 0:
                predictions_iterator = tqdm(predictions_iterator)
            for i in predictions_iterator:
                dataset = tud.TensorDataset(X.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1), Y)
                loader = tud.DataLoader(dataset, batch_size = batch_size)
                next_probabilities = []
                iterator = iter(loader)
                if progress_bar > 1:
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
            learning_rate = 10**-4, 
            weight_decay = 0, 
            progress_bar = 2, 
            save_path = None):
        """
        A generic training method with Adam and Cross Entropy.

        Parameters
        ----------    
        X_train: LongTensor of shape (train_examples, train_input_length)
            The input sequences of the training set.
            
        Y_train: LongTensor of shape (train_examples, train_output_length)
            The output sequences of the training set.
            
        X_dev: LongTensor of shape (dev_examples, dev_input_length), optional
            The input sequences for the development set.
            
        Y_train: LongTensor of shape (dev_examples, dev_output_length), optional
            The output sequences for the development set.
            
        batch_size: int
            The number of examples to process in each batch.

        epochs: int
            The number of epochs of the training process.
            
        learning_rate: float
            The learning rate to use with Adam in the training process. 
            
        weight_decay: float
            The weight_decay parameter of Adam (L2 penalty), useful for regularizing models. For a deeper 
            documentation, go to https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam            

        progress_bar: int
            Shows a tqdm progress bar, useful for tracking progress with large tensors.
            If equal to 0, no progress bar is shown. 
            If equal to 1, shows a bar with one step for every epoch.
            If equal to 2, shows the bar when equal to 1 and also shows a bar with one step per batch for every epoch.
            If equal to 3, shows the bars when equal to 2 and also shows a bar to track the progress of the evaluation
            in the development set.
            
        save_path: string, optional
            Path to save the .pt file containing the model parameters when the training ends.

        Returns
        -------
        performance: Pandas DataFrame
            DataFrame with the following columns: epoch, train_loss, train_error_rate, (optionally dev_loss and 
            dev_error_rate), minutes, learning_rate, weight_decay, model, encoder_embedding_dimension, 
            decoder_embedding_dimension, encoder_hidden_units, encoder_layers, decoder_hidden_units, decoder_layers, 
            dropout, parameters and one row for each of the epochs, containing information about the training process.
        """
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
        if progress_bar > 0:
            epochs_iterator = tqdm(epochs_iterator)
        print("Training started")
        print(f"Epochs: {epochs:,}\nLearning rate: {learning_rate}\nWeight decay: {weight_decay}")
        header_1 = "Epoch | Train                "
        header_2 = "      | Loss     | Error Rate"
        rule = "-" * 29
        if dev:
            header_1 += " | Development          "
            header_2 += " | Loss     | Error Rate"
            rule += "-" * 24
        header_1 += " | Minutes"
        header_2 += " |"
        rule += "-" * 10
        print(header_1, header_2, rule, sep = "\n")
        for e in epochs_iterator:
            self.train()
            losses = []
            errors = []
            sizes = []
            train_iterator = train_loader
            if progress_bar > 1:
                train_iterator = tqdm(train_iterator)
            for x, y in train_iterator:
                # compute loss and backpropagate
                probabilities = self.forward(x, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]
                loss = criterion(probabilities, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # compute accuracy
                predictions = probabilities.argmax(1)
                batch_errors = (predictions != y)
                # append the results
                losses.append(loss.item())
                errors.append(batch_errors.sum().item())
                sizes.append(batch_errors.numel())
            train_loss = sum(losses) / len(losses)
            train_error_rate = 100 * sum(errors) / sum(sizes)
            t = (timer() - start) / 60
            status_string = f"{e:>5} | {train_loss:>8.4f} | {train_error_rate:>10.3f}"
            status = {"epoch":e,
                      "train_loss": train_loss,
                      "train_error_rate": train_error_rate}
            if dev:
                dev_loss, dev_error_rate = self.evaluate(X_dev, 
                                                         Y_dev, 
                                                         batch_size = batch_size, 
                                                         progress_bar = progress_bar > 2, 
                                                         criterion = criterion)
                status_string += f" | {dev_loss:>8.4f} | {dev_error_rate:>10.3f}"
                status.update({"dev_loss": dev_loss, "dev_error_rate": dev_error_rate})
            status.update({"training_minutes": t,
                           "learning_rate": learning_rate,
                           "weight_decay": weight_decay})
            performance.append(status)
            if save_path is not None:  
                if (not dev) or (e < 2) or (dev_loss < min([p["dev_loss"] for p in performance[:-1]])):
                    torch.save(self.state_dict(), save_path)
            status_string += f" | {t:>7.1f}"
            print(status_string)
        return pd.concat((pd.DataFrame(performance), 
                          pd.DataFrame([self.architecture for i in performance])), axis = 1)\
               .drop(columns = ["in_vocabulary", "out_vocabulary"])
    
    def print_architecture(self):
        """
        Displays the information about the model in standard output. 
        """
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
            
    def evaluate(self, 
                 X, 
                 Y, 
                 criterion, 
                 batch_size = 100, 
                 progress_bar = True):
        """
        Evaluates the model on a dataset.
        
        Parameters
        ----------
        X: LongTensor of shape (examples, input_length)
            The input sequences of the dataset.
            
        Y: LongTensor of shape (examples, output_length)
            The output sequences of the dataset.
            
        criterion: PyTorch module
            The loss function to evalue the model on the dataset, has to be able to compare self.forward(X, Y) and Y
            to produce a real number.
            
        batch_size: int
            The batch size of the evaluation loop.
            
        progress_bar: bool
            Shows a tqdm progress bar, useful for tracking progress with large tensors.
            
        Returns
        -------
        loss: float
            The average of criterion across the whole dataset.
            
        error_rate: float
            The step-by-step accuracy of the model across the whole dataset. Useful as a sanity check, as it should
            go to zero as the loss goes to zero.
            
        """
        dataset = tud.TensorDataset(X, Y)
        loader = tud.DataLoader(dataset, batch_size = batch_size)
        self.eval()
        losses = []
        errors = []
        sizes = []
        with torch.no_grad():
            iterator = iter(loader)
            if progress_bar:
                iterator = tqdm(iterator)
            for batch in iterator:
                x, y = batch
                # compute loss
                probabilities = self.forward(x, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]
                loss = criterion(probabilities, y)
                # compute accuracy
                predictions = probabilities.argmax(1)
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
                predictions = None, 
                method = "beam_search",
                main_batch_size = 100,
                main_progress_bar = True,
                **kwargs):
        """
        Wrapper unifying the different decoding methods with a data loader, useful for large datasets.
        
        Parameters
        ---------- 
        X: LongTensor of shape (examples, length)
            The sequences to start the decoding process. This is the input for the decoding method.
            
        predictions: int
            The number of tokens to append to X.
            
        method: string
            Decoding method to use, can be one of "beam_search", "greedy_search" or "sample".
            
        main_batch_size: int
            Batch size of the dataset loop. The decoding method is applied to every batch.
            
        main_progress_bar: bool
            Shows a progress for the dataset loop, useful to track progress with large datasets.
            
        **kwargs
            Parameters to pass to the decoding method.
            
        Returns
        -------
        Y: LongTensor of shape (examples, length + predictions)
            The output sequences.
            
        probabilities: FloatTensor of length examples
            The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
            probability of the next token at every step.
        """
        if predictions is None:
            predictions = X.shape[1]
        self.eval()
        dataset = tud.TensorDataset(X.to(next(self.parameters()).device))
        loader = tud.DataLoader(dataset, batch_size = main_batch_size)
        Y = []
        probabilities = []
        iterator = iter(loader)
        if main_progress_bar:
            iterator = tqdm(iterator)
        if method == "beam_search":
            for x in iterator:
                batch_Y, batch_probabilities = self.beam_search(X = x[0], 
                                                                predictions = predictions, 
                                                                **kwargs)
                # In this case, we only return the best candidate for each example
                Y.append(batch_Y[:, 0, :])
                probabilities.append(batch_probabilities)
        elif method == "greedy_search":
            for x in iterator:
                batch_Y, batch_probabilities = self.greedy_search(X = x[0], 
                                                                  predictions = predictions, 
                                                                  **kwargs)
                Y.append(batch_Y)
                probabilities.append(batch_probabilities)
        elif method == "sample":
            for x in iterator:
                batch_Y, batch_probabilities = self.sample(X = x[0], 
                                                           predictions = predictions, 
                                                           **kwargs)        
                Y.append(batch_Y)
                probabilities.append(batch_probabilities)
        else:
            raise ValueError("Decoding method not implemented")
        Y = torch.cat(Y, axis = 0)
        probabilities = torch.cat(probabilities, axis = 0)
        return Y, probabilities
    
    def save_architecture(self, path):
        """
        Saves a dictionary containing all the hyper-parameters of the model to reconstruct it later.
        
        Parameters
        ----------
            path: string
                Path to save the dictionary.
        """
        with open(path, "wb") as file:
            pickle.dump(self.architecture, file)

    def text2tensor(self, strings, vocabulary = None, device = None):
        """
        Utility function to convert  a list of strings into a tensor of integers using a mapping vocabulary.
        
        Parameters
        ----------
        strings: list of strings
            The strings to convert.
           
        vocabulary: dictionary
            Dictionary containing the index:token pairs to perform the conversion.
            
        device: PyTorch device
            Device to allocate the output in.
            
        Returns
        -------
        output: LongTensor of shape (len(strings), max([len(s) for s in strings]))
            Tensor containing the input after conversion.
        """
        if vocabulary is None:
            vocabulary = self.in2i
        if device is None:
            device = next(self.parameters()).device
        return nn.utils.rnn.pad_sequence([torch.tensor([1] + [vocabulary[c] for c in l] + [2]) 
                                          for l in strings], 
                                         batch_first = True).to(device)

    def tensor2text(self, X, separator = "", vocabulary = None, end = "<END>"):
        """
        Utility function to convert a tensor of integers into a list of strings using a mapping vocabulary.
        
        Parameters
        ----------
        X: LongTensor of shape (examples, length)
            Tensor containing the output of
        """
        if vocabulary is None:
            vocabulary = self.i2out
        return [re.sub(end + ".*", end, separator.join([vocabulary[i] for i in l]), flags = re.DOTALL) 
                for l in X.tolist()] 

def load_architecture(path):
    """
    Utility function to reconstruct a model from the dictionary containing its architecture, usually 
    the output of the save_architecture method.
    
    Parameters
    ----------
    path: string
        Path containing the architecture dictionary.
    """
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
  
#######################################################
# MODELS
#######################################################
    
class LSTM(Seq2Seq):
    def __init__(self, 
                 in_vocabulary, 
                 out_vocabulary, 
                 encoder_embedding_dimension = 32,
                 decoder_embedding_dimension = 32,
                 encoder_hidden_units = 128, 
                 encoder_layers = 2,
                 decoder_hidden_units = 128,
                 decoder_layers = 2,
                 dropout = 0.0):
        """
        A standard Seq2Seq LSTM model as in 'Learning Phrase Representations using RNN Encoder-Decoder 
        for Statistical Machine Translation' by Cho et al. (2014). 
        
        Parameters
        ----------
        in_vocabulary: dictionary
            Vocabulary with the index:token pairs for the inputs of the model.
            
        out_vocabulary: dictionary
            Vocabulary with the token:index pairs for the outputs of the model.
            
        encoder_embedding_dimension: int
            Dimension of the embeddings to feed into the encoder.
            
        decoder_embedding_dimension: int
            Dimension of the embeddings to feed into the decoder.
            
        encoder_hidden_units: int
            Hidden size of the encoder.
            
        encoder_layers: int
            Hidden layers of the encoder.
            
        decoder_hidden_units: int
            Hidden units of the decoder.
            
        decoder_layers: int
            Hidden layers of the decoder.
            
        dropout: float between 0.0 and 1.0
            Dropout rate to apply to whole model.
        """
        super().__init__(in_vocabulary = in_vocabulary, out_vocabulary = out_vocabulary)
        self.in_embeddings = nn.Embedding(len(self.in2i), encoder_embedding_dimension)
        self.out_embeddings = nn.Embedding(len(self.out2i), decoder_embedding_dimension)
        self.encoder_rnn = nn.LSTM(input_size = encoder_embedding_dimension, 
                                   hidden_size = encoder_hidden_units, 
                                   num_layers = encoder_layers,
                                   dropout = dropout)
        self.decoder_rnn = nn.LSTM(input_size = encoder_layers * encoder_hidden_units + decoder_embedding_dimension, 
                                   hidden_size = decoder_hidden_units, 
                                   num_layers = decoder_layers,
                                   dropout = dropout)
        self.output_layer = nn.Linear(decoder_hidden_units, len(self.i2out))
        self.architecture = dict(model = "Seq2Seq LSTM",
                                 in_vocabulary = in_vocabulary, 
                                 out_vocabulary = out_vocabulary, 
                                 encoder_embedding_dimension = encoder_embedding_dimension,
                                 decoder_embedding_dimension = decoder_embedding_dimension,
                                 encoder_hidden_units = encoder_hidden_units, 
                                 encoder_layers = encoder_layers,
                                 decoder_hidden_units = decoder_hidden_units,
                                 decoder_layers = decoder_layers,
                                 dropout = dropout,
                                 parameters = sum([t.numel() for t in self.parameters()]))
        self.print_architecture()
        
    def forward(self, X, Y):
        """
        Forward method of the model.
        
        Parameters
        ----------
        X: LongTensor of shape (batch_size, input_length)
            Tensor of integers containing the inputs for the model.
            
        Y: LongTensor of shape (batch_size, output_length)
            Tensor of integers containing the output produced so far.
            
        Returns
        -------
        output: FloatTensor of shape (batch_size, output_length, len(out_vocabulary))
            Tensor of floats containing the inputs for the final Softmax layer (usually integrated into the loss function).
        """
        X = self.in_embeddings(X.T)
        encoder, (encoder_last_hidden, encoder_last_memory) = self.encoder_rnn(X)
        encoder_last_hidden = encoder_last_hidden.transpose(0, 1).flatten(start_dim = 1)
        encoder_last_hidden = encoder_last_hidden.repeat((Y.shape[1], 1, 1))
        Y = self.out_embeddings(Y.T)
        Y = torch.cat((Y, encoder_last_hidden), axis = -1)
        decoder, (decoder_last_hidden, decoder_last_memory) = self.decoder_rnn(Y)
        output = self.output_layer(decoder.transpose(0, 1))
        return output        
    
class ReversingLSTM(Seq2Seq):
    def __init__(self, 
                 in_vocabulary, 
                 out_vocabulary, 
                 encoder_embedding_dimension = 32,
                 decoder_embedding_dimension = 32,
                 encoder_hidden_units = 128, 
                 encoder_layers = 2,
                 decoder_hidden_units = 128,
                 decoder_layers = 2,
                 dropout = 0.0):
        """
        A standard Seq2Seq LSTM model that reverses the order of the input as in 
        'Sequence to sequence learning with Neural Networks' by Sutskever et al. (2014). 
        
        Parameters
        ----------
        in_vocabulary: dictionary
            Vocabulary with the index:token pairs for the inputs of the model.
            
        out_vocabulary: dictionary
            Vocabulary with the token:index pairs for the outputs of the model.
            
        encoder_embedding_dimension: int
            Dimension of the embeddings to feed into the encoder.
            
        decoder_embedding_dimension: int
            Dimension of the embeddings to feed into the decoder.
            
        encoder_hidden_units: int
            Hidden size of the encoder.
            
        encoder_layers: int
            Hidden layers of the encoder.
            
        decoder_hidden_units: int
            Hidden units of the decoder.
            
        decoder_layers: int
            Hidden layers of the decoder.
            
        dropout: float between 0.0 and 1.0
            Dropout rate to apply to whole model.
        """
        super().__init__(in_vocabulary = in_vocabulary, out_vocabulary = out_vocabulary)
        self.in_embeddings = nn.Embedding(len(self.in2i), encoder_embedding_dimension)
        self.out_embeddings = nn.Embedding(len(self.out2i), decoder_embedding_dimension)
        self.encoder_rnn = nn.LSTM(input_size = encoder_embedding_dimension, 
                                   hidden_size = encoder_hidden_units, 
                                   num_layers = encoder_layers,
                                   dropout = dropout)
        self.decoder_rnn = nn.LSTM(input_size = decoder_embedding_dimension, 
                                   hidden_size = decoder_hidden_units, 
                                   num_layers = decoder_layers,
                                   dropout = dropout)
        self.output_layer = nn.Linear(decoder_hidden_units, len(self.i2out))
        self.enc2dec = nn.Linear(encoder_hidden_units * encoder_layers, decoder_hidden_units * decoder_layers)
        self.architecture = dict(model = "Seq2Seq Reversing LSTM",
                                 in_vocabulary = in_vocabulary, 
                                 out_vocabulary = out_vocabulary, 
                                 encoder_embedding_dimension = encoder_embedding_dimension,
                                 decoder_embedding_dimension = decoder_embedding_dimension,
                                 encoder_hidden_units = encoder_hidden_units, 
                                 encoder_layers = encoder_layers,
                                 decoder_hidden_units = decoder_hidden_units,
                                 decoder_layers = decoder_layers,
                                 dropout = dropout,
                                 parameters = sum([t.numel() for t in self.parameters()]))
        self.print_architecture()
        
    def forward(self, X, Y):
        """
        Forward method of the model.
        
        Parameters
        ----------
        X: LongTensor of shape (batch_size, input_length)
            Tensor of integers containing the inputs for the model.
            
        Y: LongTensor of shape (batch_size, output_length)
            Tensor of integers containing the output produced so far.
            
        Returns
        -------
        output: FloatTensor of shape (batch_size, output_length, len(out_vocabulary))
            Tensor of floats containing the inputs for the final Softmax layer (usually integrated into the loss function).
        """
        X = self.in_embeddings(torch.flip(X.T, dims = (1, )))
        encoder, (encoder_last_hidden, encoder_last_memory) = self.encoder_rnn(X)
        encoder_last_hidden = encoder_last_hidden.transpose(0, 1).flatten(start_dim = 1)
        enc2dec = self.enc2dec(encoder_last_hidden)\
        .reshape(-1, self.decoder_rnn.num_layers, self.decoder_rnn.hidden_size)\
        .transpose(0, 1)\
        .contiguous()
        Y = self.out_embeddings(Y.T)
        decoder, (decoder_last_hidden, decoder_last_memory) = self.decoder_rnn(Y, (enc2dec, torch.zeros_like(enc2dec)))
        output = self.output_layer(decoder.transpose(0, 1))
        return output
    
class Transformer(Seq2Seq):
    def __init__(self, 
                 in_vocabulary, 
                 out_vocabulary,
                 max_sequence_length = 32,
                 embedding_dimension = 32,
                 feedforward_dimension = 128,
                 encoder_layers = 2,
                 decoder_layers = 2,
                 attention_heads = 2,
                 activation = "relu",
                 dropout = 0.0):
        """
        The standard PyTorch implementation of a Transformer model.
        
        Parameters
        ----------
        in_vocabulary: dictionary
            Vocabulary with the index:token pairs for the inputs of the model.
            
        out_vocabulary: dictionary
            Vocabulary with the token:index pairs for the outputs of the model.
            
        max_sequence_length: int
            Maximum sequence length accepted by the model, both for the encoder and the decoder.
            
        embedding_dimension: int
            Dimension of the embeddings of the model.
            
        feedforward_dimension: int
            Dimension of the feedforward network inside the self-attention layers of the model.
            
        encoder_layers: int
            Hidden layers of the encoder.
            
        decoder_layers: int
            Hidden layers of the decoder.
            
        attention_heads: int
            Attention heads inside every self-attention layer of the model.
            
        activation: string
            Activation function of the feedforward network inside the self-attention layers of the model. Can
            be either 'relu' or 'gelu'.
            
        dropout: float between 0.0 and 1.0
            Dropout rate to apply to whole model.
        """
        super().__init__(in_vocabulary = in_vocabulary, out_vocabulary = out_vocabulary)
        self.in_embeddings = nn.Embedding(len(self.in2i), embedding_dimension)
        self.out_embeddings = nn.Embedding(len(self.out2i), embedding_dimension)
        self.positional_embeddings = nn.Embedding(max_sequence_length, embedding_dimension)
        self.transformer = nn.Transformer(d_model = embedding_dimension, 
                                          dim_feedforward = feedforward_dimension,
                                          nhead = attention_heads, 
                                          num_encoder_layers = encoder_layers, 
                                          num_decoder_layers = decoder_layers,
                                          activation = activation,
                                          dropout = dropout)
        self.output_layer = nn.Linear(embedding_dimension, len(self.i2out))
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
        """
        Forward method of the model.
        
        Parameters
        ----------
        X: LongTensor of shape (batch_size, input_length)
            Tensor of integers containing the inputs for the model.
            
        Y: LongTensor of shape (batch_size, output_length)
            Tensor of integers containing the output produced so far.
            
        Returns
        -------
        output: FloatTensor of shape (batch_size, output_length, len(out_vocabulary))
            Tensor of floats containing the inputs for the final Softmax layer (usually integrated in the loss function).
        """
        assert X.shape[1] <= self.architecture["max_sequence_length"]
        assert Y.shape[1] <= self.architecture["max_sequence_length"]
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
