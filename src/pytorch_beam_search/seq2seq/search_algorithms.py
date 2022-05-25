import torch
import torch.utils.data as tud
from tqdm.auto import tqdm
import warnings

def greedy_search(
    model, 
    X, 
    predictions = 20,
    progress_bar = False
):
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
        Y = torch.ones(X.shape[0], 1).long().to(next(model.parameters()).device)
        probabilities = torch.zeros(X.shape[0]).to(next(model.parameters()).device)
        iterator = range(predictions)
        if progress_bar:
            iterator = tqdm(iterator)
        for i in iterator:
            next_probabilities = model.forward(X, Y)[:, -1].log_softmax(-1)
            max_next_probabilities, next_chars = next_probabilities.max(-1)
            next_chars = next_chars.unsqueeze(-1)
            Y = torch.cat((Y, next_chars), axis = 1)
            probabilities += max_next_probabilities
    return Y, probabilities

def sample(
    model, 
    X, 
    predictions = 20,
    temperature = 1.0,
    progress_bar = False
):
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
        Y = torch.ones(X.shape[0], 1).long().to(next(model.parameters()).device)
        probabilities = torch.zeros(X.shape[0]).to(next(model.parameters()).device)
        iterator = range(predictions)
        if progress_bar:
            iterator = tqdm(iterator)
        for i in iterator:
            next_probabilities = model.forward(X, Y)[:, -1]
            next_probabilities = (next_probabilities / temperature).softmax(1)
            random = torch.rand((next_probabilities.shape[0], 1)).to(next(model.parameters()).device)
            next_chars = ((next_probabilities.cumsum(1) < random).sum(1, keepdims = True))
            probabilities += torch.gather(input = next_probabilities.log(), dim = 1, index = next_chars).squeeze()
            Y = torch.cat((Y, next_chars), axis = 1)
        return Y, probabilities

def beam_search(
    model, 
    X, 
    predictions = 20,
    beam_width = 5,
    batch_size = 50, 
    progress_bar = 0
):
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

    progress_bar: int 
        Shows a tqdm progress bar, useful for tracking progress with large tensors. Ranges from 0 to 2.

    Returns
    -------
    Y: LongTensor of shape (examples, length + predictions)
        The output sequences.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """
    with torch.no_grad():
        Y = torch.ones(X.shape[0], 1).to(next(model.parameters()).device).long()
        # The next command can be a memory bottleneck, can be controlled with the batch 
        # size of the predict method.
        next_probabilities = model.forward(X, Y)[:, -1, :]
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
                next_probabilities.append(model.forward(x, y)[:, -1, :].log_softmax(-1))
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
