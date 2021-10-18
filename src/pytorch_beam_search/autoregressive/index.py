from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from nltk import lm
import re

class Index():
    def __init__(self, corpus):
        self.special_tokens = ["<PAD>"]
        self.vocabulary = lm.Vocabulary(corpus)
        tokens = self.special_tokens + list(sorted(self.vocabulary))
        self.voc2idx = {c:i for i, c in enumerate(tokens)}
        self.idx2voc = {i:c for i, c in enumerate(tokens)}

    def __len__(self):
        return len(self.voc2idx)
    
    def __str__(self):
        return f"<Autoregressive Index with {len(self):,} items>"
        
    def text2tensor(self, 
                    strings, 
                    progress_bar = True):
        if progress_bar:
            iterator = tqdm(strings)
        else:
            iterator = strings
        m = max([len(s) for s in strings])
        idx = []
        for l in iterator:
            idx.append([0 for i in range(m - len(l))] + [self.voc2idx[self.vocabulary.lookup(c)] for c in l])
        return torch.tensor(idx)

    def tensor2text(self, 
                    X, 
                    separator = "", 
                    end = "<PAD>"):
        return [separator.join([self.idx2voc[i] for i in l]) for l in X.tolist()] 
    