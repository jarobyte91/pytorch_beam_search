from pprint import pprint
from pytorch_beam_search import autoregressive

# Create vocabulary and examples
# tokenize the way you need
corpus = list("abcdefghijklmnopqrstwxyz ")
print("corpus")
print(corpus, "\n")
# len(corpus) == 25
# An Index object represents a mapping from the vocabulary
# to integers (indices) to feed into the models
print("creating index")
index = autoregressive.Index(corpus)
print(index, "\n")
n_gram_size = 17    # 16 with an offset of 1 
n_grams = [
    corpus[i:n_gram_size + i] 
    for i in range(len(corpus))[:-n_gram_size + 1]
]

# Create tensor
print("creating tensor...")
X = index.text2tensor(n_grams)
# X.shape == (n_examples, len_examples) == (25 - 17 + 1 = 9, 17)
print("X", X.shape, "\n")

# Create and train the model
model = autoregressive.TransformerEncoder(index)    # just a PyTorch model
model.fit(X)    # basic method included

# Generate new predictions
print("\ntest data")
new_examples = [list("new first"), list("new second")]
pprint(new_examples)
print("\ncreating tensor...")
X_new = index.text2tensor(new_examples)
print("X_new", X_new.shape, "\n")
print("evaluating model...")
loss, error_rate = model.evaluate(X_new)    # basic method included
print("loss:", loss)
print("error_rate:", error_rate, "\n")
print("beam_search...")
predictions, log_probabilities = autoregressive.beam_search(
    model, 
    X_new,
    predictions = 5,
    # progress_bar = 1
)
# every element in predictions is the list of candidates for each example
output = [index.tensor2text(p) for p in predictions]
print("\npredictions")
pprint(output)
