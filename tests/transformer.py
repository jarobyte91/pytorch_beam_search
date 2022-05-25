from pprint import pprint
from pytorch_beam_search import seq2seq

# Create vocabularies
# Tokenize the way you need
print("train data")
source = [list("abcdefghijkl"), list("mnopqrstwxyz")]
target = [list("ABCDEFGHIJKL"), list("MNOPQRSTWXYZ")]
print("source")
pprint(source)
print("target")
pprint(target)
# An Index object represents a mapping from the vocabulary
# to integers (indices) to feed into the models
print("creating indexes...")
source_index = seq2seq.Index(source)
target_index = seq2seq.Index(target)

# Create tensors
print("creating tensors...")
X = source_index.text2tensor(source)
Y = target_index.text2tensor(target)
# X.shape == (n_source_examples, len_source_examples) == (2, 11)
# Y.shape == (n_target_examples, len_target_examples) == (2, 12)

# Create and train the model
print("creating model...")
model = seq2seq.Transformer(source_index, target_index)    # just a PyTorch model
# model = seq2seq.LSTM(source_index, target_index)    # just a PyTorch model
print("training model...")
model.fit(X, Y, epochs = 100)    # basic method included

# Generate new predictions
print("test data")
new_source = [list("new first in"), list("new second in")]
new_target = [list("new first out"), list("new second out")]
print("new source")
pprint(new_source)
print("new target")
pprint(new_target)
print("creating tensors...")
X_new = source_index.text2tensor(new_source)
Y_new = target_index.text2tensor(new_target)
print("evaluating model...")
loss, error_rate = model.evaluate(X_new, Y_new)    # basic method included
print("beam search...")
predictions, log_probabilities = seq2seq.beam_search(
    model, 
    X_new, 
    # progress_bar = 1, 
    # predictions = 100
)
output = [target_index.tensor2text(p) for p in predictions]
print("\npredictions")
pprint(output)
