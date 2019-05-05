import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    
scores = [7, -2, -5, 4, 0]

print("Question 1: Convert the following set of scores to probabilities: 7, -2, -5, 4, 0\n")
print(softmax(scores))

