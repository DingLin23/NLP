import sys
import os
import json


def count_frequencies(text):
    freq = {}
    for word in text:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq


def remove_unseen_words(words, vocab):
    return [word for word in words if word in vocab]


def add_to_feature_vectors(feature_vectors, _class, processed_text):
    processed_text = remove_unseen_words(processed_text, vocabulary)
    features = count_frequencies(processed_text)
    feature_vectors.append({_class: features})


vocabulary = set([line.rstrip() for line in open('Reviews/imdb.vocab')])

punctuation = {'"', '*', '+', ',', '.', '/', '<', '>', '@', '^', '_', '`', '{', '|', '~'}

reviews_folder = sys.argv[1]

classes = os.listdir(reviews_folder)

feature_vectors = []

for _class in classes:
    current_class = os.path.join(reviews_folder, _class)
    if os.path.isdir(current_class):
        for file in os.listdir(current_class):
            if file.endswith(".txt"):
                review_file = open(os.path.join(current_class, file), "r")
                review_text = review_file.read()
                processed_text = ""
                for c in review_text:
                    if c in '!?':
                        processed_text += " " + c.lower()
                    elif c not in punctuation:
                        processed_text += c.lower()
                processed_text = processed_text.split()
                add_to_feature_vectors(feature_vectors, _class, processed_text)


print(feature_vectors)

output_file_name = sys.argv[1]
#output_file_name = output_file_name[:output_file_name.index("/")]
output_file_name = output_file_name.replace("/", "")

output_file = open("movie-review-" + output_file_name + ".NB", 'w')
for vector in feature_vectors:
    output_file.writelines(json.dumps(vector) + "\n")

