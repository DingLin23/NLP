import os
import sys
import json
import math


def turn_to_list(count):
    doc = []
    for key, value in count.items():
        for i in range(value):
            doc.append(key)
    return doc

def create_documents(train_documents, test_documents, classes):
    training_file = sys.argv[1]
    processed_feature_vectors = open(training_file, 'r')
    for line in processed_feature_vectors.readlines():
        vector = json.loads(line)
        train_documents.append(vector)
        key = list(vector.keys())[0]
        if key in classes:
            classes[key].append(vector[key])
        else:
            classes[key] = [vector[key]]

    processed_feature_vectors.close()

    test_file = sys.argv[2]
    processed_feature_vectors = open(test_file, 'r')
    for line in processed_feature_vectors.readlines():
        vector = json.loads(line)
        key = list(vector.keys())[0]
        if key in test_documents:
            test_documents[key].append(turn_to_list(vector[key]))
        else:
            test_documents[key] = [turn_to_list(vector[key])]
    processed_feature_vectors.close()


def train_naive_bayes(number_of_documents, classes, vocabulary):
    prior_probabilities = {}
    num_of_words_in_each_class = {}
    bow = {}
    each_word_probability = {}
    for label, document in classes.items():
        num_of_documents_in_class = len(document)
        prior_probabilities[label] = math.log2(num_of_documents_in_class/number_of_documents)
        bow[label] = {}
        num_of_words_in_each_class[label] = 0
        for doc in document:
            for word, value in doc.items():
                if word in {"the", "in", "a", "it", "are", "an", "and", "as", "at", "be", "by", "for", "from", "has", "he", "is", "it", "of", "on", "that", "to", "were", "was", "will", "with"}:
                    continue
                num_of_words_in_each_class[label] += value
                if word in bow[label]:
                   bow[label][word] += value
                else:
                    bow[label][word] = value
        for word in vocabulary:
            count = 0
            if word in bow[label]:
                count = bow[label][word]
            each_word_probability[(word, label)] = math.log2((count+1) / (num_of_words_in_each_class[label] + len(vocabulary)))
    return prior_probabilities, each_word_probability, bow


def test_naive_bayes(test_documents, classes, vocabulary, prior_probabilities, each_word_probabilities):
    sum_prior_probabilities = {}
    for label, documents_in_each_class in classes.items():
        sum_prior_probabilities[label] = prior_probabilities[label]
        for word in test_documents:
            if word in {"the", "in", "a", "it", "are", "an", "and", "as", "at", "be", "by", "for", "from", "has", "he",
                        "is", "it", "of", "on", "that", "to", "were", "was", "will", "with"}:
                continue
            if word in vocabulary:
                sum_prior_probabilities[label] += each_word_probabilities[(word, label)]
    return arg_max(sum_prior_probabilities)


def arg_max(sum_prior_probabilities):
    v = list(sum_prior_probabilities.values())
    k = list(sum_prior_probabilities.keys())
    return k[v.index(max(v))]


def pretty_print(each_word_probabilities):
    pretty = ""
    for key, val in each_word_probabilities.items():
        w = str(key[0])
        c = str(key[1])
        pretty += 'Probability of "' + w + '" being classified as ' + c + ' is ' + str(val) + '\n'
    return pretty



parameters_file = sys.argv[3]
predictions_file = sys.argv[4]

vocabulary = set([line.rstrip() for line in open('Reviews/imdb.vocab')])

train_documents = []
test_documents = {}
classes = {}

create_documents(train_documents, test_documents, classes)
print("Finished Creating Documents\n")
number_of_documents = len(train_documents)
prior_probabilities, each_word_probabilities, bow = train_naive_bayes(number_of_documents, classes, vocabulary)
print("Finished Computing Prior Probabilities\n")

results = {True: 0, False: 0}
predictions = "Predictions for Test Reviews: \n\n"
num = 1
for label, documents in test_documents.items():
    for document in documents:
        test_result = test_naive_bayes(document, classes, vocabulary, prior_probabilities, each_word_probabilities)
        results[test_result == label] +=1
        predictions += "\t" + str(num) + "\t\t | \t\t Predicted Class: " + test_result + "\t\t | \t\t Actual Class: " + label + "\n"
        num += 1
        print("Finished First Document of " + label + "\n")
    print("finished " + label + "\n")

parameters_file = open(parameters_file, 'w')
parameters_file.write(pretty_print(each_word_probabilities))
parameters_file.close()
predictions_file = open(predictions_file, 'w')
accuracy = (results[True]/(results[False] + results[True])) *100
predictions += "\n\n Total Reviews: " + str(results[False] + results[True])
predictions += "\n Amount Predicted Correctly: " + str(results[True])
predictions += "\n Amount Predicted Incorrectly: " + str(results[False])
predictions += "\n Accuracy: " + str(accuracy) + "%"
predictions_file.write(predictions)
predictions_file.write("\n\n Prior Probabilities: \n\t Negative: " + str(prior_probabilities["neg"]) + "\n\t Positive: " +str(prior_probabilities["pos"]))
predictions_file.close()
print("Done")




# with open(training_file) as training_file:
#     for