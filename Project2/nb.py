
from preprocess import preprocess
import os

class NaiveBayes():
    
    classification = {}

    number_of_docs = None
    inputData = None
    vocab_unique = None
    classes = None

    words = {}
    process = preprocess()
    testing = {}

    train_file = None
    test_file = None

    def __init__(self, train=None, test=None, train_dir=None, test_dir=None):

        if train is not None or test is not None:
            self.train_file = train
            self.test_file = test

            self.number_of_docs = self.process.totalDocs(train)
            self.inputData = self.process.convertFile(train)
            self.vocab_unique = self.process.vocab(self.inputData)
            self.classes = self.process.classes(self.inputData)

        else:
            for i in os.listdir(train_dir + "pos"):

                self.number_of_docs += self.process.totalDocs(train)
                self.inputData += self.process.convertFile(train)
                self.vocab_unique += self.process.vocab(self.inputData)
                self.classes += self.process.classes(self.inputData)
            
            for i in os.listdir(train_dir + "neg"):

                self.number_of_docs += self.process.totalDocs(train)
                self.inputData += self.process.convertFile(train)
                self.vocab_unique += self.process.vocab(self.inputData)
                self.classes += self.process.classes(self.inputData)

            

    def classify(self):
        
        for c in self.classes:
            temp_dict = {}
            for word in self.vocab_unique:
                temp_dict[word] = 0
            self.classification[c] = temp_dict

        for key in self.classification:
            for line in self.inputData:
                if line[len(line) - 1] != key:
                    continue
                else:
                    for word in line:
                        if word not in self.classification[key]:
                            continue
                        (self.classification[key])[word] += 1

    def count_words(self):

        for cl in self.classes:
            count = 0
            for k in self.classification[cl]:
                count += self.classification[cl][k]
            self.words[cl] = count

    def classify_smoothed(self):
        
        self.dict_likelihood = self.classification.copy()

        for cl in self.classes:
            for k in self.dict_likelihood[cl]:
                self.dict_likelihood[cl][k] = (self.dict_likelihood[cl][k] + 1) / float(self.words[cl] +
                                                                                        len(self.vocab_unique))

    def test_prob(self):

        for cl in self.classes:
            d = self.process.totalDocsGivenClass(cl, self.inputData)
            self.process.documents[cl] = d

        self.priors = {}
        for cl in self.classes:
            if cl not in self.priors:
                self.priors[cl] = self.process.documents[cl] / float(self.number_of_docs)

        with open(self.test_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        test_data = content

        for cl in self.classes:
            if cl not in self.testing:
                self.testing[cl] = self.priors[cl]

        for cl in self.classes:
            for w in test_data:
                for key in self.dict_likelihood[cl]:
                    if key != w:
                        continue
                    elif key == w:
                        self.testing[cl] *= self.dict_likelihood[cl][w]
                        
        self.result = max(self.testing, key=self.testing.get)

    def outputResult(self):
        
        string = \
        "Total number of documents/lines in input" + str(self.number_of_docs) + "\n" + \
        "Vocabulary of unique words: " + str(self.vocab_unique) + "\n"+ \
        "Classes:" + str(self.classes) + "\n"+ \
        "Count of words, given class: " + str(self.words) + "\n"+ \
        "Word likelihoods with add - 1 smoothing with respect to class: " + str(self.dict_likelihood) + "\n"+ \
        "Prior probabilities: ",  str(self.priors) + "\n" + \
        "Probabilities of test data: " + str(self.testing) + "\n" + \
        "The most likely class for the test document: " + self.result

        if self.train_file is not None:
            f = open("small.txt", "w")
            f.write("\n".join(string))
            f.close()
        else:
            pass

def small():

    nb = NaiveBayes(train="movie-review-small.NB", test="document.NB")
    nb.classify()
    nb.count_words()
    nb.classify_smoothed()
    nb.test_prob()
    nb.outputResult()

def question1():

    print()
    pos_list = dict()
    pos_list["I"] = 0.09
    pos_list["always"] = 0.07
    pos_list["like"] = 0.29
    pos_list["foreign"] = 0.04
    pos_list["films"] = 0.08

    neg_list = dict()
    neg_list["I"] = 0.16
    neg_list["always"] = 0.06
    neg_list["like"] = 0.06
    neg_list["foreign"] = 0.15
    neg_list["films"] = 0.11

    pos_class_prob = pos_list["I"] * pos_list["always"] * pos_list["like"] * pos_list["foreign"] * pos_list["films"]
    print("P(pos) * P(S|pos) = P(I|pos) * P(always|pos) * P(like|pos) * P(foreign|pos) * P(films|pos)")
    print("= " + str(pos_list["I"]) + " * " + str(pos_list["always"]) + " * " + str(pos_list["like"]) + " * " +
          str(pos_list["foreign"]) + " * " + str(pos_list["films"]))
    print("= " + str(pos_class_prob)+'\n')

    neg_class_prob = neg_list["I"] * neg_list["always"] * neg_list["like"] * neg_list["foreign"] * neg_list["films"]
    print("P(neg) * P(S|neg) = P(I|neg) * P(always|neg) * P(like|neg) * P(foreign|neg) * P(films|neg)")
    print("= " + str(neg_list["I"]) + " * " + str(neg_list["always"]) + " * " + str(neg_list["like"]) +
          " * " + str(neg_list["foreign"]) + " * " + str(neg_list["films"]))

    print("= " + str(neg_class_prob) + '\n')

    if neg_class_prob < pos_class_prob:
        print("Naive Bayes Classifier: Positive Class")
    else:
        print("Naive Bayes Classifier: Negative Class")
        
question1()

if __name__ == "__main__":
    small()
    
