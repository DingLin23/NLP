import math

def addPadding(fileName, newFile):
    with open(fileName, 'r') as f:
        with open(newFile, 'w') as w:
            for line in f.readlines():
                lineEdited = "<s> " + line.replace('\n', ' </s>\n')
                w.write(lineEdited.lower())
                
def unigram(textFile, unigramDict):
    tokens = 0
    tokensNotInTraining = 0
    for line in textFile:
        for word in line.split():
            tokens += 1
            if word not in unigramDict:
                tokensNotInTraining += 1
    print(textFile.name)
    print("Total # of words =", tokens)
    print("Words not in training data =", tokensNotInTraining)
    print("Percentage =", (tokensNotInTraining / tokens))
    print()

def bigram(textFile, unigramDict, bigramDict):
    bigrams = 0
    bigramsNotInTraining = 0
    for line in textFile:
        previous = ""
        for word in line.split():
            if word not in unigramDict:
                word = "<unk>"
            if word == '<s>':
                previous = word
            elif previous + " " + word not in bigramDict:
                bigramsNotInTraining += 1
                bigrams += 1
                previous = word
            else:
                bigrams += 1
                previous = word
    print( textFile.name)
    print("Total # of words =", bigrams)
    print("Bigrams not in training data =", bigramsNotInTraining)
    print("Percentage =", (bigramsNotInTraining / bigrams))
    print()

def sentencePerplexity(sentence, unigramDict, bigramDict, numTokens):
    print(sentence)
    print()
    numWords = len(sentence.split())
    logProb = 0
    for word in sentence.lower().split():
        if word not in unigramDict:
            word = "<unk>"
        if word == '<s>':
            pass
        else:
            logProb += math.log2(unigramDict[word] / numTokens)
            print("p(" + word + ") =", str(unigramDict[word]) + "/" + str(numTokens))
    print()
    print("Unigram log probability =", logProb)
    l = (1 / numWords) * logProb
    print("Unigram perplexity =", pow(2, -l))
    print()
    sentenceProb = 1
    previous = ""
    for word in sentence.lower().split():
        if word not in unigramDict:
            word = "<unk>"
        if word == '<s>':
            previous = word
        elif previous + " " + word not in bigramDict:
            sentenceProb *= (0 / unigramDict[previous])
            print("p(" + word + "|" + previous + ") =", str(0) + "/" + str(unigramDict[previous]))
            previous = word
        else:
            sentenceProb *= (bigramDict[previous + " " + word] / unigramDict[previous])
            print("p(" + word + "|" + previous + ") =",
                  str(bigramDict[previous + " " + word]) + "/" + str(unigramDict[previous]))
            previous = word
    print()
    if sentenceProb == 0:
        print("Bigram log probability =", str(0))
        print("Bigram perplexity =", "Undefined")
    else:
        print("Bigram log probability =", math.log2(sentenceProb))
        l = (1 / numWords) * math.log2(sentenceProb)
        print("Bigram perplexity =", pow(2, -l))
    print()
    logProb = 0
    for word in sentence.lower().split():
        if word not in unigramDict:
            word = "<unk>"
        if word == '<s>':
            previous = word
        elif previous + " " + word not in bigramDict:
            logProb += math.log2(1 / (unigramDict[previous] + len(unigramDict.items())))
            print("p(" + word + "|" + previous + ") =", str(1) + "/" + str(unigramDict[word] + len(unigramDict.items())))
            previous = word
        else:
            logProb += math.log2((bigramDict[previous + " " + word] + 1) / (unigramDict[previous] + len(unigramDict.items())))
            print("p(" + word + "|" + previous + ") =", str(bigramDict[previous + " " + word] + 1) + "/" + str(
                unigramDict[previous] + len(unigramDict.items())))
            previous = word
    print()
    print("Bigram smoothing log probability =", logProb)
    l = (1 / numWords) * logProb
    print("Bigram smoothing perplexity =", pow(2, -l))
    print()
    
def filePerplexity(textFile, unigramDict, bigramDict, tokens):
    numWords = 0
    numWordsBigram = 0
    numLines = 0
    numLinesDiscarded = 0
    for line in textFile:
        numLines += 1
        for word in line.split():
            numWords += 1
            numWordsBigram += 1
    textFile = open(textFile.name, 'r')
    totalUnigramLogProb = 0
    totalBigramLogProb = 0
    totalSmoothedBigramLogProb = 0
    for line in textFile:
        logProb = 0
        for word in line.split():
            if word not in unigramDict:
                word = "<unk>"
            if word == '<s>':
                pass
            else:
                logProb += math.log2(unigramDict[word] / tokens)
        totalUnigramLogProb += logProb
        sentenceProb = 1
        previous = ""
        for word in line.split():
            if word not in unigramDict:
                word = "<unk>"
            if word == '<s>':
                previous = word
            elif previous + " " + word not in bigramDict:
                sentenceProb *= (0 / unigramDict[previous])
                previous = word
            else:
                sentenceProb *= (bigramDict[previous + " " + word] / unigramDict[previous])
                previous = word
        if sentenceProb == 0:
            numWordsBigram -= len(line.split())
            numLinesDiscarded += 1
        else:
            totalBigramLogProb += math.log2(sentenceProb)
        logProb = 0
        for word in line.split():
            if word not in unigramDict:
                word = "<unk>"
            if word == '<s>':
                previous = word
            elif previous + " " + word not in bigramDict:
                logProb += math.log2(1 / (unigramDict[previous] + len(unigramDict.items())))
                previous = word
            else:
                logProb += math.log2((bigramDict[previous + " " + word] + 1) / (unigramDict[previous] + len(unigramDict.items())))
                previous = word
        totalSmoothedBigramLogProb += logProb
    print(textFile.name)
    print()
    l = (1 / numWords) * totalUnigramLogProb
    print("Unigram perplexity =", pow(2, -l))
    print()
    l = (1 / numWordsBigram) * totalBigramLogProb
    print("Bigram perplexity =", pow(2, -l))
    print(numLinesDiscarded, "of", numLines, "sentences had zero probability and were discarded")
    print()
    l = (1 / numWords) * totalSmoothedBigramLogProb
    print("Bigram smoothing perplexity =", pow(2, -l))
    print()

addPadding('brown-train.txt', 'modified-brown-train.txt')
addPadding('brown-test.txt', 'modified-brown-test.txt')
addPadding('learner-test.txt', 'modified-learner-test.txt')

brown_train_file = open('modified-brown-train.txt', 'r')
count = {}
for line in brown_train_file:
    for word in line.split():
        if word in count:
            count[word] += 1
        else:
            count[word] = 1
brown_train_file.close()

currentFile = open('modified-brown-train.txt', 'r')
newFile = open('unknown-brown-train.txt', 'w')
for line in currentFile:
    for word in line.split():
        if count[word] == 1:
            newFile.write("<unk>" + " ")
        elif word == '</s>':
            newFile.write(word)
        else:
            newFile.write(word + " ")
    newFile.write("\n")
currentFile.close()
newFile.close()

textFile = open('unknown-brown-train.txt', 'r')
unigramDict = {}
bigramDict = {}
previous = ""
for line in textFile:
    for word in line.split():
        if word in unigramDict:
            unigramDict[word] += 1
        else:
            unigramDict[word] = 1
        if word == '<s>':
            previous = word
        elif previous + " " + word in bigramDict:
            bigramDict[previous + " " + word] += 1
            previous = word
        else:
            bigramDict[previous + " " + word] = 1
            previous = word
unigramVocSize = len(unigramDict.keys())
textFile.close()

File = open('unknown-brown-train.txt', 'r')
bigramDict = {}
previous = ""
for line in File:
    for word in line.split():
        if word == '<s>':
            previous = word
        elif previous + " " + word not in bigramDict:
            bigramDict[previous + " " + word] = 1
            previous = word
        else:
            bigramDict[previous + " " + word] += 1
            previous = word
File.close()

print("SOLUTION TO #1")
print("Unique words in training corpus =", unigramVocSize)
print()

values = unigramDict.values()
tokens = 0
for val in values:
    tokens += val

print("SOLUTION TO #2")
print("Word tokens in training corpus =", tokens)
print()

print("SOLUTION TO #5/#6")
sentence1 = "<s> He was laughed off the screen . </s>"
sentence2 = "<s> There was no compulsion behind them . </s>"
sentence3 = "<s> I look forward to hearing your reply . </s>"
sentencePerplexity(sentence1, unigramDict, bigramDict, tokens)
sentencePerplexity(sentence2, unigramDict, bigramDict, tokens)
sentencePerplexity(sentence3, unigramDict, bigramDict, tokens)

print("SOLUTION TO #7")
textFile1 = open('modified-brown-test.txt', 'r')
textFile2 = open('modified-learner-test.txt', 'r')
filePerplexity(textFile1, unigramDict, bigramDict, tokens)
filePerplexity(textFile2, unigramDict, bigramDict, tokens)
textFile1.close()
textFile2.close()

unigram_file = open('modified-brown-train.txt', 'r')
unigramDict = {}
for line in unigram_file:
    for word in line.split():
        if word in unigramDict:
            unigramDict[word] += 1
        else:
            unigramDict[word] = 1
unigram_file.close()

print("SOLUTION TO #3")
print()
textFileOne = open('modified-brown-test.txt', 'r')
textFileTwo = open('modified-learner-test.txt', 'r')
unigram(textFileOne, unigramDict)
unigram(textFileTwo, unigramDict)
textFileOne.close()
textFileTwo.close()

print("SOLUTION TO #4")
print()
first_text_File = open('modified-brown-test.txt', 'r')
second_text_File = open('modified-learner-test.txt', 'r')
bigram(first_text_File, unigramDict, bigramDict)
bigram(second_text_File, unigramDict, bigramDict)
first_text_File.close()
second_text_File.close()
