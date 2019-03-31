import math
import os
import glob
import pre_process
import time
import collections
import bisect

def get_uniquewords_with_count(document):
    words = document.split()
    return get_counted_list(words)

def get_counted_list(tokens):
    dict_tokens = dict(collections.Counter(tokens))
    return list(dict_tokens.items())

def get_prob_with_add_one_smoothing(target_document, prior_prob, sorted_count_list, num_tokens, vocab):
    vocab_size = get_vocabsize(vocab)
    max_index_dict = len(sorted_count_list)
    max_index_vocab = len(vocab)
    keys = [str(token[0]) for token in sorted_count_list]
    index = -1
    input_list = get_uniquewords_with_count(target_document)
    total_prob = math.log(prior_prob, 2)
    for token in input_list:
        index = bisect.bisect(keys, token[0])
        if (0 < index and index < max_index_dict) or ( max_index_dict == index and keys[index-1] == token[0]):
            numerator = sorted_count_list[index-1][1] + 1
            denominator = num_tokens + vocab_size
            total_prob += math.log(numerator/denominator, 2) * token[1]
            index = bisect.bisect(vocab, token[0])
            if (0 < index and index < max_index_vocab) or (max_index_vocab == index and vocab[index-1] == token[0]):
                denominator = num_tokens + vocab_size
                total_prob += math.log(1/ denominator, 2) * token[1]
    return total_prob

def get_vocabsize(vocab):
    return len(vocab)

def get_cmap(c1, c1_prob, c2, c2_prob):
    cmap = c1
    if c1_prob < c2_prob:
        cmap = c2
    return cmap

class1_path = 'movie-review-HW2/aclImdb/train/neg'
class1_name = os.path.basename(class1_path)
class2_path ='movie-review-HW2/aclImdb/train/pos'
class2_name = os.path.basename(class2_path)
class1_tokens = pre_process.get_token_list(class1_path)
class2_tokens = pre_process.get_token_list(class2_path)

class1_list = get_counted_list(class1_tokens)
class2_list = get_counted_list(class2_tokens)
class1_list.sort()
class2_list.sort()

# Output parameters
pre_process.output_paramenters(pre_process.format_parameter_for_output(class1_name, class1_list), pre_process.format_parameter_for_output(class2_name, class2_list), (os.getcwd() + "params.txt"))

num_files_in_class1 = pre_process.num_of_txt_files(class1_path)
num_files_in_class2 = pre_process.num_of_txt_files(class2_path)

vocab_file_path = "movie-review-HW2/aclImdb/imdb.vocab"
vocab_string = open(vocab_file_path, "r", encoding="utf=8").read()
vocabulary = vocab_string.split()

path = "movie-review-HW2/aclImdb/test/neg"
wfile = open("/home/nyjoey/Downloads/NLP/Project2/train_neg.txt", "w+")
for filename in glob.glob(os.path.join(path, "*.txt")):
    file = open(filename, 'r', encoding="utf=8")
    content = file.read()
    if len(content) <= 0:
        break
    pre_processed = pre_process.preprocess(content)
    class1_prob = get_prob_with_add_one_smoothing(pre_processed, num_files_in_class1 / (num_files_in_class1 + num_files_in_class2)
    , class1_list, len(class1_tokens), vocabulary)
    class2_prob = get_prob_with_add_one_smoothing(pre_processed, num_files_in_class2 / (num_files_in_class1 + num_files_in_class2), class2_list, len(class2_tokens), vocabulary)

    wfile.write("File name: " + os.path.basename(filename) + ", Given class: "
               + str(get_cmap(class1_name, class1_prob, class2_name, class2_prob))
               +  ", Original Class: " + os.path.basename(path) + "\n\n")
    print(os.path.basename(filename))

path = "movie-review-HW2/aclImdb/test/pos"
for filename in glob.glob(os.path.join(path, "*.txt")):
    content = open(filename, 'r', encoding="utf=8").read()
    if len(content) <= 0:
        break
    pre_processed = pre_process.preprocess(content)
    class1_prob = get_prob_with_add_one_smoothing(pre_processed, num_files_in_class1 / (num_files_in_class1 +num_files_in_class2),class1_list,len(class1_tokens), vocabulary)
    class2_prob = get_prob_with_add_one_smoothing(pre_processed, num_files_in_class2 /(num_files_in_class1 +num_files_in_class2),class2_list,len(class2_tokens), vocabulary)

    wfile.write("File name: " + os.path.basename(filename) + ", Given class: " + str(get_cmap(class1_name, class1_prob, class2_name, class2_prob)) +  ", Original Class: " + os.path.basename(path) + "\n\n")
    print(os.path.basename(filename))
    
#NB classifier for Question1.1
def NBClassifier():
    pos_class = 0.4
    neg_class = 0.6
    pos_list = {}
    pos_list["I"] = 0.09
    pos_list["always"] = 0.07
    pos_list["like"] = 0.29
    pos_list["foreign"] = 0.04
    pos_list["films"] = 0.08
    neg_list = {}
    neg_list["I"] = 0.16
    neg_list["always"] = 0.06
    neg_list["like"] = 0.06
    neg_list["foreign"] = 0.15
    neg_list["films"] = 0.11
    pos_class_prob = pos_list["I"] * pos_list["always"] * pos_list["like"] * pos_list["foreign"] * pos_list["films"]
    print("P(pos) * P(S|pos) = P(I|pos) * P(always|pos) * P(like|pos) * P(foreign|pos) * P(films|pos)")
    print( "= " + str(pos_list["I"]) + " * " + str(pos_list["always"]) + " * " + str(pos_list["like"]) + " * " + str(pos_list["foreign"]) + " * " + str(pos_list["films"]))
    print( "= " + str(pos_class_prob)+'\n')

    neg_class_prob = neg_list["I"] * neg_list["always"] * neg_list["like"] * neg_list["foreign"] * neg_list["films"]
    print( "P(neg) * P(S|neg) = P(I|neg) * P(always|neg) * P(like|neg) * P(foreign|neg) * P(films|neg)")
    print( "= " + str(neg_list["I"]) + " * " + str(neg_list["always"]) + " * " + str(neg_list["like"]) + " * " + str(neg_list["foreign"]) + " * " + str(neg_list["films"]))
    print( "= " + str(neg_class_prob)+'\n')
    if neg_class_prob > pos_class_prob:
        print( "Naive Bayes Classifier: Negative Class")
    else:
        print( "Naive Bayes Classifier: Positive Class")
    print()

NBClassifier()

list1 =  ["fun", "couple", "love", "love", "comedy"];
list2 =  ["fast", "furious", "shoot", "action"];
list3 = ["couple", "fly", "fast", "fun", "fun", "comedy"];
list4 =  ["furious", "shoot", "shoot", "fun", "action"];
list5 =  ["fly", "fast", "shoot", "love", "action"];

Test = ["fast","couple","shoot","fly"];

fast = Test[0]
couple = Test[1]
shoot = Test[2]
fly = Test[3]

fastCom = 0
coupleCom = 0
shootCom = 0
flyCom = 0

fastAc = 0
coupleAc = 0
shootAc = 0
flyAc = 0

comedy = 0
action = 0

lastWordCom = list1[-1]

cleanlist1 = []
[cleanlist1.append(x) for x in list1 if x not in cleanlist1]
print(cleanlist1)
if(cleanlist1[-1] == lastWordCom): 
    comedy += 1
    if(fast in cleanlist1):
        fastCom += 1
    if(couple in cleanlist1):
        coupleCom += 1
    if(shoot in cleanlist1):
        shootCom += 1
    if(fly in cleanlist1):
        flyCom += 1
else:
    action += 1
    if(fast in cleanlist1):
        fastAc += 1
    if(couple in cleanlist1):
        coupleAc += 1
    if(shoot in cleanlist1):
        shootAc += 1
    if(fly in cleanlist1):
        flyAc += 1

print("fastCom: " , fastCom)
print("coupleCom: " , coupleCom)
print("shootCom: " , shootCom)
print("flyCom: " , flyCom)
print("               ")
print("fastAc: " , fastAc)
print("coupleAc: " , coupleAc)
print("shootAc: " , shootAc)
print("flyAc: " , flyAc)
print("               ")

cleanlist2 = []
[cleanlist2.append(x) for x in list2 if x not in cleanlist2]
print(cleanlist2)
if(cleanlist2[-1] == lastWordCom):
    comedy += 1
    if(fast in cleanlist2):
        fastCom += 1
    if(couple in cleanlist2):
        coupleCom += 1
    if(shoot in cleanlist2):
        shootCom += 1
    if (fly in cleanlist2):
        flyCom += 1
else:
    action += 1
    if(fast in cleanlist2):
        fastAc += 1
    if(couple in cleanlist2):
        coupleAc += 1
    if(shoot in cleanlist2):
        shootAc += 1
    if (fly in cleanlist2):
        flyAc += 1

print("fastCom: " , fastCom)
print("coupleCom: " , coupleCom)
print("shootCom: " , shootCom)
print("flyCom: " , flyCom)
print("            ")
print("fastAc: " , fastAc)
print("coupleAc: " , coupleAc)
print("shootAc: " , shootAc)
print("flyAc: " , flyAc)
print("                 ")

cleanlist3 = []
[cleanlist3.append(x) for x in list3 if x not in cleanlist3]
print(cleanlist3)
if(cleanlist3[-1] == lastWordCom):
    comedy += 1
    if(fast in cleanlist3):
        fastCom += 1
    if(couple in cleanlist3):
        coupleCom += 1
    if(shoot in cleanlist3):
        shootCom += 1
    if(fly in cleanlist3):
        flyCom += 1
else:
    action += 1
    if(fast in cleanlist3):
        fastAc += 1
    if(couple in cleanlist3):
        coupleAc += 1
    if(shoot in cleanlist3):
        shootAc += 1
    if(fly in cleanlist3):
        flyAc += 1

print("fastCom: " , fastCom)
print("coupleCom: " , coupleCom)
print("shootCom: " , shootCom)
print("flyCom: " , flyCom)
print("                   ")
print("fastAc: " , fastAc)
print("coupleAc: " , coupleAc)
print("shootAc: " , shootAc)
print("flyAc: " , flyAc)
print("                 ")

cleanlist4 = []
[cleanlist4.append(x) for x in list4 if x not in cleanlist4]
print(cleanlist4)
if(cleanlist4[-1] == lastWordCom): 
    comedy += 1
    if(fast in cleanlist4):
        fastCom += 1
    if(couple in cleanlist4):
        coupleCom += 1
    if(shoot in cleanlist4):
        shootCom += 1
    if(fly in cleanlist4):
        flyCom += 1
else:
    action += 1
    if(fast in cleanlist4):
        fastAc += 1
    if(couple in cleanlist4):
        coupleAc += 1
    if(shoot in cleanlist4):
        shootAc += 1
    if(fly in cleanlist4):
        flyAc += 1


print("fastCom: " , fastCom)
print("coupleCom: " , coupleCom)
print("shootCom: " , shootCom)
print("flyCom: " , flyCom)
print("                 ")
print("fastAc: " , fastAc)
print("coupleAc: " , coupleAc)
print("shootAc: " , shootAc)
print("flyAc: " , flyAc)
print("                 ")

cleanlist5 = []
[cleanlist5.append(x) for x in list5 if x not in cleanlist5]
print(cleanlist5)
if(cleanlist5[-1] == lastWordCom): 
    comedy += 1
    if(fast in cleanlist5):
        fastCom += 1
    if(couple in cleanlist5):
        coupleCom += 1
    if(shoot in cleanlist5):
        shootCom += 1
    if(fly in cleanlist5):
        flyCom += 1
else:
    action += 1
    if(fast in cleanlist5):
        fastAc += 1
    if(couple in cleanlist5):
        coupleAc += 1
    if(shoot in cleanlist5):
        shootAc += 1
    if(fly in cleanlist5):
        flyAc += 1


print("fastCom: " , fastCom)
print("coupleCom: " , coupleCom)
print("shootCom: " , shootCom)
print("flyCom: " , flyCom)
print("                 ")
print("fastAc: " , fastAc)
print("coupleAc: " , coupleAc)
print("shootAc: " , shootAc)
print("flyAc: " , flyAc)
print("                 ")


print("Count of sentences that end with comedy are: ", comedy)
print("Count of sentences that end with action are: ",action)

posteriorProbCom = comedy / ((comedy+action)*1.0)
print("Posterior probability of comedy is: ",posteriorProbCom)

posteriorProbAct = action/ ((comedy+action)*1.0)
print("Posterior probability of actions is: ",posteriorProbAct)
print("                 ")

probFastCom = (fastCom * 1.0) / comedy
probFastAc = (fastAc*1.0)/action
print("probability of fast in comedy is: ", probFastCom)
print("probability of fast in action is: ", probFastAc)
print("                 ")

probCoupleCom = (coupleCom * 1.0) / comedy
probCoupleAc = (coupleAc*1.0)/action
print("probability of couple in comedy is: ", probCoupleCom)
print("probability of couple in action is: ", probCoupleAc)
print("                  ")

probShootCom = (shootCom * 1.0) / comedy
probShootAc = (shootAc*1.0)/action
print("probability of shoot in comedy is: ", probShootCom)
print("probability of shoot in action is: ", probShootAc)
print("                 ")

probFlyCom = (flyCom * 1.0) / comedy
probFlyAc = (flyAc*1.0)/action
print("probability of fly in comedy is: ", probFlyCom)
print("probability of fly in action is: ", probFlyAc)
print("                 ")

uniCom = []
uniAc = []
if(lastWordCom in cleanlist1):
 cleanlist1.remove("comedy")
 uniCom.extend(cleanlist1)
else:
    cleanlist1.remove("action")
    uniAc.extend(cleanlist1)


if(lastWordCom in cleanlist2):
 cleanlist2.remove("comedy")
 uniCom.extend(cleanlist2)
else:
    cleanlist2.remove("action")
    uniAc.extend(cleanlist2)

print(cleanlist2)

if(lastWordCom in cleanlist3):
 cleanlist3.remove("comedy")
 uniCom.extend(cleanlist3)
else:
    cleanlist3.remove("action")
    uniAc.extend(cleanlist3)

print(cleanlist3)

if(lastWordCom in cleanlist4):
 cleanlist4.remove("comedy")
 uniCom.extend(cleanlist4)
else:
    cleanlist4.remove("action")
    uniAc.extend(cleanlist4)

print(cleanlist4)

if(lastWordCom in cleanlist5):
 cleanlist5.remove("comedy")
 uniCom.extend(cleanlist5)
else:
    cleanlist5.remove("action")
    uniAc.extend(cleanlist5)

print(cleanlist5)

def getUniqueWords(allWords) :
    uniqueWords = []
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords

print("these are unique words in comedy: ",getUniqueWords(uniCom))
print("these are unique words in action: ",getUniqueWords(uniAc))

uniLastCom = getUniqueWords(uniCom)
uniLastAc = getUniqueWords(uniAc)
f = open("movie-review-small.NB",'w')
print ('Unique words in comedy sentences are: ', uniLastCom)
print ('Unique Words in action sentences are ', uniLastAc)
