import string
import glob   
import os

def get_token_list(path_to_dir):
    tokens = []
    for filename in glob.glob(os.path.join(path_to_dir, "*.txt")):
        file = open(filename, 'r', encoding="utf=8")
        content = file.read()
        pre_processed = preprocess(content)
        separated = pre_processed.split()
        for x in separated:
            tokens.append(x)
    return tokens

def num_of_txt_files(path_to_dir):
    return len(glob.glob(os.path.join(path_to_dir, "*.txt")))

def format_parameter_for_output(classname, dict):
    text = str(dict)
    length = len(text)
    return classname + " " + text[1:length - 1]


def output_paramenters(formatetted_param1, formatted_param2, path_to_file):
    file = open(path_to_file, 'w', encoding="utf=8")
    file.write(formatetted_param1 + "\n" + formatted_param2)

def get_preprocessed_document(path_to_file):
    file = open(path_to_file, "r", encoding="utf=8")
    text = file.read()
    return pre_process.preprocess(text)

def format_output(file_name,  given_class):
    return "The file name: " + file_name + " - Given class: " + given_class

def preprocess(document):
    lowered = document.lower()
    punc_removed = remove_punctuations(lowered)
    stop_removed = remove_punctuations(punc_removed)
    return stop_removed

def remove_punctuations(document):
    table = str.maketrans('', '', string.punctuation)
    return document.translate(table)

def remove_stopwords(document):
    stop_words = str( ['a', 'about', 'above', 'after', 'again', 'against', 'aint', 'all', 'am', 'an', 'and', 'any', 'are',
                 'arent', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
                 'can', 'couldn', 'couldnt', 'did', 'didn', 'didnt', 'do', 'does', 'doesn', 'doesnt', 'doing',
                 'dont', 'during', 'each', 'few', 'for', 'from', 'had', 'hadn', 'hadnt', 'has',
                 'hasnt', 'have', 'haven', 'havent', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
                 'how', 'i', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'its', 'itself', 'just', 'll', 'm',
                 'me', 'mightn', 'mightnt', 'more', 'most', 'mustn', 'mustnt', 'my', 'myself', 'needn', 'neednt', 'no', 'nor',
                 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over',
                 'own', 're', 'same', 'shan', 'shant', 'she', 'shes', 'should', 'shouldve', 'shouldn', 'shouldnt', 'so',
                 'some', 'such', 'than', 'that', 'thatll', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
                 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
                 'wasnt', 'we', 'were', 'weren', 'werent', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
                 'will', 'with', 'won', 'wont', 'wouldn', 'wouldnt', 'y', 'you', 'youd', 'youll', 'youre', 'youve', 'your',
                 'yours', 'yourself', 'yourselves', 'could', 'hed', 'hell', 'hes', 'heres', 'hows', 'id', 'ill', 'im', 'ive',
                 'lets', 'ought', 'shed',  'thats', 'theres', 'theyd', 'theyll', 'theyre', 'theyve', 'wed', 'well',
                 'were', 'weve', 'whats', 'whens', 'wheres', 'whos', 'whys', 'would'] )
    tokens = document.split()
    result = [token + " " not in stop_words for token in tokens]
    return result
