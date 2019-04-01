
class preprocess:

    documents = {}
    
    def vocab (self, inList):
        
        vocab = []
        new_list = []
        for line in inList:
            for words in line:
                if words != line[len(line)-1]:
                    new_list.append(words)
        for w in new_list:
            if w in vocab:
                continue
            else:
                vocab.append(w)
        return vocab
    
    def totalDocs (self, textFile):

        count = 0
        with open(textFile, 'r') as read_input:
            for line in read_input:
                count += 1
        read_input.close()
        return count

    def convertFile (self, inputText):
        
        output_list = []
        read_input = open(inputText, 'r')
        for line in read_input:
            temp = line.strip().split()
            output_list.append(temp)
        read_input.close()
        return output_list
    
    def totalDocsGivenClass(self, cl, inputList):

        count = 0
        for line in inputList:
            if line[len(line)-1] != cl:
                continue
            else:
                count += 1
        return count

    def classes (self, inputList):

        output_list = []
        for line in inputList:
            if line[len(line)-1] in output_list:
                continue
            else:
                output_list.append(line[len(line)-1])
        return output_list
