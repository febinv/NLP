from typing import List, Text, Tuple
import numpy as np
from collections import Counter
import re
from  sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
import numpy as np



def most_common(word_pos_path: Text,
                word_regex=".*",
                pos_regex=".*",
                n=10) -> List[Tuple[Text, int]]:
    """Finds the most common words and/or parts of speech in a file.

    :param word_pos_path: The path of a file containing part-of-speech tagged
    text. The file should be formatted as a sequence of tokens separated by
    whitespace. Each token should be a word and a part-of-speech tag, separated
    by a slash. For example: "The/at Hartsfield/np home/nr is/bez at/in 637/cd
    E./np Pelham/np Rd./nn-tl Aj/nn ./."

    :param word_regex: If None, do not include words in the output. If a regular
    expression string, all words included in the output must match the regular
    expression.

    :param pos_regex: If None, do not include part-of-speech tags in the output.
    If a regular expression string, all part-of-speech tags included in the
    output must match the regular expression.

    :param n: The number of most common words and/or parts of speech to return.

    :return: A list of (token, count) tuples for the most frequent words and/or
    parts-of-speech in the file. Note that, depending on word_regex and
    pos_regex (as described above), the returned tokens will contain either
    words, part-of-speech tags, or both.
    """
    words=[]
    #Reading the file line by line and using regex to catch the appropriate groups
    with open(word_pos_path,'r') as f:
        for line in f:
            for word in line.split():
                #Regex rule1
                if pos_regex is None:
                    m=re.match("("+word_regex+")/",word)
                    words.append(m.group(1))
                #Regex rule2
                elif word_regex is None:
                    m=re.match(pos_regex+"/(.+)",word)
                    words.append(m.group(1))
                #Regex rule3
                else:
                    m=re.match(word_regex,word.split('/')[0])
                    m1=re.match(pos_regex,word.split('/')[1])
                    if m and m1:
                        words.append(word)
    #Using Counter to find the most frequent words/pos in the file
    word_count=Counter(words)
    return_list=[]
    #Getting the most frequent words in the file based on the argument and returning it
    for letter,count in word_count.most_common(n):
        return_list.append((letter,count))
    return return_list






class WordVectors(object):
    def __init__(self, word_vectors_path: Text):
        """Reads words and their vectors from a file.

        :param word_vectors_path: The path of a file containing word vectors.
        Each line should be formatted as a single word, followed by a
        space-separated list of floating point numbers. For example:

            the 0.063380 -0.146809 0.110004 -0.012050 -0.045637 -0.022240
        """

        #Opening the file and initializing required lists
        self.content=open(word_vectors_path,'r')
        self.words=[]
        self.vectors=[]
        #looping through file line by line and extracting words and vector values into initialized lists
        for line in self.content.readlines():
            row = line.strip().split(' ')
            #Extracting the word
            word_vocab = row[0]
            #Extracting the vector values
            vector = np.fromiter((float(x) for x in row[1:]),dtype=np.float)
            self.words.append(word_vocab)
            self.vectors.append(vector)



    def average_vector(self, words: List[Text]) -> np.ndarray:
        """Calculates the element-wise average of the vectors for the given
        words.

        For example, if the words correspond to the vectors [1, 2, 3] and
        [3, 4, 5], then the element-wise average should be [2, 3, 4].

        :param words: The words whose vectors should be looked up and averaged.
        :return: The element-wise average of the word vectors.
        """
        base_vector=[]
        #Finding the index for the input argument Words and using that to find the respective vector values
        for i in words:
            base_vector.append(self.vectors[self.words.index(i)])
        #returning the element-wise average of the passed in vectors
        return np.average(base_vector,axis=0)



    def most_similar(self, word: Text, n=10) -> List[Tuple[Text, int]]:
        """Finds the most similar words to a query word. Similarity is measured
        by cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)
        over the word vectors.

        :param word: The query word.
        :param n: The number of most similar words to return.
        :return: The n most similar words to the query word.
        """
        #Retrieving the vector values of the query word
        base_vector=self.vectors[self.words.index(word)]
        sim=[]
        #looping through the words list and calculating cosine similarity values for each word with respect to input word
        for w in self.words:
                sim.append(cosine_similarity(base_vector.reshape(1,-1),self.vectors[self.words.index(w)].reshape(1,-1)))
        #finding the index of top n similar words as per cosine similarity
        largest=nlargest(n+1,enumerate(sim),key=lambda x: x[1])
        final=[]
        #using the index position from above to find the top n words
        for key,value in largest:
            if self.words[key]!=word:
                final.append((self.words[key],float(value)))
        #returning the top n similar words and their cosine similarity value
        return final






# -------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------
print(most_common("brown_sample.txt", n=2))
#print(most_common("brown_sample.txt",word_regex=".*",pos_regex=None,n=10))
#print(most_common("brown_sample.txt",word_regex=None,pos_regex=".*",n=5))
#print(most_common("brown_sample.txt",  word_regex=".*",pos_regex="vb.*",n=3))
#print(most_common("brown_sample.txt",word_regex=".*ing$",pos_regex=".*",n=4))


#wv=WordVectors("vectors_top3000.txt")
#print(wv.most_similar("house", 3))
#print(wv.most_similar("white", 5))
#print(wv.average_vector(["white", "house"]))










