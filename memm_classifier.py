from typing import Iterator, Sequence, Text, Tuple, Union
import itertools
import numpy as np
from scipy.sparse import spmatrix
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import unicodedata

NDArray = Union[np.ndarray, spmatrix]
TokenSeq = Sequence[Text]
PosSeq = Sequence[Text]

def read_ptbtagged(ptbtagged_path: str) -> Iterator[Tuple[TokenSeq, PosSeq]]:
    """Reads sentences from a Penn TreeBank .tagged file.
    Each sentence is a sequence of tokens and part-of-speech tags.

    Penn TreeBank .tagged files contain one token per line, with an empty line
    marking the end of each sentence. Each line is composed of a token, a tab
    character, and a part-of-speech tag. Here is an example:

        What	WP
        's	VBZ
        next	JJ
        ?	.

        Slides	NNS
        to	TO
        illustrate	VB
        Shostakovich	NNP
        quartets	NNS
        ?	.

    :param ptbtagged_path: The path of a Penn TreeBank .tagged file, formatted
    as above.
    :return: An iterator over sentences, where each sentence is a tuple of
    a sequence of tokens and a corresponding sequence of part-of-speech tags.
    """
    sentence=tuple()
    tokens=[]
    tags=[]
    #Read the file
    with open(ptbtagged_path,"r",encoding="utf-8") as f:
        for line in f:
            #if not a blank line split into tokens and tags
            if line.strip():
                tokens.append(line.split()[0])
                tags.append(line.split()[1])
            #if blank line then append as a tuple pair
            elif len(line.strip()) == 0:
                sentence+=(tuple([tokens]+[tags]),)
                tokens=[]
                tags=[]
        sentence+=(tuple([tokens]+[tags]),)
    #return iterator of tuple pair of (token,part-of-speech tags)
    return iter(sentence)

class Classifier(object):
    def __init__(self):
        """Initializes the classifier."""
        self.feature_matrix=[]
        self.label=[]
        self.le = preprocessing.LabelEncoder()
        self.dictvec = DictVectorizer()
        self.log_reg = linear_model.LogisticRegression(multi_class='multinomial',solver = 'liblinear')

    def train(self, tagged_sentences: Iterator[Tuple[TokenSeq, PosSeq]]) -> Tuple[NDArray, NDArray]:
        """Trains the classifier on the part-of-speech tagged sentences,
        and returns the feature matrix and label vector on which it was trained.

        The feature matrix should have one row per training token. The number
        of columns is up to the implementation, but there must at least be 1
        feature for each token, named "token=T", where "T" is the token string,
        and one feature for the part-of-speech tag of the preceding token,
        named "pos-1=P", where "P" is the part-of-speech tag string, or "<s>" if
        the token was the first in the sentence. For example, if the input is:

            What	WP
            's	VBZ
            next	JJ
            ?	.

        Then the first row in the feature matrix should have features for
        "token=What" and "pos-1=<s>", the second row in the feature matrix
        should have features for "token='s" and "pos-1=WP", etc. The alignment
        between these feature names and the integer columns of the feature
        matrix is given by the `feature_index` method below.

        The label vector should have one entry per training token, and each
        entry should be an integer. The alignment between part-of-speech tag
        strings and the integers in the label vector is given by the
        `label_index` method below.

        :param tagged_sentences: An iterator over sentences, where each sentence
        is a tuple of a sequence of tokens and a corresponding sequence of
        part-of-speech tags.
        :return: A tuple of (feature-matrix, label-vector).
        """
        val=None
        f=[]

        #Loop through iterator to get the token,pos pair
        while True:
           try:
                   val=next(tagged_sentences)
                   self.label.extend(val[1])
                   val[1].insert(0,'<s>')
                   val[1].pop()
                   f+=[list(i) for i in zip(val[0], val[1])]
           except StopIteration:
               break

        #Create features
        for i in range(len(f)):
            features={}

            #create prefix and suffix feature
            features["PREFIX_{0}".format(f[i][0]).lower()[:3]] = 1
            features["SUFFIX_{0}".format(f[i][0]).lower()[-2:]] = 1

            #create next word feature
            if i< len(f)-1:
                features["word+1"] = f[i+1][0]

            #create shape feature
            features["shape"]=''.join(map(unicodedata.category,f[i][0]))
            #create numeric feature
            if f[i][0].isnumeric():
                features["NUMERIC"] = 1

            #create initcaps word feature
            if f[i][0].strip()[0].isupper():
                features["INITCAPS"] = 1
            #creat unigram word feature
            features["UNIGRAM_%s" % f[i][0].lower()] = 1

            #create bigram and first word in sentence feature
            if i != 0:
                features["BIGRAM_{0}_{1}".format(f[i][0].lower(), f[i-1][0].lower())] = 1
            else:
                features["FIRST_WORD_IN_SEQUENCE"] = 1
            #create bigram word  and last word in sentence feature
            if i != len(f)- 1:
                features["BIGRAM_{0}_{1}".format(f[i][0].lower(), f[i+1][0].lower())] = 1
            else:
                features["LAST_WORD_IN_SEQUENCE"] = 1
            #create Trigram word features
            if i >= 1 and i < len(f) - 1:
                w1 = f[i][0].lower()
                w2 = f[i-1][0].lower()
                w3 = f[i+1][0].lower()
                features["TRIGRAM_{0}_{1}_{2}".format(w2, w1, w3)] = 1
            features.update({'token':f[i][0],'pos-1':f[i][1]})
            self.feature_matrix.append(features)

        #Fitting the classifier
        self.log_reg.fit(self.dictvec.fit_transform(self.feature_matrix),self.le.fit_transform(self.label))

        return tuple([self.dictvec.fit_transform(self.feature_matrix)]+[self.le.fit_transform(self.label)],)


    def feature_index(self, feature: Text) -> int:
        """Returns the column index corresponding to the given named feature.

        The `train` method should always be called before this method is called.

        :param feature: The string name of a feature.
        :return: The column index of the feature in the feature matrix returned
        by the `train` method.
        """
        return self.dictvec.get_feature_names().index(feature)

    def label_index(self, label: Text) -> int:
        """Returns the integer corresponding to the given part-of-speech tag

        The `train` method should always be called before this method is called.

        :param label: The part-of-speech tag string.
        :return: The integer for the part-of-speech tag, to be used in the label
        vector returned by the `train` method.
        """
        return {l: i for i, l in enumerate(self.le.classes_)}[label]

    def predict(self, tokens: TokenSeq) -> PosSeq:
        """Predicts part-of-speech tags for the sequence of tokens.

        This method delegates to either `predict_greedy` or `predict_viterbi`.
        The implementer may decide which one to delegate to.

        :param tokens: A sequence of tokens representing a sentence.
        :return: A sequence of part-of-speech tags, one for each token.
        """
        _, pos_tags = self.predict_greedy(tokens)
        # _, _, pos_tags = self.predict_viterbi(tokens)
        return pos_tags

    def predict_greedy(self, tokens: TokenSeq) -> Tuple[NDArray, PosSeq]:
        """Predicts part-of-speech tags for the sequence of tokens using a
        greedy algorithm, and returns the feature matrix and predicted tags.

        Each part-of-speech tag is predicted one at a time, and each prediction
        is considered a hard decision, that is, when predicting the
        part-of-speech tag for token i, the model will assume that its
        prediction for token i-1 is correct and unchangeable.

        The feature matrix should have one row per input token, and be formatted
        in the same way as the feature matrix in `train`.

        :param tokens: A sequence of tokens representing a sentence.
        :return: The feature matrix and the sequence of predicted part-of-speech
        tags (one for each input token).
        """
        Y_pred= []
        feature_matrix=[]
        #creating features
        for i in range(len(tokens)):
            features={}
            #create prefix and suffix feature
            features["PREFIX_{0}".format(tokens[i]).lower()[:3]] = 1
            features["SUFFIX_{0}".format(tokens[i]).lower()[-2:]] = 1
            #create shape feature
            features["shape"]=''.join(map(unicodedata.category,tokens[i]))
            #create next word feature
            if i< len(tokens)-1:
                features["word+1"] = tokens[i+1]

            #create numeric feature
            if tokens[i].strip().isnumeric():
               features["NUMERIC"] = 1

            #create initcaps feature
            if tokens[i].strip()[0].isupper():
                features["INITCAPS"] = 1

            #creat unigram word feature
            features["UNIGRAM_%s" % tokens[i].lower()] = 1

            #create bigram and first word in sequence feature
            if i != 0:
                features["BIGRAM_{0}_{1}".format(tokens[i].lower(), tokens[i-1][0].lower())] = 1
            else:
                features["FIRST_WORD_IN_SEQUENCE"] = 1
            #create bigram and last word in sequence feature
            if i != len(tokens)- 1:
                features["BIGRAM_{0}_{1}".format(tokens[i].lower(), tokens[i+1].lower())] = 1
            else:
                features["LAST_WORD_IN_SEQUENCE"] = 1

            #create Trigram word features
            if i >= 1 and i < len(tokens) - 1:
                w1 = tokens[i].lower()
                w2 = tokens[i-1].lower()
                w3 = tokens[i+1].lower()
                features["TRIGRAM_{0}_{1}_{2}".format(w2, w1, w3)] = 1
            feature_matrix.append(features)
            #Predicting
            Y_pred.append(self.le.classes_[self.log_reg.predict(self.dictvec.transform(features))][0])
            #Adding pos-1 feature
            if i==0:
                features["pos-1"]='<s>'
            else:
                features["pos-1"]=Y_pred[i-1]

        return tuple([self.dictvec.transform(feature_matrix)]+[Y_pred],)





    def predict_viterbi(self, tokens: TokenSeq) -> Tuple[NDArray, NDArray, PosSeq]:
        """Predicts part-of-speech tags for the sequence of tokens using the
        Viterbi algorithm, and returns the transition probability tensor,
        the Viterbi lattice, and the predicted tags.

        The entry i,j,k in the transition probability tensor should correspond
        to the log-probability estimated by the classifier of token i having
        part-of-speech tag k, given that the previous part-of-speech tag was j.
        Thus, the first dimension should match the number of tokens, the second
        dimension should be one more than the number of part of speech tags (the
        last entry in this dimension corresponds to "<s>"), and the third
        dimension should match the number of part-of-speech tags.

        The entry i,k in the Viterbi lattice should correspond to the maximum
        log-probability achievable via any path from token 0 to token i and
        ending at assigning token i the part-of-speech tag k.

        The predicted part-of-speech tags should correspond to the highest
        probability path through the lattice.

        :param tokens: A sequence of tokens representing a sentence.
        :return: The transition probability tensor, the Viterbi lattice, and the
        sequence of predicted part-of-speech tags (one for each input token).
        """


def test_train_tensors():
    classifier = memm.Classifier()
    ptb_train = memm.read_ptbtagged("PTBSmall/train.tagged")
    ptb_train = itertools.islice(ptb_train, 2)  # just the 1st 2 sentences
    features_matrix, labels_vector = classifier.train(ptb_train)
    assert features_matrix.shape[0] == 31
    assert labels_vector.shape[0] == 31

    # train.tagged starts with
    # Pierre	NNP
    # Vinken	NNP
    # ,	,
    # 61	CD
    # years	NNS
    # old	JJ
    assert features_matrix[4, classifier.feature_index("token=years")] == 1
    assert features_matrix[4, classifier.feature_index("token=old")] == 0
    assert features_matrix[4, classifier.feature_index("pos-1=CD")] == 1
    assert features_matrix[4, classifier.feature_index("pos-1=NNS")] == 0
    assert features_matrix[0, classifier.feature_index("pos-1=<s>")] == 1
    assert labels_vector[3] == classifier.label_index("CD")
    assert labels_vector[4] == classifier.label_index("NNS")


test_train_tensors()






