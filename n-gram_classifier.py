from typing import Iterator, Iterable, Tuple, Text, Union
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import string
import re
import sklearn.preprocessing


NDArray = Union[np.ndarray, spmatrix]


def read_smsspam(smsspam_path: str) -> Iterator[Tuple[Text, Text]]:
    """Generates (label, text) tuples from the lines in an SMSSpam file.

    SMSSpam files contain one message per line. Each line is composed of a label
    (ham or spam), a tab character, and the text of the SMS. Here are some
    examples:

      spam	85233 FREE>Ringtone!Reply REAL
      ham	I can take you at like noon
      ham	Where is it. Is there any opening for mca.

    :param smsspam_path: The path of an SMSSpam file, formatted as above.
    :return: An iterator over (label, text) tuples.
    """
    sms_iterator_tuple=[]
    #Opening and reading the file
    with open(smsspam_path,'r+', encoding="utf-8") as f:
        for l in f.readlines():
            #Splitting the content into label and texts
            label=l.strip().split('\t')[0]
            text=l.strip().split('\t')[1]
            #Using regex to replace email address to term emailaddr
            cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)
            #Using regex to replace http address to term httpaddr
            cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr',cleaned)
            #Using regex to replace money symbols to term moneysymb
            cleaned = re.sub(r'£|\$', 'moneysymb', cleaned)
            #Using regex to replace phone number to term phonenumbr
            cleaned = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b','phonenumbr', cleaned)
            #Using regex to replace digits to term numbr
            cleaned = re.sub(r'\d+(\.\d+)?', 'numbr', cleaned)
            #Using regex to replace punctuations and other characters
            cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)
            #Using regex to replace multiple whitespaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            #Using regex to convert to lowercase
            cleaned = re.sub(r'^\s+|\s+?$', '', cleaned.lower())
            sms_iterator_tuple.append(tuple([label]+[cleaned]))
    #Return iterable of cleaned text
    return iter(sms_iterator_tuple)


class TextToFeatures:
    def __init__(self, texts: Iterable[Text]):
        """Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.

        It is up to the implementer exactly what features to produce from a
        text, but the features will always include some single words and some
        multi-word expressions (e.g., "need" and "to you").

        :param texts: The training texts.
        """
        #Initialize the Count vectorizer and selecting ngram range 1 and 2. Also binarising the features.
        self.cv = CountVectorizer(ngram_range=(1,2),binary=True)
        self.countvectorizer=self.cv.fit_transform(texts)
        self.binarizer = preprocessing.Binarizer().fit(self.countvectorizer)


    def index(self, feature: Text):
        """Returns the index in the vocabulary of the given feature value.

        :param feature: A feature
        :return: The unique integer index associated with the feature.
        """
        #Return the index of the input feature
        return self.cv.get_feature_names().index(feature)


    def __call__(self, texts: Iterable[Text]) -> NDArray:
        """Creates a feature matrix from a sequence of texts.

        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.

        It is up to the implementer what the value of a feature that is present
        in a text should be, though a common choice is 1. Features that are
        absent from a text will have the value 0.

        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        """
        #transforming input to count vectorizer features
        self.count=self.cv.transform(texts)
        #returning the feature values as a matrix
        return self.binarizer.transform(self.count.todense())

class TextToLabels:
    def __init__(self, labels: Iterable[Text]):
        """Initializes an object for converting texts to labels.

        During initialization, the provided training labels are analyzed to
        determine the vocabulary, i.e., all labels that the converter will
        support. Each such label will be associated with a unique integer index
        that may later be accessed via the .index() method.

        :param labels: The training labels.
        """
        #Using labelencoder to encode the training labels
        self.le = preprocessing.LabelEncoder()
        #fitting the input labels
        self.label=self.le.fit(labels)
   
        

    def index(self, label: Text) -> int:
        """Returns the index in the vocabulary of the given label.

        :param label: A label
        :return: The unique integer index associated with the label.
        """
        #Returning the index associated with the input label
        return {l: i for i, l in enumerate(self.le.classes_)}[label]

    def __call__(self, labels: Iterable[Text]) -> NDArray:
        """Creates a label vector from a sequence of labels.

        Each entry in the vector corresponds to one of the input labels. The
        value at index j is the unique integer associated with the jth label.

        :param labels: A sequence of labels.
        :return: A vector, with one entry for each label.
        """
        #returning the label vector
        return self.le.fit_transform(labels)


class Classifier:
    def __init__(self):
        """Initalizes a logistic regression classifier.
        """
        #Initializing the Classifier
        self.clf = LogisticRegression(solver='sag',random_state=0,penalty='l2',class_weight='balanced')
    def train(self, features: NDArray, labels: NDArray) -> None:
        """Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        Such vectors will typically be generated via TextToLabels.
        """
        #fitting the training data
        self.clf.fit(features,labels)

    def predict(self, features: NDArray) -> NDArray:
        """Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :return: A prediction vector, where each entry represents a label.
        """
        #predicting the label for the passed in features
        return self.clf.predict(features)



