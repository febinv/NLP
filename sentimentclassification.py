from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import spacy
import numpy as np
import unicodedata
nlp=spacy.load('en')
import nltk
from nltk.stem.porter import *
from sklearn.svm import LinearSVC
from sklearn import svm, grid_search
from sklearn.grid_search import GridSearchCV
import re
from sklearn.cross_validation import KFold, cross_val_score
import emoji
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


def read_parse(path):
    tweet_combined=[]
    class_assigned=[]
    final=[]
    emojis=make_emoji_dict()
    emojis1=[]

    with open(path,"r",encoding="utf-8") as f:
        next(f)
        for line in f:
                id=line.split('\t')[0]
                tweet=line.split('\t')[1]
                tweet=tweet.replace('#','')
                tweet=tweet.replace('?',' ?')
                tweet=tweet.replace('\n',' ')
                tweet=tweet.replace('.',' ')
                tweet=tweet.replace('*',' ')
                tweet=tweet.replace('-',' ')
                tweet=tweet.replace(',',' ')
                #email
                tweet = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'email', tweet)
                #http
                tweet = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'url',tweet)
                #phonenumber
                tweet = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b','phonenumber', tweet)
                #username
                tweet=re.sub(r'@([A-Za-z0-9_]+)','username',tweet)
                affect_dimension=line.split('\t')[2]
                class_assigned=line.split('\t')[3].split(':')[0]
                intensity_class_desc=line.split('\t')[3].split(':')[1]
                text_no_emoji_lst = []
                #emojis1.extend(''.join(c for c in tweet if c in emoji.UNICODE_EMOJI))
                for token in tweet:
                    if token in emoji.UNICODE_EMOJI:
                        description = emojis[str(token)]
                        text_no_emoji_lst.append(description)
                    else:
                        text_no_emoji_lst.append(token)
                tweet = "".join(x for x in text_no_emoji_lst)
                final.append(tuple([tweet]+[class_assigned],))
    print(final)
    return iter(final)


def cross_Val(X,y,k_fold):
    k_fold = KFold(len(y), n_folds=10, shuffle=True, random_state=0)
    clf = linear_model.LogisticRegression(multi_class='multinomial',solver = 'sag',max_iter=10000)
    print(cross_val_score(clf, X, y, cv=k_fold, n_jobs=1))



def read_parse_Test(path):
    tweet_combined=[]
    class_assigned=[]
    final=[]

    with open(path,"r",encoding="utf-8") as f:
        next(f)
        for line in f:
            id=line.split('\t')[0]
            tweet=line.split('\t')[1]
            affect_dimension=line.split('\t')[2]
            class_assigned=line.split('\t')[3].split(':')[0]
            final.append(tuple([tweet],))
    return iter(final)


def svc_param_selection(X, y):
    # Cs = [0.001, 0.01, 0.1, 1, 10]
    # gammas = [0.001, 0.01, 0.1, 1]
    # param_grid = {'C': Cs, 'gamma' : gammas}
    # grid_search = GridSearchCV(linear_model.LogisticRegression(multi_class='multinomial',solver = 'sag',max_iter=1000), param_grid, cv=nfolds)
    # grid_search.fit(X, y)
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    kernel=['linear','poly','rbf','sigmoid']
    param_grid = {'C': Cs, 'gamma' : gammas,'kernel':kernel}

    clf = GridSearchCV(svm.SVC(), param_grid)
    best_model=clf.fit(X,y)
    print('Best Kernel:', best_model.best_estimator_.get_params()['kernel'])
    print('Best Gamma:', best_model.best_estimator_.get_params()['gamma'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])
    return None


def make_lex_dict():
    """
    Convert lexicon file to a dictionary
    """
    lex_dict = {}
    with open("vader_lexicon.txt","r",encoding="utf-8") as f:
        for line in f:
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
    return lex_dict


def make_emoji_dict():
    """
    Convert lexicon file to a dictionary
    """
    emoji_dict = {}
    with open("emoji_desc.txt","r",encoding="utf-8") as f:
        for line in f:
            (word, measure) = line.strip().split('\t')[0:2]
            emoji_dict[word] = measure
    return emoji_dict



class Classifier(object):
    def __init__(self):
        """Initializes the classifier."""
        self.features_list=[]
        self.label=[]
        self.le = preprocessing.LabelEncoder()
        self.dictvec = DictVectorizer()
        self.wordlength=0
        self.log_reg = linear_model.LogisticRegression(multi_class='multinomial',solver = 'sag',max_iter=1000)
        #self.log_reg=svm.SVC(kernel='rbf',gamma=0.001,C=10)
        #self.log_reg = Sequential()

    def feature_extractor(self,feature):
        lex=make_lex_dict()
        features={}
        chrlength=0
        tweet_spacy=nlp(feature)
        stemmer = PorterStemmer()
        first_word=0
        pos_scores=[]
        neg_scores=[]
        token_1=''
        for token in tweet_spacy:
            if token.is_stop is True:
                continue
            stemmed_word=stemmer.stem(str(token))

            if str(token).lower() in lex:
                if lex[str(token).lower()]>0:
                    pos_scores.append(lex[str(token).lower()])
                elif lex[str(token).lower()]<0:
                    neg_scores.append(lex[str(token).lower()])
            else:
                pass


            if str(token).strip() and str(token).strip()[0].isupper():
                features["INITCAPS"] = 1

            if first_word!=0 and len(token_1)>=1 and "BIGRAM_{0}_{1}".format(str(token).lower(), str(token_1).lower()) not in features:
                features["BIGRAM_{0}_{1}".format(str(token).lower(), str(token_1).lower())] = 1
            elif len(token_1)>=1:
                #print(token,token_1)
                features["BIGRAM_{0}_{1}".format(str(token).lower(), str(token_1).lower())] += 1


            if first_word== 0:
                features["FIRST_WORD_IN_SEQUENCE"] = 1
                first_word=1


            #features["UNIGRAM_shape_%s" % str(token).lower()] = token.shape_
            # if "UNIGRAM_%s" % str(token.pos).lower() not in features:
            #     features["UNIGRAM_%s" % str(token.pos).lower()] = 1
            # else:
            #     features["UNIGRAM_%s" % str(token.pos).lower()] +=1

            if "UNIGRAM_%s" % str(token).lower() not in features:
                features["UNIGRAM_%s" % str(token).lower()] = 1
            else:
                features["UNIGRAM_%s" % str(token).lower()] += 1


            if "UNIGRAM_stemmed_%s" % str(stemmed_word).lower() not in features:
                features["UNIGRAM_stemmed_%s" % str(stemmed_word).lower()] = 1
            else:
                features["UNIGRAM_stemmed_%s" % str(stemmed_word).lower()] += 1
            if "LEMMA_%s" % str(token.lemma_).lower() not in features:
               features["LEMMA_%s" % str(token.lemma_).lower()]=1
            else:
               features["LEMMA_%s" % str(token.lemma_).lower()] += 1
            token_1=token
            self.wordlength+=1
            # chrlength+=len(token)
            # features["chrlength"]=chrlength
            # features["length"]=wordlength


        features["comp_score"]=sum(pos_scores)+sum(neg_scores)
        features["pos_score"]=sum(pos_scores)
        features["neg_score"]=sum(neg_scores)
        features["pos_words"]=len(pos_scores)
        features["neg_words"]=len(neg_scores)

        return features.copy()


    def train(self, tweets_and_labels):
        for tweet in tweets_and_labels:
            self.label.append(tweet[1])
            self.features_list.append(self.feature_extractor(tweet[0]))
        #print(self.features_list)


        #print(svc_param_selection(self.dictvec.fit_transform(self.features_list),self.le.fit_transform(self.label),10))
        #print(self.features_list)
        print(svc_param_selection(self.dictvec.fit_transform(self.features_list),self.le.fit_transform(self.label)))

        self.log_reg.fit(self.dictvec.fit_transform(self.features_list),self.le.fit_transform(self.label))


    def train_combined(self, tweets_and_labels):
        for tweet in tweets_and_labels:
            self.label.append(tweet[1])
            self.features_list.append(self.feature_extractor(tweet[0]))
        #print(self.features_list)

        #cross_Val(self.dictvec.fit_transform(self.features_list),self.le.fit_transform(self.label),10)
        #print(svc_param_selection(self.dictvec.fit_transform(self.features_list),self.le.fit_transform(self.label),10))


        #self.log_reg.fit(self.dictvec.fit_transform(self.features_list),self.le.fit_transform(self.label))


    def predict(self,tweets_and_labels):
        total_count = 0.0
        correct_count = 0
        predicted_val_final=[]
        for tweet in tweets_and_labels:
            total_count += 1
            #predicted_val= self.le.classes_[self.log_reg.predict(self.dictvec.transform(self.feature_extractor(tweet[0])))][0]
            #score = self.model.evaluate(X_test, y_test,batch_size=256, verbose=1)
            predicted_val = self.le.classes_[self.log_reg.predict(self.dictvec.transform(self.feature_extractor(tweet[0])))][0]
            predicted_val_final.append(predicted_val)
            if predicted_val==tweet[1]:
                correct_count+=1
        accuracy=correct_count/total_count


        with open("V-oc_en_pred.txt","w") as fw:
            for item in predicted_val_final:
                fw.write(str(item)+'\n')

        return accuracy

    def predict_test(self,tweets_and_labels):
        predicted_val_final=[]
        for tweet in tweets_and_labels:
            predicted_val= self.le.classes_[self.log_reg.predict(self.dictvec.transform(self.feature_extractor(tweet[0])))][0]
            predicted_val_final.append(predicted_val)

        with open("V-oc_en_pred.txt","w") as fw:
            for item in predicted_val_final:
                fw.write(str(item)+'\n')

        return None


train_dataset = read_parse("Data/2018-Valence-oc-En-train.txt")
classifier = Classifier()
classifier.train(train_dataset)
#dev_dataset= read_parse("Data/2018-Valence-oc-En-dev.txt")
#accuracy=classifier.predict(dev_dataset)
test_dataset = read_parse_Test("Data/2018-Valence-oc-En-test.txt")
accuracy=classifier.predict_test(test_dataset)
msg = "\n{:.1%} accuracy on development data"

#combined_dataset=read_parse("Data/2018-Valence-oc-En-combined.txt")
#classifier.train_combined(combined_dataset)

print(msg.format(accuracy))