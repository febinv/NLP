from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Sequence, Text, Union
import itertools
from collections import deque,defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn import preprocessing

@dataclass()
class Dep:
    """A word in a dependency tree.
    The fields are defined by https://universaldependencies.org/format.html.
    """
    id: Text
    form: Union[Text, None]
    lemma: Union[Text, None]
    upos: Text
    xpos: Union[Text, None]
    feats: Sequence[Text]
    head: Union[Text, None]
    deprel: Union[Text, None]
    deps: Sequence[Text]
    misc: Union[Text, None]


def read_conllu(path: Text) -> Iterator[Sequence[Dep]]:
    """Reads a CoNLL-U format file into sequences of Dep objects.
    The CoNLL-U format is described in detail here:
    https://universaldependencies.org/format.html
    A few key highlights:
    * Word lines contain 10 fields separated by tab characters.
    * Blank lines mark sentence boundaries.
    * Comment lines start with hash (#).
    Each word line will be converted into a Dep object, and the words in a
    sentence will be collected into a sequence (e.g., list).
    :return: An iterator over sentences, where each sentence is a sequence of
    words, and each word is represented by a Dep object.
    """

    #Initializing variables
    sentence=[]
    words=[]
    none_val=None
    #Reading the file
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            #Parsing sentence by sentence
            if not line.strip().startswith('#') and line.strip():
                lines=line.strip()
                id=lines.split('\t')[0]
                if lines.split('\t')[1].startswith('_'):
                    form=None
                else:
                    form=lines.split('\t')[1]
                if lines.split('\t')[2].startswith('_'):
                    lemma=None
                else:
                    lemma=lines.split('\t')[2]
                upos=lines.split('\t')[3]
                if lines.split('\t')[4].startswith('_'):
                    xpos=None
                else:
                    xpos=lines.split('\t')[4]
                if lines.split('\t')[5].startswith('_'):
                    feats=[]
                else:
                    feats=lines.split('\t')[5].split('|')
                if lines.split('\t')[6].startswith('_'):
                    head=None
                else:
                    head=lines.split('\t')[6]
                if lines.split('\t')[7].startswith('_'):
                    deprel=None
                else:
                    deprel=lines.split('\t')[7]
                deps=lines.split('\t')[8].split('|')
                if lines.split('\t')[9].startswith('_'):
                    misc=None
                else:
                    misc=lines.split('\t')[9]
                words.append(Dep(id,form,lemma,upos,xpos,feats,head,deprel,deps,misc))
            #Append at end of Sentence
            elif len(line.strip())==0:
                sentence.append(words)
                words=[]
    return iter(sentence)



class Action(Enum):
    """An action in an "arc standard" transition-based parser."""
    SHIFT = 1
    LEFT_ARC = 2
    RIGHT_ARC = 3


def parse(deps: Sequence[Dep],
          get_action: Callable[[Sequence[Dep], Sequence[Dep]], Action]) -> None:
    """Parse the sentence based on "arc standard" transitions.
    Following the "arc standard" approach to transition-based parsing, this
    method creates a stack and a queue, where the input Deps start out on the
    queue, are moved to the stack by SHIFT actions, and are combined in
    head-dependent relations by LEFT_ARC and RIGHT_ARC actions.
    This method does not determine which actions to take; those are provided by
    the `get_action` argument to the method. That method will be called whenever
    the parser needs a new action, and then the parser will perform whatever
    action is returned. If `get_action` returns an invalid action (e.g., a
    SHIFT when the queue is empty), an arbitrary valid action will be taken
    instead.
    This method does not return anything; it modifies the `.head` field of the
    Dep objects that were passed as input. Each Dep object's `.head` field is
    assigned the value of its head's `.id` field, or "0" if the Dep object is
    the root.
    :param deps: The sentence, a sequence of Dep objects, each representing one
    of the words in the sentence.
    :param get_action: a function or other callable that takes the parser's
    current stack and queue as input, and returns an "arc standard" action.
    :return: Nothing; the `.head` fields of the input Dep objects are modified.
    """
    #Initializing
    deq=deque()
    stack=[]
    arcs=[]

    #Appending all the words to the deque
    for dep in deps:
        deq.append(dep)

    #Iterating one by one
    while True:
        try:
            #Calling the action to performed
            action=get_action(stack,deq)
            # print(action)
            # print(stack)
            # print(deq)

            #Done Processing
            if (len(deq)==0 and len(stack)==1):
                break
            #Shift
            elif action== Action.SHIFT and len(deq)>0 or len(stack)<2:
                stack.append(deq.popleft())

            #Left-arc
            elif action == Action.LEFT_ARC and len(stack)>0:
                child=stack.pop(-2)
                child.head=stack[-1].id
                arcs.append(child)
            #Right-arc
            elif action == Action.RIGHT_ARC and len(stack)>0:
                child=stack.pop()
                child.head=stack[-1].id
                arcs.append(child)

        except StopIteration:
            break
    #Last element on the stack
    root=stack.pop()
    root.head='0'
    arcs.append(root)
    #sort as per the dep ids
    arcs.sort(key=lambda x: float(x.id))
    #assign to original deps
    deps=arcs


class Oracle:
    def __init__(self, deps: Sequence[Dep]):
        """Initializes an Oracle to be used for the given sentence.
        Minimally, it initializes a member variable `actions`, a list that
        will be updated every time `__call__` is called and a new action is
        generated.
        Note: a new Oracle object should be created for each sentence; an
        Oracle object should not be re-used for multiple sentences.
        :param deps: The sentence, a sequence of Dep objects, each representing
        one of the words in the sentence.
        """
        #Initialize the requried variables
        self.actions=[]
        self.features_list=[]
        #Maintain assigned dependencies
        self.connected_childs = set()
        #Maintain all dependents
        self.dependencies=defaultdict(set)
        #Assign all dependents
        for dep in deps:
            self.dependencies[dep.head].add(dep.id)


    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]) -> Action:
        """Returns the Oracle action for the given "arc standard" parser state.
        The oracle for an "arc standard" transition-based parser inspects the
        parser state and the reference parse (represented by the `.head` fields
        of the Dep objects) and:
        * Chooses LEFT_ARC if it produces a correct head-dependent relation
          given the reference parse and the current configuration.
        * Otherwise, chooses RIGHT_ARC if it produces a correct head-dependent
          relation given the reference parse and all of the dependents of the
          word at the top of the stack have already been assigned.
        * Otherwise, chooses SHIFT.
        The chosen action should be both:
        * Added to the `actions` member variable
        * Returned as the result of this method
        Note: this method should only be called on parser state based on the Dep
        objects that were passed to __init__; it should not be used for any
        other Dep objects.
        :param stack: The stack of the "arc standard" transition-based parser.
        :param queue: The queue of the "arc standard" transition-based parser.
        :return: The action that should be taken given the reference parse
        (the `.head` fields of the Dep objects).
        """

        #Left-arc condition
        if len(stack)>=2 and stack[-1].id==stack[-2].head and len(self.dependencies[stack[-2].id] - self.connected_childs) < 1:
                self.connected_childs.add(stack[-2].id)
                self.actions.append(Action.LEFT_ARC)
                self.features_list.append(feature_extraction(stack,queue))
                return Action.LEFT_ARC

        #Right-arch condition
        elif len(stack)>=2 and stack[-1].head==stack[-2].id and len(self.dependencies[stack[-1].id] - self.connected_childs) < 1:
                self.actions.append(Action.RIGHT_ARC)
                self.connected_childs.add(stack[-1].id)
                self.features_list.append(feature_extraction(stack,queue))
                return Action.RIGHT_ARC

        #Ensure Queue has elements for Shift
        elif len(queue)>0:
            self.actions.append(Action.SHIFT)
            self.features_list.append(feature_extraction(stack,queue))
            return Action.SHIFT

        else:
            #ensureq queue has elements for shift
            if len(queue)>0:
                self.actions.append(Action.SHIFT)
                self.features_list.append(feature_extraction(stack,queue))
                return Action.SHIFT
            #dummy action
            elif not (len(queue)==0 and len(stack)==1):
                self.features_list.append(feature_extraction(stack,queue))
                self.actions.append(Action.LEFT_ARC)
                return Action.LEFT_ARC



class Classifier:
    def __init__(self, parses: Iterator[Sequence[Dep]]):
        """Trains a classifier on the given parses.
        There are no restrictions on what kind of classifier may be trained,
        but a typical approach would be to
        1. Define features based on the stack and queue of an "arc standard"
           transition-based parser (e.g., part-of-speech tags of the top words
           in the stack and queue).
        2. Apply `Oracle` and `parse` to each parse in the input to generate
           training examples of parser states and oracle actions. It may be
           helpful to modify `Oracle` to call the feature extraction function
           defined in 1, and store the features alongside the actions list that
           `Oracle` is already creating.
        3. Train a machine learning model (e.g., logistic regression) on the
           resulting features and labels (actions).
        :param parses: An iterator over sentences, where each sentence is a
        sequence of words, and each word is represented by a Dep object.
        """
        #Initialize
        self.label=[]
        self.features_list=[]
        self.le = preprocessing.LabelEncoder()
        self.dictvec = DictVectorizer()
        self.log_reg = linear_model.LogisticRegression(multi_class='multinomial',solver = 'sag',max_iter=200)

        #Loop through parses one by one
        for i, deps in enumerate(parses):
            oracle = Oracle(deps)
            parse(deps, oracle)
            #Append actions and features
            self.label.extend([e.value for e in oracle.actions])
            self.features_list.extend(oracle.features_list)
        #Transform the features list and labels and then Fit the classifier
        self.log_reg.fit(self.dictvec.fit_transform(self.features_list),self.le.fit_transform(self.label))

    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]) -> Action:
        """Predicts an action for the given "arc standard" parser state.
        There are no restrictions on how this prediction may be made, but a
        typical approach would be to convert the parser state into features,
        and then use the machine learning model (trained in `__init__`) to make
        the prediction.
        :param stack: The stack of the "arc standard" transition-based parser.
        :param queue: The queue of the "arc standard" transition-based parser.
        :return: The action that should be taken.
        """

        features={}

        #Extract the features
        features=feature_extraction(stack,queue)

        #Make the prediction
        predicted_val= [self.le.classes_[self.log_reg.predict(self.dictvec.transform(features))][0]]

        #Choose which action based on predicted value
        if predicted_val == [1]:
             return Action.SHIFT
        elif predicted_val == [2]:
             return Action.LEFT_ARC
        else:
             return Action.RIGHT_ARC

def test_parse():
    # consider a specific sentence from the training data

    # # sent_id = weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0026
    # # text = The future president joined the Guard in May 1968.
    # 1	The	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
    # 2	future	future	ADJ	JJ	Degree=Pos	3	amod	3:amod	_
    # 3	president	president	NOUN	NN	Number=Sing	4	nsubj	4:nsubj	_
    # 4	joined	join	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	0	root	0:root	_
    # 5	the	the	DET	DT	Definite=Def|PronType=Art	6	det	6:det	_
    # 6	Guard	Guard	PROPN	NNP	Number=Sing	4	obj	4:obj	_
    # 7	in	in	ADP	IN	_	8	case	8:case	_
    # 8	May	May	PROPN	NNP	Number=Sing	4	obl	4:obl:in	_
    # 9	1968	1968	NUM	CD	NumType=Card	8	nummod	8:nummod	SpaceAfter=No
    # 10	.	.	PUNCT	.	_	4	punct	4:punct	_
    parses = read_conllu("UD_English-EWT/en_ewt-ud-train.conllu")
    [deps] = itertools.islice(parses, 352, 353)

    # clear out all the head information
    orig_heads = clear_heads(deps)

    # run the parser with the oracle list of actions
    parse(deps, IterActions([
        Action.SHIFT,
        Action.SHIFT,
        Action.SHIFT,
        Action.LEFT_ARC,
        Action.LEFT_ARC,
        Action.SHIFT,
        Action.LEFT_ARC,
        Action.SHIFT,
        Action.SHIFT,
        Action.LEFT_ARC,
        Action.RIGHT_ARC,
        Action.SHIFT,
        Action.SHIFT,
        Action.LEFT_ARC,
        Action.SHIFT,
        Action.RIGHT_ARC,
        Action.RIGHT_ARC,
        Action.SHIFT,
        Action.RIGHT_ARC,
    ]))

    # make sure that the original heads have been restored by the parser

    assert [dep.head for dep in deps] == orig_heads

def clear_heads(deps: Sequence[Dep]):
    """Removes all head information (.head, .deprel, .deps) from the sentence"""
    heads = []
    for dep in deps:
        heads.append(dep.head)
        dep.head = None
        dep.deprel = None
        dep.deps = None
    return heads

#Convert None to space
def xstr(data):
      if data is None:
         return ' '
      return data

#Function to extract features
def feature_extraction(stack,queue):
    features={}

    #Extracting top 3rd stack word features
    if len(stack)>3:
        features["pos_stack_3"]=xstr(stack[-3].upos)
        features["form_stack_3"]=xstr(stack[-3].form)
        features["xpos_stack_3"]=xstr(stack[-3].xpos)
        features["lemma_stack_3"]=xstr(stack[-3].lemma)
    else:
        features["pos_stack_3"]=' '
        features["form_stack_3"]=' '
        features["xpos_stack_3"]=' '
        features["lemma_stack_3"]=' '

    #Extracting top 2nd stack word features
    if len(stack)>2:
        features["pos_stack_2"]=xstr(stack[-2].upos)
        features["form_stack_2"]=xstr(stack[-2].form)
        features["xpos_stack_2"]=xstr(stack[-2].xpos)
        features["lemma_stack_2"]=xstr(stack[-2].lemma)
        #features["id_stack_2"]=xstr(stack[-2].id)
        #features["head_stack_2"]=xstr(stack[-2].head)
    else:
        features["pos_stack_2"]=' '
        features["form_stack_2"]=' '
        features["xpos_stack_2"]=' '
        features["lemma_stack_2"]=' '
        #features["id_stack_2"]=' '
        #features["head_stack_2"]=' '

    #Extracting top stack word features
    if len(stack)>1:
        features["pos_stack_1"]=xstr(stack[-1].upos)
        features["form_stack_1"]=xstr(stack[-1].form)
        features["xpos_stack_1"]=xstr(stack[-1].xpos)
        features["lemma_stack_1"]=xstr(stack[-1].lemma)
        #features["id_stack_1"]=xstr(stack[-1].id)
        #features["head_stack_1"]=xstr(stack[-1].head)
    else:
        features["pos_stack_1"]=' '
        features["form_stack_1"]=' '
        features["xpos_stack_1"]=' '
        features["lemma_stack_1"]=' '
        #features["id_stack_1"]=' '
        #features["head_stack_1"]=' '

    #Extracting 3rd front queue word features
    if len(queue)>3:
        features["pos_queue_3"]=xstr(queue[2].upos)
        features["form_queue_3"]=xstr(queue[2].form)
        features["xpos_queue_3"]=xstr(queue[2].xpos)
        features["lemma_queue_3"]=xstr(queue[2].lemma)


    else:
        features["pos_queue_3"]=' '
        features["form_queue_3"]=' '
        features["xpos_queue_3"]=' '
        features["lemma_queue_3"]=' '

    #Extracting 2nd front queue word features
    if len(queue)>2:
        features["pos_queue_2"]=xstr(queue[1].upos)
        features["form_queue_2"]=xstr(queue[1].form)
        features["xpos_queue_2"]=xstr(queue[1].xpos)
        features["lemma_queue_2"]=xstr(queue[1].lemma)
    else:
        features["pos_queue_2"]=' '
        features["form_queue_2"]=' '
        features["xpos_queue_2"]=' '
        features["lemma_queue_2"]=' '

    #Extracting front queue word features
    if len(queue)>1:
        features["pos_queue_1"]=xstr(queue[0].upos)
        features["form_queue_1"]=xstr(queue[0].form)
        features["xpos_queue_1"]=xstr(queue[0].xpos)
        features["lemma_queue_1"]=xstr(queue[0].lemma)

    else:
        features["pos_queue_1"]=' '
        features["form_queue_1"]=' '
        features["xpos_queue_1"]=' '
        features["lemma_queue_1"]=' '

    #Concatenating pos and form feature for stack
    if len(stack)>1 and len(queue)>1:
        features["stack1_pos_with_queue1form"]=xstr(stack[-1].upos)+xstr(queue[0].form)
        features["stack1_xpos_with_queue1lemma"]=xstr(stack[-1].xpos)+xstr(queue[0].lemma)
    else:
        features["stack1_pos_with_queue1form"]=' '
        features["stack1_xpos_with_queue1lemma"]=' '

    #Concatenating pos and form feature for queue
    if len(stack)>2 and len(queue)>2:
        features["stack2_pos_with_queue2form"]=xstr(stack[-2].upos)+xstr(queue[1].form)
        features["stack2_xpos_with_queue2lemma"]=xstr(stack[-2].xpos)+xstr(queue[1].lemma)
        #features["stacktopword_stacknexttopword"]=xstr(stack[-1].form)+xstr(stack[-2].form)

    else:
        features["stack2_pos_with_queue2form"]=' '
        features["stack2_xpos_with_queue2lemma"]=' '
        #features["stacktopword_stacknexttopword"]=' '

    if len(stack)>3 and len(queue)>3:
        features["stack3_pos_with_queue3form"]=xstr(stack[-3]).upos+xstr(queue[2].form)
        #features["stack3_xpos_with_queue3lemma"]=xstr(stack[-3]).xpos+xstr(queue[2].lemma)
    else:
        features["stack3_pos_with_queue3form"]=' '
        #features["stack3_xpos_with_queue3lemma"]=' '

    features["stack_len"]=len(stack)
    features["queue_len"]=len(queue)

    return features.copy()

class IterActions:
    """A class for feeding a list of actions, one at a time, to a parser"""

    def __init__(self, actions: Sequence[Action]):
        self.actions_iter = iter(actions)

    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]):
        return next(self.actions_iter)