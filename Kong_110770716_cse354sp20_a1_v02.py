#!/usr/bin/python3

#!/usr/bin/python3
# CSE354 Sp20; Assignment 1 Template v02
##################################################################
_version_ = 0.2

import sys

##################################################################
#1. Tokenizer

import re #python's regular expression package
import numpy
import math
def tokenize(sent):
    #input: a single sentence as a string.
    #output: a list of each "word" in the text
    # must use regular expressions

    tokens = []
    #<FILL IN>
    #The second group does the majority of words with hashtags and @'s
    #The first group tackles the abbreviations
    tokenizer = re.compile('(?:[a-zA-Z]\.){2,}|[#@]*[a-zA-Z0-9]*[\'’-]*[a-z]+|[.,!?;]|[A-Z]+|[\n]')
    return tokenizer.findall(sent)


##################################################################
#2. Pig Latinizer

def pigLatinizer(tokens):
    #input: tokens: a list of tokens,
    #output: plTokens: tokens after transforming to pig latin

    plTokens = []
    for i in range(0,len(tokens)):
        if tokens[i].isalpha():
            if not tokens[i][0] in "AEIOUaeiou":
                for j in range(0, len(tokens[i])):
                    if len(tokens[i]) == j+1:
                        temp = tokens[i] + "ay"
                        plTokens.append(temp)
                        break
                    if tokens[i][j] in "AEIOUaeiou":
                        temp = tokens[i][j:] + tokens[i][0:j] + "ay"
                        plTokens.append(temp)
                        break
            else:
                temp = tokens[i] + "way"
                plTokens.append(temp)
        else:
            plTokens.append(tokens[i])
    return plTokens
    

##################################################################
#3. Feature Extractor

import numpy as np

def getFeaturesForTokens(tokens, wordToIndex):
    #input: tokens: a list of tokens,
    #wordToIndex: dict mapping 'word' to an index in the feature list.
    #output: list of lists (or np.array) of k feature values for the given target
    num_words = len(tokens)
    
    featuresPerTarget = list() #holds arrays of feature per word
    #Offset for the feature vector, 2 for number of vowels and number of consonants
    offset = 2 
    #Used for "concatenation" of the one hot vectors
    onehotOffset = len(wordToIndex)
    for targetI in range(num_words):
        vowelctr = 0
        consonantctr = 0
        prevword = 0
        nextword = 0
        for i in range(0, len(tokens[targetI])):
            if tokens[targetI][i] in "aeiouAEIOU":
                vowelctr += 1
            if not tokens[targetI][i] in "aeiouAEIOU" and tokens[targetI].isalpha():
                consonantctr += 1
        if not (targetI == 0 or targetI == num_words-1):
            prevword = wordToIndex.get(tokens[targetI-1].lower(), 0)
            nextword = wordToIndex.get(tokens[targetI+1].lower(), 0)
        #Cases for the beginning and end of the tokens
        if targetI == 0 and num_words > 1:
            nextword = wordToIndex.get(tokens[targetI+1].lower(), 0)
        if targetI == num_words-1 and num_words > 1:
            prevword = wordToIndex.get(tokens[targetI-1].lower(), 0)

        pass
        featureVector = [0]*(len(wordToIndex)*3+offset)
        featureVector[0] = vowelctr
        featureVector[1] = consonantctr
        featureVector[prevword + offset] = 1
        featureVector[wordToIndex.get(tokens[targetI].lower(), 0) + offset + onehotOffset] = 1
        featureVector[nextword + offset + 2*onehotOffset] = 1
        featuresPerTarget.insert(targetI,featureVector)
    return featuresPerTarget #a (num_words x k) matrix


##################################################################
#4. Adjective Classifier

from sklearn.linear_model import LogisticRegression

def trainTestAdjectiveClassifier(X_subtrain, X_dev, y_subtrain, y_dev, c):
    #inputs: features: feature vectors (i.e. X)
    #        adjs: whether adjective or not: [0, 1] (i.e. y)
    #output: model -- a trained sklearn.linear_model.LogisticRegression object
    model = LogisticRegression(C=c, penalty="l1", solver='liblinear')
    model.fit(X_train, y_train)
    return model

def trainAdjectiveClassifier(features, adjs):
    #inputs: features: feature vectors (i.e. X)
    #        adjs: whether adjective or not: [0, 1] (i.e. y)
    #output: model -- a trained sklearn.linear_model.LogisticRegression object
    X_subtrain, X_dev, y_subtrain, y_dev = train_test_split(features,
                                                            adjs,
                                                            test_size=0.10)
    oldacc = 0
    optimalc = 0
    for i in range(0, 10):
        model = trainTestAdjectiveClassifier(X_subtrain, X_dev, y_subtrain, y_dev, math.pow(10, i))
        y_pred = model.predict(X_dev)
        #calculate accuracy:
        newacc = (1 - np.sum(np.abs(y_pred - y_dev))/len(y_dev) )
        if newacc > oldacc:
            optimalc = math.pow(10, i)
            oldacc = newacc
    bestmodel = trainTestAdjectiveClassifier(X_subtrain, X_dev, y_subtrain, y_dev, optimalc)
    print("Optimal C is : " + str(optimalc))
    return bestmodel

##################################################################
##################################################################
## Main and provided complete methods
## Do not edit.
## If necessary, write your own main, but then make sure to replace
## and test with this before you submit.
##
## Note: Tests below will be a subset of those used to test your
##       code for grading.

def getConllTags(filename):
    #input: filename for a conll style parts of speech tagged file
    #output: a list of list of tuples
    #        representing [[[word1, tag1], [word2, tag2]]]
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag=wordtag.strip()
            if wordtag:#still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word,tag))
            else:#new sentence
                wordTagsPerSent.append([])
                sentNum+=1
    return wordTagsPerSent

# Main
if __name__== '__main__':
    print("Initiating test. Version " , _version_)
    #Data for 1 and 2
    testSents = ['I am attending NLP class 2 days a week at S.B.U. this Spring.',
                 "I don't think data-driven computational linguistics is very tough.",
                 '@mybuddy and the drill begins again. #SemStart']

    #1. Test Tokenizer:
    print("\n[ Tokenizer Test ]\n")
    tokenizedSents = []
    for s in testSents:
        tokenizedS = tokenize(s)
        print(s, tokenizedS, "\n")
        tokenizedSents.append(tokenizedS)

    #2. Test Pig Latinizer:
    print("\n[ Pig Latin Test ]\n")
    for ts in tokenizedSents:
        print(ts, pigLatinizer(ts), "\n")
        
    #load data for 3 and 4 the adjective classifier data:
    taggedSents = getConllTags('daily547.conll')

    #3. Test Feature Extraction:
    print("\n[ Feature Extraction Test ]\n")
    #first make word to index mapping: 
    wordToIndex = set() #maps words to an index
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent) #splits [(w, t), (w, t)] into [w, w], [t, t]
            wordToIndex |= set([w.lower() for w in words]) #union of the words into the set
    print("  [Read ", len(taggedSents), " Sentences]")
    #turn set into dictionary: word: index
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)}

    #Next, call Feature extraction per sentence
    sentXs = []
    sentYs = []
    print("  [Extracting Features]")
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            sentXs.append(getFeaturesForTokens(words, wordToIndex)) 
            sentYs.append([1 if t == 'A' else 0 for t in tags])
    #test sentences
    print("\n", taggedSents[5], "\n", sentXs[5], "\n")
    print(taggedSents[192], "\n", sentXs[192], "\n")


    #4. Test Classifier Model Building
    print("\n[ Classifier Test ]\n")
    #setup train/test:
    from sklearn.model_selection import train_test_split
    #flatten by word rather than sent: 
    X = [j for i in sentXs for j in i]
    y= [j for i in sentYs for j in i]
    try: 
        X_train, X_test, y_train, y_test = train_test_split(np.array(X),
                                                            np.array(y),
                                                            test_size=0.20,
                                                            random_state=42)
    except ValueError:
        print("\nLooks like you haven't implemented feature extraction yet.")
        print("[Ending test early]")
        sys.exit(1)
    print("  [Broke into training/test. X_train is ", X_train.shape, "]")
    #Train the model.
    print("  [Training the model]")
    tagger = trainAdjectiveClassifier(X_train, y_train)
    print("  [Done]")
    

    #Test the tagger.
    from sklearn.metrics import classification_report
    #get predictions:
    y_pred = tagger.predict(X_test)
    #compute accuracy:
    leny = len(y_test)
    print(X_test[0])
    print(y_pred)
    if 2 in y_pred:
        print("In")
    print("test n: ", leny)
    acc = np.sum([1 if (y_pred[i] == y_test[i]) else 0 for i in range(leny)]) / leny
    print("Accuracy: %.4f" % acc)
    #print(classification_report(y_test, y_pred, ['not_adj', 'adjective']))