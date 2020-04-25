#Darren Kong
#CSE354 HW 3
import  re, numpy as np, sklearn, scipy.stats, pandas, csv, gensim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge #just like L2 regularized logistic regression
from sklearn.decomposition import PCA #just like L2 regularized logistic regression
import scipy.stats as ss #for distributions
from gensim.models import Word2Vec
from happiestfuntokenizing.happiestfuntokenizing import Tokenizer
import sys, math
from collections import Counter, defaultdict

# Step 1.1 Read reviews and ratings from the file
def preparecsv(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        id_dict = dict()
        for row in reader:
            unique_id = row['id']
            # Special Token <e> for empty reviews
            if row['reviewText'] == "":
                id_dict.update({unique_id : [row['rating'], row['vote'], row['asin'], row['user_id'], "<s> <e> </s>"]})
            else:
                id_dict.update({unique_id : [row['rating'], row['vote'], row['asin'], row['user_id'], "<s>" + row['reviewText'] + "</s>"]})
        #print(id_dict)
        return id_dict
# Step 1.2 Tokenize the file.
def tokenizeReviews(id, csv_dict):
    tokenizer = Tokenizer()
    # Index 4 is the reviewText
    tokens = tokenizer.tokenize(csv_dict.get(id)[4])
    return tokens
# Step 1.3 Use GenSim word2vec to train a 128-dimensional word2vec model utilizing only the training data.
def genWord2Vec(csv_dict):
    sentences = []
    idList = list(csv_dict.keys())
    for id in idList:
        sentences.append(tokenizeReviews(id, csv_dict))

    model = gensim.models.Word2Vec(sentences=sentences, 
                                       size=128, 
                                       min_count=1)
    return model
# Step 1.4 Extract features: utilizing your word2vec model, get a representation for each word per review. 
# Then average these embeddings (using mean) into a single 128-dimensional dense set of features.
def extractMeanFeaturesVector(id, csv_dict, model):
    # Returns a single 128-dimensional set of features in reviewText per word per id specified
    wordsInReview = tokenizeReviews(id, csv_dict)
    wordVector = []
    for word in wordsInReview:
        if word in model.wv.vocab:
            wordVector += [model.wv[word]]
        #else:
            # OOV is considered neutral
        #    wordVector += [[0]*128]
    wordVector = np.array(wordVector)
    # Gets the mean vector
    wordVector = wordVector.mean(axis=0)
    #print(wordVector)
    return wordVector
# Step 1.5 Build a rating predictor using L2 *linear* regression (can use the SKLearn Ridge class) with word2vec features.
def extractFeaturesVectors(csv_dict, model):
    idList = list(csv_dict.keys())
    # Paranoia
    idList.sort()
    reviewVectors = list()
    for id in idList:
        reviewVectors += [extractMeanFeaturesVector(id, csv_dict, model)]
    return reviewVectors
def extractYVector(csv_dict):
    idList = list(csv_dict.keys())
    # Paranoia
    idList.sort()
    yVector = list()
    for id in idList:
        # Ratings is at index 0 and needed to be converted from string to float
        yVector.append(float(csv_dict.get(id)[0]))
    #print(len(yVector))
    return yVector

def checkVectorAcc(theoretical, experimental):
    # The vector lengths must match for this to work properly
    incorrectCntr = 0
    for i in range(len(theoretical)):
        if theoretical[i] != experimental[i]:
            incorrectCntr += 1
    return 1-(incorrectCntr/len(theoretical))

def trainTestRater(X_train, y_train, X_subtrain, X_dev, y_subtrain, y_dev, alpha):
    #inputs: features: feature vectors (i.e. X)
    #        adjs: whether adjective or not: [0, 1] (i.e. y)
    #output: model -- a trained sklearn.linear_model.LogisticRegression object
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model
# Step 1.6 Run the predictor on the second set of data 
# and print both the mean absolute error and then Pearson correlation 
# between the predictions and the dev set.  
def trainRater(features, ratings):
    #inputs: features: feature vectors (i.e. X)
    #        ratings: ratings value [0, 1, 2, 3, 4, 5] (i.e. y)
    #output: model -- a trained sklearn.linear_model.Ridge object
    X_subtrain, X_dev, y_subtrain, y_dev = train_test_split(features,
                                                            ratings,
                                                            test_size=0.10, random_state = 42)
    oldMAE = 0
    optimalAlpha = 0
    for i in range(-10, 10):
        model = trainTestRater(features, ratings, X_subtrain, X_dev, y_subtrain, y_dev, math.pow(10, i))
        y_pred = model.predict(X_dev)
        #calculate MAE
        newMAE = mean_absolute_error(y_dev, y_pred)
        #newacc = checkVectorAcc(y_dev, y_pred)
        # The best MAE is 0
        if newMAE <= oldMAE:
            optimalAlpha = math.pow(10, i)
            oldMAE = newMAE
    bestmodel = trainTestRater(features, ratings, X_subtrain, X_dev, y_subtrain, y_dev, optimalAlpha)
    pearsons = scipy.stats.pearsonr(y_dev, bestmodel.predict(X_dev))
    #print(bestmodel.predict(X_dev))
    #print(y_dev)
    print("Mean Absolute Error is : " + str((oldMAE)))
    print("Pearson Correlation is : " + str(pearsons))
    print("Optimal Alpha is : " + str(optimalAlpha))
    return bestmodel
def checkpointOne(train_csv, test_csv):
    #Training
    train_model = genWord2Vec(train_csv)
    Xs_train = extractFeaturesVectors(train_csv, train_model)
    ys_train = extractYVector(train_csv)
    X_train, X_test, y_train, y_test = train_test_split(np.array(Xs_train),
                                                        np.array(ys_train),
                                                        test_size=0.20)
    rater = trainRater(X_train, y_train)
    #Testing
    Xs_test = extractFeaturesVectors(test_csv, train_model)
    #ys_test = extractYVector(test_csv)
    ys_pred = rater.predict(Xs_test)
    pred_dict = dict()
    idList = list(test_csv.keys())
    for i in range(len(idList)):
        pred_dict.update({idList[i] : ys_pred[i]})
    #print(pred_dict)
    print("For ID 548\nPredicted Value is " + str(pred_dict.get("548")) + "\nTrue Value is " + str(test_csv.get("548")[0]))
    print("For ID 4258\nPredicted Value is " + str(pred_dict.get("4258")) + "\nTrue Value is " + str(test_csv.get("4258")[0]))
    print("For ID 4766\nPredicted Value is " + str(pred_dict.get("4766")) + "\nTrue Value is " + str(test_csv.get("4766")[0]))
    print("For ID 5800\nPredicted Value is " + str(pred_dict.get("5800")) + "\nTrue Value is " + str(test_csv.get("5800")[0]))
    return rater
def main(argv):
    sys.stderr = open('output.txt', 'w')
    if len(argv) != 2:
        print("Needs a train and test file")
    else:
        train_csv = preparecsv(argv[0])
        test_csv = preparecsv(argv[1])
        checkpointOne(train_csv, test_csv)
        return train_csv, test_csv  
        
if __name__== '__main__':
    main(sys.argv[1:])