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
def tokenizeReviews(id, csv_dict, vocab_dict, count):
    tokenizer = Tokenizer()
    # Index 4 is the reviewText
    tokens = tokenizer.tokenize(csv_dict.get(id)[4])
    if vocab_dict == None and count == None:
        return tokens
    else:
        vocab = list(vocab_dict.keys())
        for i in range(len(tokens)):
            if tokens[i] not in vocab or count > vocab_dict.get(tokens[i]):
                tokens[i] = "<OOV>"
        return tokens
# Step 1.3 Use GenSim word2vec to train a 128-dimensional word2vec model utilizing only the training data.
def buildVocab(csv_dict):
    words = []
    vocab_dict = dict()
    idList = list(csv_dict.keys())
    for id in idList:
        words = tokenizeReviews(id, csv_dict, None, None)
        for word in words:
            if word in vocab_dict:
                vocab_dict.update({word : vocab_dict.get(word) + 1})
            else:
                vocab_dict.update({word : 1})
    return vocab_dict
def genWord2Vec(csv_dict, vocab_dict, count):
    sentences = []
    idList = list(csv_dict.keys())
    for id in idList:
        sentences.append(tokenizeReviews(id, csv_dict, vocab_dict, count))
    
    model = gensim.models.Word2Vec(sentences=sentences, 
                                       size=128, 
                                       min_count=count)
    return model
# Step 1.4 Extract features: utilizing your word2vec model, get a representation for each word per review. 
# Then average these embeddings (using mean) into a single 128-dimensional dense set of features.
def extractMeanFeaturesVector(id, csv_dict, model):
    # Returns a single 128-dimensional averaged set of features in reviewText per word per id specified
    wordsInReview = tokenizeReviews(id, csv_dict, None, None)
    wordVector = []
    for word in wordsInReview:
        if word in model.wv.vocab:
            wordVector += [model.wv[word]]
        else:
            wordVector += [model.wv["<OOV>"]]
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
    #print(len(reviewVectors[0]))
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
    model = Ridge(alpha=alpha, solver='auto')
    model.fit(X_train, y_train)
    return model
def fixOutput(y_pred, min, max):
    for i in range(len(y_pred)):
        if y_pred[i] > max:
            y_pred[i] = max
        if y_pred[i] < min:
            y_pred[i] = min
    return y_pred
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
    oldMAE = sys.maxsize
    optimalAlpha = 0
    for i in range(-20, 10):
        model = trainTestRater(features, ratings, X_subtrain, X_dev, y_subtrain, y_dev, math.pow(10, i))
        y_pred = model.predict(X_dev)
        y_pred = fixOutput(y_pred, 1, 5)
        #calculate MAE
        newMAE = mean_absolute_error(y_dev, y_pred)
        #newacc = checkVectorAcc(y_dev, y_pred)
        # The best MAE is 0
        if newMAE < oldMAE:
            optimalAlpha = math.pow(10, i)
            oldMAE = newMAE
            #print(i)
    bestmodel = trainTestRater(features, ratings, X_subtrain, X_dev, y_subtrain, y_dev, optimalAlpha)
    pearsons = scipy.stats.pearsonr(y_dev, bestmodel.predict(X_dev))
    #print(bestmodel.predict(X_dev))
    #print(y_dev)
    print("Best Training Model")
    print("Mean Absolute Error is : " + str((oldMAE)))
    print("Pearson Correlation is : " + str(pearsons))
    print("Optimal Alpha is : " + str(optimalAlpha))
    return bestmodel
def checkpointOne(train_csv, test_csv):
    #Training
    trainVocab_dict = buildVocab(train_csv)
    count = 2
    #The train model has an <OOV> index
    train_model = genWord2Vec(train_csv, trainVocab_dict, count)
    Xs_train = extractFeaturesVectors(train_csv, train_model)
    ys_train = extractYVector(train_csv)
    X_train, X_test, y_train, y_test = train_test_split(np.array(Xs_train),
                                                        np.array(ys_train),
                                                        test_size=0.20)
    rater = trainRater(X_train, y_train)
    #Testing
    Xs_test = extractFeaturesVectors(test_csv, train_model)
    #print(X_test[0])
    #ys_test = extractYVector(test_csv)
    ys_pred = fixOutput(rater.predict(Xs_test), 1, 5)
    pred_dict = dict()
    idList = list(test_csv.keys())
    for i in range(len(idList)):
        pred_dict.update({idList[i] : ys_pred[i]})
    #print(pred_dict)
    ys_true = extractYVector(test_csv)
    MAE = mean_absolute_error(ys_true, ys_pred)
    print("Testing Results")
    print("Mean Absolute Error is : " + str((MAE)))
    print("For ID 548\nPredicted Value is " + str(pred_dict.get("548")) + "\nTrue Value is " + str(test_csv.get("548")[0]))
    print("For ID 4258\nPredicted Value is " + str(pred_dict.get("4258")) + "\nTrue Value is " + str(test_csv.get("4258")[0]))
    print("For ID 4766\nPredicted Value is " + str(pred_dict.get("4766")) + "\nTrue Value is " + str(test_csv.get("4766")[0]))
    print("For ID 5800\nPredicted Value is " + str(pred_dict.get("5800")) + "\nTrue Value is " + str(test_csv.get("5800")[0]))

    #Training
    testVocab_dict = buildVocab(test_csv)
    count = 2
    #The train model has an <OOV> index
    test_model = genWord2Vec(test_csv, testVocab_dict, count)
    checkpointTwo(train_csv, test_csv, train_model, test_model)
    return rater
# Step 2.1 Grab the user_ids for both datasets.
def getUserID_Dict(csv_dict):
    idList = list(csv_dict.keys())
    userID_Dict = dict()
    for id in idList:
        user_id = csv_dict.get(id)[3]
        if user_id in userID_Dict:
            temp = userID_Dict.get(user_id)
            temp += [id]
            userID_Dict.update({user_id : temp})
        else:
            userID_Dict.update({user_id : [id]})
    return userID_Dict
# Step 2.2 For each user, treat their training data as "background" 
# (i.e. the data from which to learn the user factors) in order to learn user factors: 
# average all of their word2vec features over the training data to treat as 128-dimensional "user-language representations".
def extractEmbeddingsVectors(id, csv_dict, model):
    # Returns a single 128-dimensional set of features in reviewText per word per id specified
    wordsInReview = tokenizeReviews(id, csv_dict, None, None)
    wordVector = []
    for word in wordsInReview:
        if word in model.wv.vocab:
            wordVector += [model.wv[word]]
        else:
            wordVector += [model.wv["<OOV>"]]
    wordVector = np.array(wordVector)
    #print(len(wordVector[0]))
    return wordVector
# Step 1.3 Run PCA the matrix of user-language representations to reduce down to just three factors. 
# Save the 3 dimensional transformation matrix (V) so that you may apply it to new data 
# (i.e. the trial or test set when predicting -- when predicting you should not run PCA again; only before training).
# Returns a dict of userIDs and corresponding user factors
def genPCA(userID_List, reviewVectors):
    pca = PCA(n_components=3)
    reviewMatrix = list(reviewVectors.values())
    result = pca.fit_transform(reviewMatrix)
    userID_PCADict = dict()
    for i in range(len(userID_List)) :
        userID_PCADict.update({userID_List[i] : result[i]})
    return userID_PCADict

def checkpointTwo(train_csv, test_csv, train_model, test_model):
    train_IDList = list(train_csv.keys())
    test_IDList = list(test_csv.keys())
    full_IDList = train_IDList + test_IDList

    trainUserID_IDDict = getUserID_Dict(train_csv)
    testUserID_IDDict = getUserID_Dict(test_csv)

    userID_List = list(trainUserID_IDDict.keys()) + list(testUserID_IDDict.keys())
    #print(len(userID_List))
    #print(len(trainUserID_IDDict))
    #print(len(testUserID_IDDict))
    reviewVectors = dict()

    id_featureDict = dict() #Review embeddings corresponding to each review id, ie a list of word vectors for each reviewText
    Xs_train = extractFeaturesVectors(train_csv, train_model)
    for i in range(len(full_IDList)):
        if full_IDList[i] in train_IDList and full_IDList[i] not in id_featureDict:
            id_featureDict.update({full_IDList[i] : extractEmbeddingsVectors(full_IDList[i], train_csv, train_model)})
        elif full_IDList[i] in train_IDList and full_IDList[i] in id_featureDict:
            original = id_featureDict.get(full_IDList[i])
            id_featureDict.update({full_IDList[i] : extractEmbeddingsVectors(full_IDList[i], train_csv, train_model) + original})
        
        elif full_IDList[i] in test_IDList and full_IDList[i] not in id_featureDict:
            id_featureDict.update({full_IDList[i] : extractEmbeddingsVectors(full_IDList[i], test_csv, test_model)})
        elif full_IDList[i] in test_IDList and full_IDList[i] in id_featureDict:
            original = id_featureDict.get(full_IDList[i])
            id_featureDict.update({full_IDList[i] : extractEmbeddingsVectors(full_IDList[i], test_csv, test_model) + original})
        else:
            print("Never reaches here")
    for user_id in userID_List:
        #List of review ids that correspond to user_ids
        temp = list()
        if user_id in trainUserID_IDDict and user_id in testUserID_IDDict:
            temp = trainUserID_IDDict.get(user_id) + testUserID_IDDict.get(user_id)
        elif user_id in trainUserID_IDDict:
            temp = trainUserID_IDDict.get(user_id)
        elif user_id in testUserID_IDDict:
            temp = testUserID_IDDict.get(user_id)
        else:
            print("Never reaches here")
        reviews = list()
        for id in temp:
            reviews += [np.mean(id_featureDict.get(id),axis=0).tolist()]
        reviews = np.array(reviews)
        #Takes the average of the entire word embeddings of all the reviews related to the user_ids
        #average all of their word2vec features over the training data to treat as 128-dimensional "user-language representations".
        reviews = np.mean(reviews, axis=0)
        reviewVectors.update({user_id : reviews})
    #print(reviewVectors.get("11dbc98a59307be9e3faaad03389a0e9AF14"))

    userID_PCADict = genPCA(userID_List, reviewVectors)

    Xs_train = list()
    for id in train_IDList:
        embeddingsV = np.mean(id_featureDict.get(id),axis=0)
        #print(embeddingsV.shape)
        userFactors = list()
        # Doesn't matter where you get the user_id from, either train or test csv will sufficise if it exists in either one.
        if id in train_IDList:
            userFactors = userID_PCADict.get(train_csv.get(id)[3])
        elif id in test_IDList:
            userFactors = userID_PCADict.get(test_csv.get(id)[3])
        else:
            print("Never goes here")
        feature0 = embeddingsV*(userFactors[0])
        feature1 = embeddingsV*(userFactors[1])
        feature2 = embeddingsV*(userFactors[2])
        feature = np.concatenate((embeddingsV, feature0, feature1, feature2)).flatten()
        Xs_train += [feature]

    # The features go embeds; embeds*f1; embeds*f2; embeds*f3
    print(len(Xs_train))
    ys_train = extractYVector(train_csv)
    X_train, X_test, y_train, y_test = train_test_split(np.array(Xs_train),
                                                        np.array(ys_train),
                                                        test_size=0.20)
    
    rater = trainRater(X_train, y_train)

    print("Hello")
    '''
    #Testing
    Xs_test = extractFeaturesVectors(test_csv, train_model)
    #print(X_test[0])
    #ys_test = extractYVector(test_csv)
    ys_pred = rater.predict(Xs_test)
    pred_dict = dict()
    idList = list(test_csv.keys())
    for i in range(len(idList)):
        pred_dict.update({idList[i] : ys_pred[i]})
    '''
    return userID_PCADict
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