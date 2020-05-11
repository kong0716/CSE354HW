#Darren Kong
#CSE354 HW 3
import  re, numpy as np, sklearn, scipy.stats, pandas, csv, gensim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge #just like L2 regularized logistic regression
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD, IncrementalPCA #just like L2 regularized logistic regression
import scipy.stats as ss #for distributions
from gensim.models import Word2Vec
from happiestfuntokenizing.happiestfuntokenizing import Tokenizer
import sys, math
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random

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
    return wordVector
# Step 1.5 Build a rating predictor using L2 *linear* regression (can use the SKLearn Ridge class) with word2vec features.
def extractFeaturesVectors(csv_dict, model):
    idList = list(csv_dict.keys())
    # Paranoia
    #idList.sort()
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
    model = Ridge(alpha=alpha, solver='auto', random_state=42)
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
    for i in range(-30, 10):
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
    bestmodel = trainTestRater(features, ratings, X_subtrain, X_dev, y_subtrain, y_dev, optimalAlpha)
    pearsons = scipy.stats.pearsonr(y_dev, bestmodel.predict(X_dev))

    print("Best Training Model")
    print("Mean Absolute Error is : " + str((oldMAE)))
    print("Pearson Correlation is : " + str(pearsons))
    print("Optimal Alpha is : " + str(optimalAlpha))
    return bestmodel
def checkpointOne(filename, train_csv, test_csv, sharedTask):
    print("\nStage 1:")
    #Training
    trainVocab_dict = buildVocab(train_csv)
    count = 3
    #The train model has an <OOV> index
    train_model = genWord2Vec(train_csv, trainVocab_dict, count)
    Xs_train = extractFeaturesVectors(train_csv, train_model)
    ys_train = extractYVector(train_csv)
    #X_train, X_test, y_train, y_test = train_test_split(np.array(Xs_train),
    #                                                    np.array(ys_train),
    #                                                    test_size=0.20,
    #                                                    random_state=42)
    rater = trainRater(Xs_train, ys_train)
    if not sharedTask:
        #Testing
        Xs_test = extractFeaturesVectors(test_csv, train_model)
        #ys_test = extractYVector(test_csv)
        ys_pred = fixOutput(rater.predict(Xs_test), 1, 5)
        pred_dict = dict()
        idList = list(test_csv.keys())
        for i in range(len(idList)):
            pred_dict.update({idList[i] : ys_pred[i]})
        ys_true = extractYVector(test_csv)
        MAE = mean_absolute_error(ys_true, ys_pred)
        print("Testing Results")
        print("Mean Absolute Error is : " + str((MAE)))
        if filename == 'food':
            print("For ID 548\nPredicted Value is " + str(pred_dict.get("548")) + "\nTrue Value is " + str(test_csv.get("548")[0]))
            print("For ID 4258\nPredicted Value is " + str(pred_dict.get("4258")) + "\nTrue Value is " + str(test_csv.get("4258")[0]))
            print("For ID 4766\nPredicted Value is " + str(pred_dict.get("4766")) + "\nTrue Value is " + str(test_csv.get("4766")[0]))
            print("For ID 5800\nPredicted Value is " + str(pred_dict.get("5800")) + "\nTrue Value is " + str(test_csv.get("5800")[0]))
        if filename == 'music':
            print("For ID 329\nPredicted Value is " + str(pred_dict.get("329")) + "\nTrue Value is " + str(test_csv.get("329")[0]))
            print("For ID 11419\nPredicted Value is " + str(pred_dict.get("11419")) + "\nTrue Value is " + str(test_csv.get("11419")[0]))
            print("For ID 14023\nPredicted Value is " + str(pred_dict.get("14023")) + "\nTrue Value is " + str(test_csv.get("14023")[0]))
            print("For ID 14912\nPredicted Value is " + str(pred_dict.get("14912")) + "\nTrue Value is " + str(test_csv.get("14912")[0]))

    #Training
    testVocab_dict = buildVocab(test_csv)
    count = 2
    #The train model has an <OOV> index
    test_model = genWord2Vec(test_csv, testVocab_dict, count)
    checkpointTwo(train_csv, test_csv, train_model, sharedTask)
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
def extractEmbeddingsVectors(id, csv_dict, word2vec_model):
    # Returns a single 128-dimensional set of features in reviewText per word per id specified
    wordsInReview = tokenizeReviews(id, csv_dict, None, None)
    wordVector = []
    for word in wordsInReview:
        if word in word2vec_model.wv.vocab:
            wordVector += [word2vec_model.wv[word]]
        else:
            wordVector += [word2vec_model.wv["<OOV>"]]
    wordVector = np.array(wordVector)
    return wordVector
# Step 1.3 Run PCA the matrix of user-language representations to reduce down to just three factors. 
# Save the 3 dimensional transformation matrix (V) so that you may apply it to new data 
# (i.e. the trial or test set when predicting -- when predicting you should not run PCA again; only before training).
# Returns a dict of userIDs and corresponding user factors
def genPCA2(userID_List, reviewVectors):
    pca = PCA(n_components=3)
    reviewMatrix = list(reviewVectors.values())
    pca.fit(reviewMatrix)

    result = np.dot(reviewMatrix, pca.components_.T)
    userID_PCADict = dict()
    for i in range(len(userID_List)) :
        userID_PCADict.update({userID_List[i] : result[i]})
    return userID_PCADict

def genPCA(userID_List, train_csv, test_csv, word2vec_model, id_featureDict):
    train_IDList = list(train_csv.keys())
    test_IDList = list(test_csv.keys())
    full_IDList = train_IDList + test_IDList

    trainUserID_IDDict = getUserID_Dict(train_csv)
    testUserID_IDDict = getUserID_Dict(test_csv)

    train_UserIDList = list(trainUserID_IDDict.keys())
    test_UserIDList = list(testUserID_IDDict.keys())
    userID_List = list(set(train_UserIDList + test_UserIDList))

    reviewVectors = dict()

    for user_id in userID_List:
        #List of review ids that correspond to user_ids
        IDs = list()
        if user_id in trainUserID_IDDict and user_id in testUserID_IDDict:
            IDs = trainUserID_IDDict.get(user_id) + testUserID_IDDict.get(user_id)
        elif user_id in trainUserID_IDDict:
            IDs = trainUserID_IDDict.get(user_id)
        elif user_id in testUserID_IDDict:
            IDs = testUserID_IDDict.get(user_id)
        else:
            print("Never reaches here")
        reviews = list()
        for id in IDs:
            features = id_featureDict.get(id)
            for feature in features:
                reviews += [feature]
        reviews = np.array(reviews)
        #Takes the average of the entire word embeddings of all the reviews related to the user_ids
        #average all of their word2vec features over the training data to treat as 128-dimensional "user-language representations".
        reviews = np.mean(reviews, axis=0)
        reviewVectors.update({user_id : reviews})
    pca = PCA(n_components=3)
    reviewMatrix = list(reviewVectors.values())
    result = pca.fit(reviewMatrix)
    result = pca.transform(reviewMatrix)

    userID_PCADict = dict()
    for i in range(len(userID_List)) :
        userID_PCADict.update({userID_List[i] : result[i]})
    return userID_PCADict

def checkpointTwo(train_csv, test_csv, word2vec_model, sharedTask):
    print("\nStage 2:")
    train_IDList = list(train_csv.keys())
    test_IDList = list(test_csv.keys())
    full_IDList = train_IDList + test_IDList

    trainUserID_IDDict = getUserID_Dict(train_csv)
    testUserID_IDDict = getUserID_Dict(test_csv)

    train_UserIDList = list(trainUserID_IDDict.keys())
    test_UserIDList = list(testUserID_IDDict.keys())
    userID_List = list(set(train_UserIDList + test_UserIDList))

    id_featureDict = dict() #Review embeddings corresponding to each review id, ie a list of word vectors for each reviewText
    # Assumes each review id is unique
    for i in range(len(full_IDList)):
        if full_IDList[i] in train_IDList:
            id_featureDict.update({full_IDList[i] : extractEmbeddingsVectors(full_IDList[i], train_csv, word2vec_model)})
        elif full_IDList[i] in test_IDList:
            id_featureDict.update({full_IDList[i] : extractEmbeddingsVectors(full_IDList[i], test_csv, word2vec_model)})
        else:
            print("Never reaches here")
    userID_PCADict = genPCA(train_UserIDList, train_csv, test_csv, word2vec_model, id_featureDict)

    Xs_train = list()
    for id in train_IDList:
        embeddingsV = np.mean(id_featureDict.get(id),axis=0)
        userFactors = list()
        # Doesn't matter where you get the user_id from, either train or test csv will sufficise if it exists in either one.
        if id in train_IDList:
            userFactors = userID_PCADict.get(train_csv.get(id)[3], [1, 1, 1])
        elif id in test_IDList:
            userFactors = userID_PCADict.get(test_csv.get(id)[3], [1, 1, 1])
        else:
            print("Never goes here")
        feature0 = embeddingsV*(userFactors[0])
        feature1 = embeddingsV*(userFactors[1])
        feature2 = embeddingsV*(userFactors[2])
        feature = np.concatenate((embeddingsV, feature0, feature1, feature2)).flatten()
        Xs_train += [feature]

    # The features go embeds; embeds*f1; embeds*f2; embeds*f3
    ys_train = extractYVector(train_csv)
    print("Training")
    rater = trainRater(Xs_train, ys_train)
    if not sharedTask:
        print("Testing")
        #Testing
        Xs_test = list()
        for id in test_IDList:
            embeddingsV = np.mean(id_featureDict.get(id),axis=0)
            userFactors = list()
            # Doesn't matter where you get the user_id from, either train or test csv will sufficise if it exists in either one.
            if id in train_IDList:
                userFactors = userID_PCADict.get(train_csv.get(id)[3], [1, 1, 1])
            elif id in test_IDList:
                userFactors = userID_PCADict.get(test_csv.get(id)[3], [1, 1, 1])
            else:
                print("Never goes here")
            feature0 = embeddingsV*(userFactors[0])
            feature1 = embeddingsV*(userFactors[1])
            feature2 = embeddingsV*(userFactors[2])
            feature = np.concatenate((embeddingsV, feature0, feature1, feature2)).flatten()
            Xs_test += [feature]
        #ys_test = extractYVector(test_csv)
        ys_pred = fixOutput(rater.predict(Xs_test), 1, 5)
        pred_dict = dict()
        idList = list(test_csv.keys())
        for i in range(len(idList)):
            pred_dict.update({idList[i] : ys_pred[i]})
        ys_true = extractYVector(test_csv)
        MAE = mean_absolute_error(ys_true, ys_pred)
        print("Testing Results")
        print("Mean Absolute Error is : " + str((MAE)))
    
    checkpointThree(train_csv, test_csv, word2vec_model, userID_PCADict, sharedTask)

    return userID_PCADict

### Neural Network Part

def review2Tensor(reviewID, seq_len,  train_csv, train_model):
    # Should return a numpy array of shape(sequence_length, 128)
    # We use 128 because of the word2vec embeddings
    # If the length of the review is less than sequence_length, we pad, else we cut off anything beyond
    
    wordVectors = extractEmbeddingsVectors(reviewID, train_csv, train_model)
    if len(wordVectors) > seq_len:
        wordVectors = wordVectors[:seq_len]
    elif len(wordVectors) < seq_len:
        wordVectors = np.concatenate((wordVectors, np.zeros((seq_len - len(wordVectors), 128))), axis=0)
    wordVectors = np.array([wordVectors])
    # Tensor for my NN wants type float for some reason even though it is less precision
    return torch.Tensor(wordVectors).float()
def getXYFromData(seq_len, csv_dict, word2vec_model):
    # Returns a tensor of shape(N, seq_length, 128) and the to-be-removed reviewIDs in the list
    # We use 128 because of the word2vec embeddings
    idList = list(csv_dict)
    data = list()
    dataTensor = torch.Tensor(len(idList), seq_len, 128)
    result = list()
    for id in idList:
        data.append(review2Tensor(id, seq_len, csv_dict, word2vec_model))
        temp = csv_dict.get(id)[0]
        if "" == temp:
            # Empty string for particular case of shared task
            # Doesn't matter what is inputed as true_rating since we only care about prediction
            result.append([3])
        else:
            result.append([float(csv_dict.get(id)[0])])
    torch.cat(data, out=dataTensor)
    result = torch.Tensor(result)
    return dataTensor, result
def getMaxWordsInSentence(csv_dict):
    idList = list(csv_dict.keys())
    maxWords = -1
    for id in idList:
        wordsInReview = tokenizeReviews(id, csv_dict, None, None)
        if len(wordsInReview) > maxWords:
            maxWords = len(wordsInReview)
    return maxWords
def getAvgWordsInSentence(csv_dict):
    idList = list(csv_dict.keys())
    avgWords = list()
    for id in idList:
        wordsInReview = tokenizeReviews(id, csv_dict, None, None)    
        avgWords += [len(wordsInReview)]
    avgWords = np.mean(np.array(avgWords))
    return avgWords
def getRatingsCount(csv_dict):
    return 0

def build_dataloader(bs, shfle, csv_dict, word2vec_model):
    """
        Builds a PyTorch Dataloader object

        args:
            bs - (integer) number of examples per batch
            shfle - (bool) to randomly sample train instances from dataset
    """
    x, y = getXYFromData(50, csv_dict, word2vec_model)
    idList = list(csv_dict)
    dataset = TensorDataset(x, y)
    mapping = list()
    for id, data in zip(idList, dataset):
        mapping.append((id, data))
    return DataLoader(mapping, batch_size=bs, shuffle=shfle, num_workers=4)

class my_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super(my_LSTM, self).__init__()
    
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2rating = nn.Linear(hidden_dim, 1)
        self.hiddenPCA2rating = nn.Linear(hidden_dim*4, 1)
        self.run_cuda = torch.cuda.is_available()

    def forward(self, ids, input, userID_PCADict, train_csv, test_csv):
        train_IDList = list(train_csv.keys())
        test_IDList = list(test_csv.keys())

        lstm_out, _ = self.lstm(input)
        result = lstm_out[-1]
        #rating = self.hidden2rating(result)
        temp = result.clone().cuda()
        temp2 = torch.Tensor()
        userFactors = [1, 1, 1]
        for i in range(len(ids)):
            id = ids[i]
            embeddingsV = temp[i]
            # Doesn't matter where you get the user_id from, either train or test csv will sufficise if it exists in either one.
            if id in train_IDList:
                userFactors = userID_PCADict.get(train_csv.get(id)[3], [1, 1, 1])
            elif id in test_IDList:
                userFactors = userID_PCADict.get(test_csv.get(id)[3], [1, 1, 1])
            userFactors = torch.Tensor(userFactors).cuda()
            feature0 = torch.mul(embeddingsV,(userFactors[0])).cuda()
            feature1 = torch.mul(embeddingsV,(userFactors[1])).cuda()
            feature2 = torch.mul(embeddingsV,(userFactors[2])).cuda()
            feature = torch.cat((feature0, feature1, feature2)).cuda()
            if i == 0:
                temp2 = feature.clone().unsqueeze(0).cuda()
            else:
                temp2 = torch.cat((temp2, feature.unsqueeze(0)))
        result = torch.cat((result, temp2), dim=1)
        rating = self.hiddenPCA2rating(result)
        return rating

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        # init hidden state and cell state to tensor of 0s, both of size self.hidden_size
        hidden, cell = (weight.new_zeros(batch_size, self.hidden_dim), weight.new_zeros(batch_size, self.hidden_dim))

        # confirm they are on GPU
        if self.run_cuda:
            hidden.cuda()
            cell.cuda()

        return hidden, cell

def NNtrain(model, training_data, train_csv, test_csv, userID_PCADict, epochs=64):
    model.zero_grad()
    idList = list(train_csv.keys())
    seq_len = 50
    loss_function1 = nn.MSELoss()
    loss_function2 = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        train_loss1 = []
        train_loss2 = []
        for ids, dataXY in training_data:
            data = dataXY[0]
            labels = dataXY[1]
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            optimizer.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            # LSTM expects batch at dim=1
            data = data.transpose(0,1).cuda()
            #hidden = model.init_hidden(data.shape[0])
            # Step 3. Run our forward pass.
            pred_rating = model(ids, data, userID_PCADict, train_csv, test_csv)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            labels = labels.cuda()
            
            loss1 = loss_function1(pred_rating, labels)
            loss2 = loss_function2(pred_rating, labels)
            train_loss1.append(loss1.item())
            train_loss2.append(loss2.item())


            loss1.backward()
            optimizer.step()
        
        print("For Epoch " + str(epoch) + " MSE Loss : " + str({np.mean(train_loss1)}) + " MAE Loss: " + str({np.mean(train_loss2)}))

def NNtest(model, test_data, train_csv, test_csv, userID_PCADict, sharedTask):
    model.eval()
    loss_function1 = nn.MSELoss()
    loss_function2 = nn.L1Loss()
    id_ratingDict = dict()
    with torch.no_grad():
        test_loss1 = []
        test_loss2 = []
        for ids, dataXY in test_data:
            data = dataXY[0]
            labels = dataXY[1]
            #hidden = model.init_hidden(data.shape[0])
            data = data.transpose(0,1).cuda()

            pred_rating = model(ids, data, userID_PCADict, train_csv, test_csv)
            #loss = loss_func(preds, labels.view(-1).cuda())
            if not sharedTask:
                loss1 = loss_function1(pred_rating, labels.cuda())
                loss2 = loss_function2(pred_rating, labels.cuda())
                test_loss1.append(loss1.item())
                test_loss2.append(loss2.item())
                print(f'MSE Loss for test: {np.mean(test_loss1)} MAE Loss for test: {np.mean(test_loss2)}')
            if sharedTask:
                id_ratingDict.update({ids[0] : pred_rating[0].cpu().numpy()[0]})
    return id_ratingDict

def checkpointThree(train_csv, test_csv, word2vec_model, userID_PCADict, shared_Task):
    print("\nStage 3:")
    training_data =  build_dataloader(16, True, train_csv, word2vec_model)

    model = my_LSTM(128, 128, 2, .2)
    if torch.cuda.is_available():
        model.cuda()
    NNtrain(model, training_data, train_csv, test_csv, userID_PCADict)

    test_data =  build_dataloader(32, False, test_csv, word2vec_model)
    if not shared_Task:
        NNtest(model, test_data, train_csv, test_csv, userID_PCADict, False)
        return
    else:
        sharedTask(model, train_csv, test_csv, word2vec_model, userID_PCADict, shared_Task)
    return

def sharedTask(model, train_csv, test_csv, word2vec_model, userID_PCADict, shared_Task):
    print("\nRunning Kaggle Shared Task")

    test_data =  build_dataloader(1, False, test_csv, word2vec_model)

    my_dict = NNtest(model, test_data, train_csv, test_csv, userID_PCADict, shared_Task)
    with open('submission.csv', 'w') as f:
        fieldnames = ['id', 'Predicted']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, v])) for k, v in my_dict.items()]
        writer.writerows(data)
    print("Finished Writing Submission")
    return
def main(argv):
    if len(argv) != 2:
        print("Needs a train and test file")
    
    else:
        train_csv = preparecsv(argv[0])
        test_csv = preparecsv(argv[1])
        if argv[0] == 'food_train.csv':
            checkpointOne('food',train_csv, test_csv, False)
        if argv[0] == 'music_train.csv':
            checkpointOne('music',train_csv, test_csv, False)
        if argv[1] == 'musicAndPetsup_test_noLabels.csv':
            checkpointOne(argv[0], train_csv, test_csv, True)
    #Average words per review is about 26 for music and 23 for food
    #extractPaddedReviews(train_csv, train_model)

    return

if __name__== '__main__':
    main(sys.argv[1:])