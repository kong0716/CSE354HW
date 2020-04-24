import  re, numpy as np, sklearn, scipy.stats, pandas, csv
import random, math, sys
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

## STEP 1.2:Read the csv into memory
def preparecsv():
    filename = "songdata.csv"
    # initializing the rows list 
    rows = [] 
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        id_dict = dict()
        #id_lyrics = dict()
        for row in reader:
            #print(row['artist'])
            unique_id = row['artist'] + "-" + row['song']
            unique_id = unique_id.lower().replace(" ", "_")
            id_dict.update({unique_id : [row['song'], row['text']]})
            #print(unique_id)
        return id_dict

## STEP 1.3: Tokenize the song titles.
def tokenize(sent):
    #input: a single sentence as a string.
    #output: a list of each "word" in the text
    # must use regular expressions

    tokens = []
    #<FILL IN>
    #The second group does the majority of words with hashtags and @'s
    #The first group tackles the abbreviations
    tokenizer = re.compile('(?:[a-zA-Z]\.){2,}|[#@]*[a-zA-Z0-9]*[\'’-]*[a-z]+|[.,!?;]|[A-Z0-9-]+|[\n]')
    return tokenizer.findall(sent)
def tokenize_song(id, csv_dict):
    return tokenize(csv_dict.get(id)[0])

## STEP 1.4: Tokenize the song lyrics
def tokenize_lyrics(id, csv_dict):
    tokens = tokenize(csv_dict.get(id)[1])
    tokens.insert(0, "<s>")
    tokens.insert(len(tokens), "</s>")
    for i in range(len(tokens)):
        #tokens[i] = tokens[i].lower()
        if tokens[i] == "\n":
            tokens[i] = "<newline>"
    return tokens

#STEP 2.1: Create a vocabulary of words from lyrics
def create_vocab_dict(tokens):
    word_count = Counter(tokens)
    oov = 0
    for key in list(word_count.keys()):
        if word_count.get(key) <= 2:
            oov += word_count.get(key)
            word_count.pop(key, None)
    word_count.update({"<OOV>" : oov})
    #print(word_count.get("<OOV>"))
    return word_count
def tokensfromNlyrics(nlyrics, csv_dict):
    tokens = list()
    sample_ids = list(csv_dict.keys())[0: nlyrics]
    #print(sample_ids)
    for id in sample_ids:
        tokens += tokenize_lyrics(id, csv_dict)
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    return tokens
#STEP 2.2 - 2.3: Create a bigram/trigram matrix (rows as previous word; columns as current word)
def create_bigram_matrix(tokens, vocab):
    #print(tokens)
    #tokens = ["I", "like", "to", "eat", "Chinese", "food", "<newline>", "I", "also", "like", "to", "play"]
    #print(vocab)
    for i in range(len(tokens)):
        if tokens[i] not in vocab:
            tokens[i] = "<OOV>"
    bigram_list = list(zip(*[tokens[i:] for i in range(2)]))
    bigram_count = Counter(bigram_list)
    bigrams = defaultdict(dict)
    for tup in bigram_list:
        bigrams[tup[0]][tup[1]] = bigram_count.get(tup)
    return bigrams
def create_trigram_matrix(tokens, vocab):
    for i in range(len(tokens)):
        if tokens[i] not in vocab:
            tokens[i] = "<OOV>"
    trigram_list = list(zip(*[tokens[i:] for i in range(3)]))
    trigram_count = Counter(trigram_list)
    trigrams = defaultdict(dict)
    for tup in trigram_list:
        trigrams[(tup[0], tup[1])][tup[2]] = trigram_count.get(tup)
    return trigrams
#STEP 2.4: Create a method to calculate the probability of all possible current words wi 
#           given either a single previous word (wi-1 -- a bigram model) 
#           or two previous words (wi-1 and wi-2 -- a trigram model).
def probabilities(bigram_dict, trigram_dict, tokens, vocab, *args):
    #print("In probabilities")
    wi1 = args[len(args)-1]
    prob_dict = dict()
    sum = 0
    wi0List = list(bigram_dict.get(wi1).keys())
    if "<OOV>" in wi0List:
        wi0List.remove("<OOV>")
    if len(wi0List) == 0:
        # unigram
        #print("Doing unigram")
        for wi0 in vocab:
            if wi0 != "<OOV>":    
                prob = vocab.get(wi0)/len(tokens)
                prob_dict.update({wi0 : prob})
                sum += prob
        return prob_dict, sum
    if len(args) == 1:
        #Add-one smooth bigram model
        for wi0 in wi0List:
            prob = (bigram_dict.get(wi1).get(wi0) + 1) / (vocab.get(wi1, 0) + len(vocab))
            sum += prob
            prob_dict.update({wi0 : prob})
        return prob_dict, sum
    if len(args) == 2:
        wi2 = args[0]
        if (wi2, wi1) in trigram_dict:
            wi0List = list(trigram_dict.get((wi2, wi1)).keys())
            if "<OOV>" in wi0List:
                wi0List.remove("<OOV>")
            if len(wi0List) == 0:
                #print("Should never reach here, using unigram model")
                for wi0 in vocab:
                    if wi0 != "<OOV>":    
                        prob = vocab.get(wi0)/len(tokens)
                        prob_dict.update({wi0 : prob})
                        sum += prob
                return prob_dict, sum
            else:
                for wi0 in wi0List:
                    trigram_prob = (trigram_dict.get((wi2, wi1)).get(wi0, 0) + 1) / (bigram_dict.get(wi2).get(wi1, 0) + len(vocab))
                    bigram_prob = (bigram_dict.get(wi1).get(wi0, 0) + 1) / (vocab.get(wi1, 0) + len(vocab))
                    prob = (bigram_prob + trigram_prob) / 2
                    sum += prob
                    prob_dict.update({wi0 : prob})
        else:
            #print("Args not in trigram")
            #print(wi2)
            #print(wi1)
            for wi0 in wi0List:
                trigram_prob = 1 / (bigram_dict.get(wi2).get(wi1, 0) + len(vocab))
                bigram_prob = (bigram_dict.get(wi1).get(wi0, 0) + 1) / (vocab.get(wi1, 0) + len(vocab))
                prob = (bigram_prob + trigram_prob) / 2
                sum += prob
                prob_dict.update({wi0 : prob})
    return prob_dict, sum

def probability(wi0, bigram_dict, trigram_dict, tokens, vocab, *args):
    prob_dict, sum = probabilities(bigram_dict, trigram_dict, tokens, vocab, *args)
    if wi0 in prob_dict:
        return prob_dict.get(wi0)
    elif len(args) == 1:
        prob = vocab.get(wi0)/len(tokens)
        return prob
    elif len(args) == 2:
        # wi-1 and wi-2 respectively
        wi2 = args[0]
        wi1 = args[1]
        trigram_prob = (trigram_dict.get((wi2, wi1)).get(wi0, 0) + 1) / (bigram_dict.get(wi2).get(wi1, 0) + len(vocab))
        bigram_prob = (bigram_dict.get(wi1).get(wi0, 0) + 1) / (vocab.get(wi1, 0) + len(vocab))
        prob = (bigram_prob + trigram_prob) / 2
        return prob

def listdif(one, one_cpy):
    if one == one_cpy:
        print("Passed")
    else:
        li_dif = [i for i in one + one_cpy if i not in one or i not in one_cpy] 
        print(li_dif)
## Stage 1 Checkpoint
def stage1checkpoint(csv_dict):
    one = tokenize_lyrics("abba-burning_my_bridges", csv_dict)
    two = tokenize_lyrics("beach_boys-do_you_remember?", csv_dict)
    three = tokenize_lyrics("avril_lavigne-5,_4,_3,_2,_1_(countdown)", csv_dict)
    four = tokenize_lyrics("michael_buble-l-o-v-e", csv_dict)

    one_cpy = ['<s>', 'Well', ',', 'you', 'hoot', 'and', 'you', 'holler', 'and', 'you', 'make', 'me', 'mad', '<newline>', 'And', "I've", 'always', 'been', 'under', 'your', 'heel', '<newline>', 'Holy', 'christ', 'what', 'a', 'lousy', 'deal', '<newline>', 'Now', "I'm", 'sick', 'and', 'tired', 'of', 'your', 'tedious', 'ways', '<newline>', 'And', 'I', "ain't", 'gonna', 'take', 'it', 'no', 'more', '<newline>', 'Oh', 'no', 'no', '-', 'walkin', 'out', 'that', 'door', '<newline>', '<newline>', 'Burning', 'my', 'bridges', ',', 'cutting', 'my', 'tie', '<newline>', 'Once', 'again', 'I', 'wanna', 'look', 'into', 'the', 'eye', '<newline>', 'Being', 'myself', '<newline>', 'Counting', 'my', 'pride', '<newline>', 'No', 'un-right', "neighbour's", 'gonna', 'take', 'me', 'for', 'a', 'ride', '<newline>', 'Burning', 'my', 'bridges', '<newline>', 'Moving', 'at', 'last', '<newline>', 'Girl', "I'm", 'leaving', 'and', "I'm", 'burying', 'the', 'past', '<newline>', 'Gonna', 'have', 'peace', 'now', '<newline>', 'You', 'can', 'be', 'free', '<newline>', 'No', 'one', 'here', 'will', 'make', 'a', 'sucker', 'out', 'of', 'me', '<newline>', '<newline>', '</s>']
    two_cpy = ['<s>', 'Little', 'Richard', 'sang', 'it', 'and', 'Dick', 'Clark', 'brought', 'it', 'to', 'life', '<newline>', 'Danny', 'And', 'The', 'Juniors', 'hit', 'a', 'groove', ',', 'stuck', 'as', 'sharp', 'as', 'a', 'knife', '<newline>', 'Well', 'now', 'do', 'you', 'remember', 'all', 'the', 'guys', 'that', 'gave', 'us', 'rock', 'and', 'roll', '<newline>', '<newline>', 'Chuck', "Berry's", 'gotta', 'be', 'the', 'greatest', 'thing', "that's", 'come', 'along', '<newline>', 'hum', 'diddy', 'waddy', ',', 'hum', 'diddy', 'wadda', '<newline>', 'He', 'made', 'the', 'guitar', 'beats', 'and', 'wrote', 'the', 'all-time', 'greatest', 'song', '<newline>', 'hum', 'diddy', 'waddy', ',', 'hum', 'diddy', 'wadda', '<newline>', 'Well', 'now', 'do', 'you', 'remember', 'all', 'the', 'guys', 'that', 'gave', 'us', 'rock', 'and', 'roll', '<newline>', 'hum', 'diddy', 'waddy', 'doo', '<newline>', '<newline>', 'Elvis', 'Presley', 'is', 'the', 'king', '<newline>', "He's", 'the', 'giant', 'of', 'the', 'day', '<newline>', 'Paved', 'the', 'way', 'for', 'the', 'rock', 'and', 'roll', 'stars', '<newline>', 'Yeah', 'the', 'critics', 'kept', 'a', 'knockin', '<newline>', 'But', 'the', 'stars', 'kept', 'a', 'rockin', '<newline>', 'And', 'the', 'choppin', "didn't", 'get', 'very', 'far', '<newline>', '<newline>', 'Goodness', 'gracious', 'great', 'balls', 'of', 'fire', '<newline>', "Nothin's", 'really', 'movin', 'till', 'the', "saxophone's", 'ready', 'to', 'blow', '<newline>', 'do', 'you', 'remember', ',', 'do', 'you', 'remember', '<newline>', 'And', 'the', "beat's", 'not', 'jumpin', 'till', 'the', 'drummer', 'says', "he's", 'ready', 'to', 'go', '<newline>', 'do', 'you', 'remember', ',', 'do', 'you', 'remember', '<newline>', 'Well', 'now', 'do', 'you', 'remember', 'all', 'the', 'guys', 'that', 'gave', 'us', 'rock', 'and', 'roll', '<newline>', 'do', 'you', 'remember', '<newline>', '<newline>', "Let's", 'hear', 'the', 'high', 'voice', 'wail', 'oooooooooo', '<newline>', 'And', 'hear', 'the', 'voice', 'down', 'low', 'wah-ah', 'ah-ah', '<newline>', "Let's", 'hear', 'the', 'background', '<newline>', 'Um', 'diddy', 'wadda', ',', 'um', 'diddy', 'wadda', '<newline>', 'Um', 'diddy', 'wadda', ',', 'um', 'diddy', 'wadda', '<newline>', 'They', 'gave', 'us', 'rock', 'and', 'roll', '<newline>', 'Um', 'diddy', 'wadda', ',', 'um', 'diddy', 'wadda', '<newline>', 'They', 'gave', 'us', 'rock', 'and', 'roll', '<newline>', 'Um', 'diddy', 'wadda', ',', 'um', 'diddy', 'wadda', '<newline>', 'They', 'gave', 'us', 'rock', 'and', 'roll', '<newline>', '<newline>', '</s>']
    three_cpy = ['<s>', 'Verse', '<newline>', 'Countin', 'down', ',', "it's", 'new', 'years', 'eve', ',', '<newline>', 'You', 'come', 'on', 'over', ',', 'then', 'start', ',', 'askin', 'me', ',', '<newline>', 'Hey', 'girl', ',', 'you', 'wanna', 'dance', '?', '<newline>', 'I', 'try', 'to', 'say', 'no', ',', 'but', 'it', 'came', 'out', 'yes', 'and', ',', '<newline>', 'Here', 'it', 'goes', ',', 'middle', 'of', 'the', 'dance', 'floor', ',', '<newline>', "We're", 'both', ',', 'pretty', 'nervous', ',', 'I', 'can', 'tell', 'by', 'look', 'in', 'his', 'eyes', ',', '<newline>', 'Then', 'came', 'the', 'count', 'down', ',', '<newline>', '<newline>', 'Chorus', '<newline>', '5', ',', '4', ',', '3', ',', '2', ',', '1', '!', '<newline>', 'New', 'years', '!', '<newline>', 'Everybody', 'shouts', ',', 'and', 'here', 'it', 'goes', ',', '<newline>', 'I', 'wanna', 'see', 'it', 'flow', ',', '<newline>', 'We', ',', 'look', 'at', 'the', 'screen', ',', '<newline>', 'The', 'camera', 'was', 'focused', 'on', 'us', ',', '<newline>', 'You', 'blushed', 'and', 'turned', 'away', ',', '<newline>', 'I', 'kissed', 'you', 'on', 'the', 'cheek', 'and', 'ran', 'away', ',', '<newline>', '5', ',', '4', ',', '3', ',', '2', ',', '1', '!', '<newline>', '<newline>', 'Verse', '2', '<newline>', 'My', 'birthday', ',', "everybody's", 'here', ',', '<newline>', 'All', 'except', 'for', 'one', ',', '<newline>', 'At', 'the', 'sleepover', ',', 'you', 'showed', 'right', 'then', ',', '<newline>', 'All', 'the', 'girls', 'scream', ',', 'except', 'for', 'me', ',', '<newline>', 'You', 'have', ',', 'roses', 'in', 'your', 'hand', ',', '<newline>', 'Gave', 'em', 'to', 'me', 'and', 'whispered', ',', '<newline>', 'Payin', 'you', 'back', ',', '<newline>', 'Then', ',', 'all', 'the', 'girls', '<newline>', 'Started', 'to', 'get', 'that', 'feeling', '.', '.', '.', '<newline>', '<newline>', 'Chorus', '<newline>', '<newline>', 'Verse', '3', '<newline>', 'Then', 'they', 'shouted', '<newline>', '5', ',', '4', ',', '3', ',', '2', ',', '1', '!', '<newline>', 'Here', 'it', 'goes', ',', '<newline>', 'Then', 'he', 'kissed', 'me', 'on', 'the', 'cheek', 'and', 'said', ',', '<newline>', 'Just', 'remember', 'this', 'day', ',', 'Later', 'on', 'in', 'your', 'life', ',', '<newline>', "You'll", 'recall', 'this', 'day', ',', 'and', 'wonder', 'why', ',', '<newline>', 'I', "didn't", 'count', 'down', ',', '<newline>', '5', ',', '4', ',', '3', ',', '2', ',', '1', '!', '<newline>', 'Then', 'I', 'got', 'that', 'feeling', 'when', 'you', 'left', ',', '<newline>', 'It', 'was', 'the', 'count', 'down', ',', '<newline>', 'I', 'got', 'that', 'feeling', '<newline>', 'It', 'was', 'a', '5', ',', '4', ',', '3', ',', '2', ',', '1', '!', '<newline>', 'Once', 'again', '<newline>', '<newline>', '</s>']
    four_cpy = ['<s>', 'L', 'is', 'for', 'the', 'way', 'you', 'look', 'at', 'me', '<newline>', 'O', 'is', 'for', 'the', 'only', 'one', 'I', 'see', '<newline>', 'V', 'is', 'very', ',', 'very', 'extraordinary', '<newline>', 'E', 'is', 'even', 'more', 'than', 'anyone', 'that', 'you', 'adore', '<newline>', '<newline>', 'And', 'love', 'is', 'all', 'that', 'I', 'can', 'give', 'to', 'you', '<newline>', 'Love', 'is', 'more', 'than', 'just', 'a', 'game', 'for', 'two', '<newline>', 'Two', 'in', 'love', 'can', 'make', 'it', '<newline>', 'Take', 'my', 'heart', 'but', 'please', "don't", 'break', 'it', '<newline>', 'Love', 'was', 'made', 'for', 'me', 'and', 'you', '<newline>', '<newline>', 'L', 'is', 'for', 'the', 'way', 'you', 'look', 'at', 'me', '<newline>', 'O', 'is', 'for', 'the', 'only', 'one', 'I', 'see', '<newline>', 'V', 'is', 'very', ',', 'very', 'extraordinary', '<newline>', 'E', 'is', 'even', 'more', 'than', 'anyone', 'that', 'you', 'adore', '<newline>', '<newline>', 'And', 'love', 'is', 'all', 'that', 'I', 'can', 'give', 'to', 'you', '<newline>', 'Love', ',', 'love', ',', 'love', 'is', 'more', 'than', 'just', 'a', 'game', 'for', 'two', '<newline>', 'Two', 'in', 'love', 'can', 'make', 'it', '<newline>', 'Take', 'my', 'heart', 'but', 'please', "don't", 'break', 'it', '<newline>', 'Cause', 'love', 'was', 'made', 'for', 'me', 'and', 'you', '<newline>', 'I', 'said', 'love', 'was', 'made', 'for', 'me', 'and', 'you', '<newline>', 'You', 'know', 'that', 'love', 'was', 'made', 'for', 'me', 'and', 'you', '<newline>', '<newline>', '</s>']
    #Passed check
    '''
    listdif(one, one_cpy)
    listdif(two, two_cpy)
    listdif(four, four_cpy)
    listdif(three, three_cpy)
    '''

    print(tokenize_song("abba-burning_my_bridges", csv_dict))
    print(one)
    print(tokenize_song("beach_boys-do_you_remember?", csv_dict))
    print(two)
    print(tokenize_song("avril_lavigne-5,_4,_3,_2,_1_(countdown)", csv_dict))
    print(three)
    print(tokenize_song("michael_buble-l-o-v-e", csv_dict))
    print(four)

## Stage 2 Checkpoint
def stage2checkpoint(csv_dict):
    lyricTokens = tokensfromNlyrics(5000, csv_dict)
    vocab = create_vocab_dict(lyricTokens)
    bigram = create_bigram_matrix(lyricTokens, vocab)
    trigram = create_trigram_matrix(lyricTokens, vocab)
    print("p( you | (\'i\', \'love\') ) = " + str(probability("you", bigram, trigram, lyricTokens, vocab,"i", "love")))
    print("p( special | (\'midnight\',) ) = " + str(probability("special", bigram, trigram, lyricTokens, vocab,"midnight")))
    print("p( special | (\'\'very\',) ) = " + str(probability("special", bigram, trigram, lyricTokens, vocab,"very")))
    print("p( special | (\'something\', \'very\') ) = " + str(probability("special", bigram, trigram, lyricTokens, vocab,"something", "very")))
    print("p( funny | (\'something\', \'very\') ) = " + str(probability("funny", bigram, trigram, lyricTokens, vocab,"something", "very")))

##################################################################
#4. Adjective Classifier

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
            # If the word is OOV, default to the length of the dictionary
            curword = wordToIndex.get(tokens[targetI].lower(), num_words)
            if not (targetI == 0 or targetI == num_words-1):
                prevword = wordToIndex.get(tokens[targetI-1].lower(), num_words)
                nextword = wordToIndex.get(tokens[targetI+1].lower(), num_words)
            #Cases for the beginning and end of the tokens
            if targetI == 0 and num_words > 1:
                nextword = wordToIndex.get(tokens[targetI+1].lower(), num_words)
            if targetI == num_words-1 and num_words > 1:
                prevword = wordToIndex.get(tokens[targetI-1].lower(), num_words)
        '''
        if prevword == -1:
            for i in reversed(range(2, targetI)):
                prevword = wordToIndex.get(tokens[targetI-i].lower(), -1)
                if prevword != -1:
                    break
        if nextword == -1:
            for i in range(2, targetI):
                nextword = wordToIndex.get(tokens[targetI-i].lower(), -1)
                if nextword != -1:
                    break
        
        if prevword == -1:
            for i in reversed(range(2, targetI)):
                prevword = wordToIndex.get(tokens[targetI-i].lower(), -1)
                if prevword != -1:
                    break
        '''
        pass
        featureVector = [0]*(len(wordToIndex)*3+offset)
        featureVector[0] = vowelctr
        featureVector[1] = consonantctr
        featureVector[prevword + offset] = 1
        featureVector[curword + offset + onehotOffset] = 1
        featureVector[nextword + offset + 2*onehotOffset] = 1
        featuresPerTarget.insert(targetI,featureVector)
    return featuresPerTarget #a (num_words x k) matrix


def trainTestAdjectiveClassifier(X_train, y_train, X_subtrain, X_dev, y_subtrain, y_dev, c):
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
                                                            test_size=0.10, random_state = 42)
    oldacc = 0
    optimalc = 0
    for i in range(-10, 10):
        model = trainTestAdjectiveClassifier(features, adjs, X_subtrain, X_dev, y_subtrain, y_dev, math.pow(10, i))
        y_pred = model.predict(X_dev)
        #print(y_pred)
        #calculate accuracy:
        newacc = (1 - np.sum(np.abs(y_pred - y_dev))/len(y_dev) )
        if newacc > oldacc:
            optimalc = math.pow(10, i)
            oldacc = newacc
    bestmodel = trainTestAdjectiveClassifier(features, adjs, X_subtrain, X_dev, y_subtrain, y_dev, optimalc)
    print("Optimal C is : " + str(optimalc))
    return bestmodel

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

## Step 3.1
def getBestAdjModel(wordToIndex):
    #load data for 3 and 4 the adjective classifier data:
    taggedSents = getConllTags('daily547.conll')
    #3. Test Feature Extraction:
    #print("\n[ Feature Extraction Test ]\n")

    #Next, call Feature extraction per sentence
    sentXs = []
    sentYs = []
    #print("  [Extracting Features]")
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            sentXs.append(getFeaturesForTokens(words, wordToIndex)) 
            sentYs.append([1 if t == 'A' else 0 for t in tags])
    #test sentences
    #print("\n", taggedSents[5], "\n", sentXs[5], "\n")
    #print(taggedSents[192], "\n", sentXs[192], "\n")


    #4. Test Classifier Model Building
    #print("\n[ Classifier Test ]\n")
    #setup train/test:
    #flatten by word rather than sent: 
    X = [j for i in sentXs for j in i]
    y= [j for i in sentYs for j in i]
    np.random
    try: 
        X_train, X_test, y_train, y_test = train_test_split(np.array(X),
                                                            np.array(y),
                                                            test_size=0.20)
    except ValueError:
        print("\nLooks like you haven't implemented feature extraction yet.")
        print("[Ending test early]")
        sys.exit(1)
    #print("  [Broke into training/test. X_train is ", X_train.shape, "]")
    #Train the model.
    #print("  [Training the model]")
    tagger = trainAdjectiveClassifier(X_train, y_train)
    #print("  [Done]")
    return tagger

## STEP 3.2 Extract features for adjective classifier
def extractFeatures(unique_id, wordToIndex):
    return 0

## Step 3.3
def getAdjDict(csv_dict):
    #load data for 3 and 4 the adjective classifier data:
    taggedSents = getConllTags('daily547.conll')
    #first make word to index mapping: 
    wordToIndex = set() #maps words to an index
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent) #splits [(w, t), (w, t)] into [w, w], [t, t]
            wordToIndex |= set([w.lower() for w in words]) #union of the words into the set
    #print("  [Read ", len(taggedSents), " Sentences]")
    #turn set into dictionary: word: index
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)}

    model = getBestAdjModel(wordToIndex)

    #print(csv_dict)
    ids = csv_dict.keys()
    adj_dict = dict()
    for id in ids:
        #Tokenizes each title
        titleTokens = tokenize_song(id, csv_dict)
        for t in range(len(titleTokens)):
            titleTokens[t] = titleTokens[t].lower()
        y_pred = model.predict(getFeaturesForTokens(titleTokens, wordToIndex))
        indices = [i for i, x in enumerate(y_pred) if x == 1]
        for i in indices:
            if titleTokens[i].lower() in adj_dict:
                #print(adj_dict)
                templist = adj_dict.get(titleTokens[i].lower())
                templist.append(id)
                adj_dict.update({titleTokens[i].lower() : templist})
            else:
                templist = [id]
                adj_dict.update({titleTokens[i].lower() : templist})
    #Remove adjectives that appear 10 or less times
    keys = list(adj_dict.keys())
    #print(keys)
    for i in range(len(keys)):
        if len(adj_dict.get(keys[i])) <= 10:
            adj_dict.pop(keys[i], None)
    #print("Finished Step 3.3")
    return adj_dict

# Step 3.4
def getLanguageModel(adj_dict):
    csv_dict = preparecsv()
    languageModel = dict()
    for adj in adj_dict:
        tempID = adj_dict.get(adj)
        tokens = list()
        for id in tempID:
            tokens += tokenize_lyrics(id, csv_dict)
        for t in range(len(tokens)):
            tokens[t] = tokens[t].lower()
        vocab = create_vocab_dict(tokens)
        bigram = create_bigram_matrix(tokens, vocab)
        trigram = create_trigram_matrix(tokens, vocab)
        languageModel.update({adj : [bigram, trigram, tokens, vocab]})
    return languageModel

# Step 3.5 
def genLyrics(adj, languageModel):
    temp = languageModel.get(adj)
    if temp == None:
        return str(adj) + " was not classified as an adjective"
    bigram = temp[0]
    trigram = temp[1]
    tokens = temp[2]
    vocab = temp[3]
    lyrics = list()
    lyrics.append("<s>")
    prob_dict, sum = probabilities(bigram, trigram, tokens, vocab, "<s>")
    for w in prob_dict.keys():
        prob_dict.update({w : prob_dict.get(w)/sum})
    lyrics.append(np.random.choice(list(prob_dict.keys()), p = list(prob_dict.values())))
    for i in range(1, 32):
        prob_dict, sum = probabilities(bigram, trigram, tokens, vocab, lyrics[i-1], lyrics[i])
        # Normalizes probabilty
        for w in prob_dict.keys():
            prob_dict.update({w : prob_dict.get(w)/sum})
        #print(lyrics)
        #print(prob_dict.keys())
        #print(prob_dict.values())
        choices = list(prob_dict.keys())
        probs = list(prob_dict.values())
        #Error checking
        if len(choices) == 0:
            print(lyrics)
        word = np.random.choice(choices, p = probs)
        lyrics.append(word)
        if word == "</s>":
            break
    return lyrics

## Stage 3 Checkpoint
def stage3checkpoint(csv_dict):
    adj_dict = getAdjDict(csv_dict)
    lModel = getLanguageModel(adj_dict)
    print("Language model created")
    print("Adjective is \"good\":")
    print(adj_dict.get("good"))
    for i in range(3):
        print(genLyrics("good", lModel))
    print("Adjective is \"happy\":")
    print(adj_dict.get("happy"))
    for i in range(3):
        print(genLyrics("happy", lModel))
    print("Adjective is \"afraid\":")
    print(adj_dict.get("afraid"))
    for i in range(3):
        print(genLyrics("afraid", lModel))
    print("Adjective is \"red\":")
    print(adj_dict.get("red"))
    for i in range(3):
        print(genLyrics("red", lModel))
    print("Adjective is \"blue\":")
    print(adj_dict.get("blue"))
    for i in range(3):
        print(genLyrics("blue", lModel))
    
# Main
if __name__== '__main__':
    csv_dict = preparecsv()
    stage1checkpoint(csv_dict)
    stage2checkpoint(csv_dict)
    stage3checkpoint(csv_dict)
'''
    #Test the tagger.
    from sklearn.metrics import classification_report
    #get predictions:
    y_pred = tagger.predict(X_test)
    #compute accuracy:
    leny = len(y_test)
    print("test n: ", leny)
    acc = np.sum([1 if (y_pred[i] == y_test[i]) else 0 for i in range(leny)]) / leny
    print("Accuracy: %.4f" % acc)
    #print(classification_report(y_test, y_pred, ['not_adj', 'adjective']))
'''