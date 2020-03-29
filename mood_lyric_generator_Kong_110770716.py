import  re, numpy as np, sklearn, scipy.stats, pandas, csv
import random
from collections import Counter, defaultdict

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
    tokenizer = re.compile('(?:[a-zA-Z]\.){2,}|[#@]*[a-zA-Z0-9]+[\'’-]*[a-z\.]*|[\n]')
    return tokenizer.findall(sent)
def tokenize_song(id, csv_dict):
    return tokenize(csv_dict.get(id)[0])

## STEP 1.4: Tokenize the song lyrics
def tokenize_lyrics(id, csv_dict):
    tokens = tokenize(csv_dict.get(id)[1])
    tokens.insert(0, "<s>")
    tokens.insert(len(tokens), "</s>")
    for i in range(len(tokens)):
        if tokens[i] == "\n":
            tokens[i] = "<newline>"
    return tokens
## Stage 1 Checkpoint
#print(tokenize_song("abba-burning_my_bridges"))
#print(tokenize_lyrics("abba-burning_my_bridges"))
#print(tokenize_song("beach_boys-do_you_remember?"))
#print(tokenize_lyrics("beach_boys-do_you_remember?"))
#It seems that there is no 54321 avril song
#print(tokenize_song("avril_lavigne-5,_4,_3,_2,_1(countdown)"))
#print(tokenize_lyrics("avril_lavigne-5,_4,_3,_2,_1(countdown)"))
#print(tokenize_song("michael_buble-l-o-v-e"))
#print(tokenize_lyrics("michael_buble-l-o-v-e"))

#STEP 2.1: Create a vocabulary of words from lyrics
def create_vocab_dict(tokens):
    word_count = Counter(tokens)
    oov = 0
    for key in list(word_count.keys()):
        if word_count.get(key) <= 2:
            oov += word_count.get(key)
            del word_count[key]
    word_count.update({"<OOV>" : oov})
    #print(word_count.get("<OOV>"))
    return word_count
def tokensfromNlyrics(nlyrics):
    csv_dict = preparecsv()
    tokens = list()
    sample_ids = list(csv_dict.keys())[0: nlyrics]
    for id in sample_ids:
        tokens += tokenize_lyrics(id, csv_dict)
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
def probabilities(wi0, wi1, bigram_dict, tokens, vocab):
    if wi1 in bigram_dict:
        return (bigram_dict.get(wi1).get(wi0) + 1)/(vocab.get(wi1) + len(vocab))
    else:
        #unigram
        return vocab.get(wi0)/len(tokens)

def probabilities(wi0, wi1, wi2, trigram_dict, bigram_dict, tokens, vocab):
    if (wi1, wi2) in trigram_dict:
        #Interpolating
        return ( (probabilities(wi0, wi1, bigram_dict, tokens, vocab)) 
            + ( (trigram_dict.get((wi1, wi2)).get(wi0) + 1)/(bigram_dict.get(wi1).get(wi2) + len(vocab))))/2
    else:
        #Falls to the bigrams
        return probabilities(wi0, wi1, bigram_dict, tokens, vocab)
tokens = tokensfromNlyrics(5000)
vocab = create_vocab_dict(tokens)
bigram = create_bigram_matrix(tokens, vocab)
trigram = create_trigram_matrix(tokens, vocab)
print(probabilities("special", "midnight", bigram, tokens, vocab))
