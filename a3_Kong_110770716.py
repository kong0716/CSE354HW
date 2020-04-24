#Darren Kong
#CSE354 HW 3
import  re, numpy as np, sklearn, scipy.stats, pandas, csv, gensim
from sklearn.linear_model import Ridge #just like L2 regularized logistic regression
from sklearn.decomposition import PCA #just like L2 regularized logistic regression
import scipy.stats as ss #for distributions
from gensim.models import Word2Vec
from happiestfuntokenizing.happiestfuntokenizing import Tokenizer
import sys
from collections import Counter, defaultdict

def preparecsv(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        id_dict = dict()
        for row in reader:
            unique_id = row['id']
            id_dict.update({unique_id : [row['rating'], row['vote'], row['asin'], row['user_id'], row['reviewText']]})
        #print(id_dict)
        return id_dict

def main(argv):
    if len(argv) != 2:
        print("Needs a train and test file")
    else:
        print(argv[0])
        print(argv[1])
        train_csv = preparecsv(argv[0])
        test_csv = preparecsv(argv[1])
        return train_csv, test_csv
        
if __name__== '__main__':
    main(sys.argv[1:])