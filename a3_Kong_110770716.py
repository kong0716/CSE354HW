#Darren Kong
#CSE354 HW 3
import  re, numpy as np, sklearn, scipy.stats, pandas, csv, gensim
from sklearn.linear_model import Ridge #just like L2 regularized logistic regression
from sklearn.decomposition import PCA #just like L2 regularized logistic regression
import scipy.stats as ss #for distributions
from gensim.models import Word2Vec
import sys
from collections import Counter, defaultdict

def main(argv):
    if len(argv) != 2:
        print("Needs a train and test file")
    else:
        print(argv[0])
        print(argv[1])

if __name__== '__main__':
    main(sys.argv[1:])