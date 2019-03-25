import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import TruncatedSVD
from scipy import spatial
from scipy.spatial.distance import pdist,squareform
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, f1_score, roc_auc_score, accuracy_score
import time
from sklearn.model_selection import train_test_split
import pickle
import rcca
import sys


def get_embeddings(dataset):
    bert_embeddings = np.loadtxt('data/sentiment/' + dataset + '_bert_embeddings.txt')
    cnn_embeddings = np.loadtxt('data/sentiment/' + dataset + '_cnn_embeddings.txt')
    return bert_embeddings, cnn_embeddings


def cca_transform():
    cca = CCA(n_components=ndim, max_iter=200)
    cca.fit(embed1, embed2)
    cancomps = cca.transform(embed1, embed2)
    return 0.5 * (cancomps[0] + cancomps[1])


def kcca_transform(embed1, embed2, ndim, reg, gsigma):
    kcca = rcca.CCA(reg=reg, numCC=ndim, kernelcca=True, ktype='gaussian', gausigma=gsigma)
    cancomps = kcca.train([embed1, embed2]).comps
    return 0.5 * (cancomps[0] + cancomps[1])


def train_classifier(x_train, y_train, x_test, y_test):
    clf = LogisticRegression(penalty='l2', class_weight='balanced', fit_intercept=True, solver='newton-cg', multi_class='multinomial', max_iter=100)
    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    prec = precision_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average ='binary')
    auc = accuracy_score(y_test, y_pred)
    return prec, f1, auc


def main():
    dataset = sys.argv[1]   # amazon, yelp, imdb
    train_data_size=1000
    bert_dimension=768
    cnn_dimension=384

    bert_embeddings, cnn_embeddings = get_embeddings(dataset)
    y = np.zeros(train_data_size)
    y[:500] = 1

    # KCCA:
    kcca_xdim = min(bert_dimension, cnn_dimension)
    kcca_x = kcca_transform(bert_embeddings, cnn_embeddings, kcca_xdim, 0.01, 2.5)
    x_train, x_test, y_train, y_test = train_test_split(kcca_x, y, stratify=y, test_size=0.1)
    kcca_prec, kcca_f1, kcca_auc = train_classifier(x_train, y_train, x_test, y_test)
    print('KCCA:\nPrecision: {0}\nF1: {1}\nAccuracy: {2}\n'.format(kcca_prec, kcca_f1, kcca_auc))

    # Just CNN:
    x_train, x_test, y_train, y_test = train_test_split(cnn_embeddings, y, stratify=y, test_size=0.1)
    cnn_prec, cnn_f1, cnn_auc = train_classifier(x_train, y_train, x_test, y_test)
    print('CNN:\nPrecision: {0}\nF1: {1}\nAccuracy: {2}\n'.format(cnn_prec, cnn_f1, cnn_auc))
    
    # Just BERT without Finetuning:
    x_train, x_test, y_train, y_test = train_test_split(bert_embeddings, y, stratify=y, test_size=0.1)
    bert_prec, bert_f1, bert_auc = train_classifier(x_train, y_train, x_test, y_test)
    print('BERT:\nPrecision: {0}\nF1: {1}\nAccuracy: {2}\n'.format(bert_prec, bert_f1, bert_auc))

    # Concatenate BERT and CNN:
    concat_x = np.concatenate((bert_embeddings, cnn_embeddings), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(concat_x, y, stratify=y, test_size=0.1)
    concat_prec, concat_f1, concat_auc = train_classifier(x_train, y_train, x_test, y_test)
    print('Concatenate:\nPrecision: {0}\nF1: {1}\nAccuracy: {2}'.format(concat_prec, concat_f1, concat_auc))


if __name__ == "__main__":
    main()
