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
import random


def cca_transform():
	cca = CCA(n_components=ndim, max_iter=500)
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
	acc = accuracy_score(y_test, y_pred)
	return prec, f1, acc


def main():
	dataset = sys.argv[1]   # amazon, yelp, imdb
	train_data_size=1000
	bert_dimension=768
	cnn_dimension=384

	df = pd.read_csv('data/' + dataset + '/' + dataset + '.tsv', delimiter='\t')
	bert_embeddings = np.asarray(df['BERT'].transform(np.fromstring, sep=' ').tolist())
	cnn_embeddings = np.asarray(df['CNN_no_glove'].transform(np.fromstring, sep=' ').tolist())
	y = np.asarray(df['Sentiment'].values)

	# print(bert_embeddings)
	# print(y)

	# KCCA:
	kcca_xdim = min(bert_dimension, cnn_dimension)
	kcca_x = kcca_transform(bert_embeddings, cnn_embeddings, kcca_xdim, 0.01, 2.5)
	# Concatenate BERT and CNN:
	concat_x = np.concatenate((bert_embeddings, cnn_embeddings), axis=1)

	n_expts = 20
	seeds = random.sample(range(1, 1000), n_expts)
	kcca_precs = []
	kcca_f1s = []
	kcca_accs = []
	cnn_precs = []
	cnn_f1s = []
	cnn_accs = []
	bert_precs = []
	bert_f1s = []
	bert_accs = []
	concat_precs = []
	concat_f1s = []
	concat_accs = []

	for expt_id in range(n_expts):
		# print('Experiment:', expt_id)
		x_train, x_test, y_train, y_test = train_test_split(kcca_x, y, stratify=y, test_size=0.1, random_state=seeds[expt_id])
		kcca_prec, kcca_f1, kcca_acc = train_classifier(x_train, y_train, x_test, y_test)
		kcca_precs.append(kcca_prec)
		kcca_f1s.append(kcca_f1)
		kcca_accs.append(kcca_acc)
		# print('KCCA:\nPrecision: {0}\nF1: {1}\nAccuracy: {2}\n'.format(kcca_prec, kcca_f1, kcca_acc))

		# Just CNN:
		x_train, x_test, y_train, y_test = train_test_split(cnn_embeddings, y, stratify=y, test_size=0.1, random_state=seeds[expt_id])
		cnn_prec, cnn_f1, cnn_acc = train_classifier(x_train, y_train, x_test, y_test)
		cnn_precs.append(cnn_prec)
		cnn_f1s.append(cnn_f1)
		cnn_accs.append(cnn_acc)
		# print('CNN:\nPrecision: {0}\nF1: {1}\nAccuracy: {2}\n'.format(cnn_prec, cnn_f1, cnn_acc))
		
		# Just BERT without Finetuning:
		x_train, x_test, y_train, y_test = train_test_split(bert_embeddings, y, stratify=y, test_size=0.1, random_state=seeds[expt_id])
		bert_prec, bert_f1, bert_acc = train_classifier(x_train, y_train, x_test, y_test)
		bert_precs.append(bert_prec)
		bert_f1s.append(bert_f1)
		bert_accs.append(bert_acc)
		# print('BERT:\nPrecision: {0}\nF1: {1}\nAccuracy: {2}\n'.format(bert_prec, bert_f1, bert_acc))

		x_train, x_test, y_train, y_test = train_test_split(concat_x, y, stratify=y, test_size=0.1, random_state=seeds[expt_id])
		concat_prec, concat_f1, concat_acc = train_classifier(x_train, y_train, x_test, y_test)
		concat_precs.append(concat_prec)
		concat_f1s.append(concat_f1)
		concat_accs.append(concat_acc)
		# print('Concatenate:\nPrecision: {0}\nF1: {1}\nAccuracy: {2}'.format(concat_prec, concat_f1, concat_acc))
	
	print('\nSummary of {0} experiments:\n'.format(n_expts))
	print('KCCA:\nPrecision: {0} +- {1}\nF1: {2} +- {3}\nAccuracy: {4} +- {5}\n'.format(np.mean(kcca_precs), np.std(kcca_precs), np.mean(kcca_f1s), np.std(kcca_f1s), np.mean(kcca_accs), np.std(kcca_accs)))
	print('CNN:\nPrecision: {0} +- {1}\nF1: {2} +- {3}\nAccuracy: {4} +- {5}\n'.format(np.mean(cnn_precs), np.std(cnn_precs), np.mean(cnn_f1s), np.std(cnn_f1s), np.mean(cnn_accs), np.std(cnn_accs)))
	print('BERT:\nPrecision: {0} +- {1}\nF1: {2} +- {3}\nAccuracy: {4} +- {5}\n'.format(np.mean(bert_precs), np.std(bert_precs), np.mean(bert_f1s), np.std(bert_f1s), np.mean(bert_accs), np.std(bert_accs)))
	print('Concat:\nPrecision: {0} +- {1}\nF1: {2} +- {3}\nAccuracy: {4} +- {5}'.format(np.mean(concat_precs), np.std(concat_precs), np.mean(concat_f1s), np.std(concat_f1s), np.mean(concat_accs), np.std(concat_accs)))


if __name__ == "__main__":
	main()
