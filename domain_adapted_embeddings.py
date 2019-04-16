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
import math
import sys
import random


def cca_transform(embed1, embed2, ndim, n_iter):
	# cca = CCA(n_components=ndim, max_iter=n_iter)
	# cca.fit(embed1, embed2)
	# cancomps = cca.transform(embed1, embed2)
	cca = rcca.CCA(reg=0.01, numCC=ndim, kernelcca=False)
	cancomps = cca.train([embed1, embed2]).comps
	return 0.5 * (cancomps[0] + cancomps[1])


def kcca_transform(embed1, embed2, ndim, reg, gsigma):
	kcca = rcca.CCA(reg=reg, numCC=ndim, kernelcca=True, ktype='gaussian', gausigma=gsigma)
	cancomps = kcca.train([embed1, embed2]).comps
	return cancomps		# 0.5 * (cancomps[0] + cancomps[1])


def train_classifier(x_train, y_train, x_test, y_test):
	clf = LogisticRegression(penalty='l2', class_weight='balanced', fit_intercept=True, solver='newton-cg', multi_class='multinomial', max_iter=100)
	clf = clf.fit(x_train, y_train)

	y_pred = clf.predict(x_test)
	prec = precision_score(y_test, y_pred, average='binary')
	f1 = f1_score(y_test, y_pred, average ='binary')
	acc = accuracy_score(y_test, y_pred)
	return prec, f1, acc, clf


def intra_emb_analysis(vectors):
	angles = []
	for i in range(len(vectors)):
		for j in range(i + 1, len(vectors)):
			angles.append(angle_between_vecs(vectors[i], vectors[j]))
	print('Intra-angle between embeddings: {0} +- {1}'.format(np.mean(angles), np.std(angles)))


def analyse_embeddings(x_test, y_pred, experiment):
	if experiment == 'KCCA':
		pos_bert_embs = []
		neg_bert_embs = []
		pos_cnn_embs = []
		neg_cnn_embs = []

		bert_norms = []
		cnn_norms = []

		for i in range(len(y_pred)):
			bert_norms.append(np.linalg.norm(x_test['KCCA_0'].values[i]))
			cnn_norms.append(np.linalg.norm(x_test['KCCA_1'].values[i]))
			if y_pred[i] == 1:
				pos_bert_embs.append(np.subtract(x_test['KCCA_0'].values[i], x_test['KCCA'].values[i]))
				pos_cnn_embs.append(np.subtract(x_test['KCCA_1'].values[i], x_test['KCCA'].values[i]))
			elif y_pred[i] == 0:
				neg_bert_embs.append(np.subtract(x_test['KCCA_0'].values[i], x_test['KCCA'].values[i]))
				neg_cnn_embs.append(np.subtract(x_test['KCCA_1'].values[i], x_test['KCCA'].values[i]))
		
		mean_pos_bert_vec = np.mean(pos_bert_embs, axis=0)
		std_pos_bert_vec = np.std(pos_bert_embs, axis=0)
		mean_neg_bert_vec = np.mean(neg_bert_embs, axis=0)
		std_neg_bert_vec = np.std(neg_bert_embs, axis=0)
		mean_pos_cnn_vec = np.mean(pos_cnn_embs, axis=0)
		# std_pos_cnn_vec = np.std(pos_cnn_embs, axis=0)
		# mean_neg_cnn_vec
		# std_neg_cnn_vec

		print('BERT emb norm: {0} +- {1}\nCNN emb norm: {2} +- {3}'.format(np.mean(bert_norms), np.std(bert_norms), np.mean(cnn_norms), np.std(cnn_norms)))

		print('Printing vector embedding analysis:\nPos BERT mean and std:')
		print(mean_pos_bert_vec)
		# print(std_pos_bert_vec)
		# print('Neg BERT mean and std:')
		# print(mean_neg_bert_vec)
		# print(std_neg_bert_vec)

		print('Pos CNN mean and std:')
		print(mean_pos_cnn_vec)
		# print(mean_pos_cnn_vec)

		intra_emb_analysis(mean_pos_bert_vec)


def predicted_sentences(clf, experiment, x_test, y_test):
	correct_preds = set()
	incorrect_preds = set()
	y_pred = clf.predict(np.asarray(x_test[experiment].tolist()))
	# analyse_embeddings(x_test, y_pred, experiment)
	for i in range(len(y_pred)):
		if y_pred[i] == y_test[i]:
			correct_preds.add(x_test['Review'].values[i])
		else:
			incorrect_preds.add(x_test['Review'].values[i])
	return correct_preds, incorrect_preds


def angle_between_vecs(vec1, vec2):
	cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
	return np.degrees(math.acos(cos_sim))


def kcca_predictions_analysis(kcca_correct_bert_cnn_incorrect, x_test):
	results = []
	for pred in kcca_correct_bert_cnn_incorrect:
		df_row = x_test[x_test['Review'] == pred]
		review = pred
		bert_emb = df_row['KCCA_0']	# Projected BERT Embeddings
		cnn_emb = df_row['KCCA_1']	# Projected CNN Embeddings
		kcca_emb = df_row['KCCA']

		bert_kcca_angle_deg = angle_between_vecs(bert_emb.values[0], kcca_emb.values[0])
		cnn_kcca_angle_deg = angle_between_vecs(cnn_emb.values[0], kcca_emb.values[0])

		results.append([review, bert_emb, cnn_emb, kcca_emb])
		print(review)
		print(bert_kcca_angle_deg)
		print(cnn_kcca_angle_deg)
	return results


def main():
	dataset = sys.argv[1]   # amazon, yelp, imdb
	train_data_size=1000
	bert_dimension=768
	cnn_dimension=384

	df = pd.read_csv('data/' + dataset + '/' + dataset + '.tsv', delimiter='\t')
	df['BERT'] = df['BERT'].transform(np.fromstring, sep=' ')
	df['CNN'] = df['CNN_no_glove'].transform(np.fromstring, sep=' ')
	y = np.asarray(df['Sentiment'].values)

	# KCCA:
	kcca_xdim = min(bert_dimension, cnn_dimension)
	bert_embeddings = np.asarray(df['BERT'].tolist())
	cnn_embeddings = np.asarray(df['CNN'].tolist())
	kcca_x = kcca_transform(bert_embeddings, cnn_embeddings, kcca_xdim, 0.01, 2.5)

	# CCA:
	cca_xdim = min(bert_dimension, cnn_dimension)
	cca_x = cca_transform(bert_embeddings, cnn_embeddings, cca_xdim, 500)

	# Concatenate BERT and CNN:
	concat_x = np.concatenate((bert_embeddings, cnn_embeddings), axis=1)
	df['KCCA_0'] = pd.Series(map(lambda x:[x], kcca_x[0])).apply(lambda x:x[0])
	df['KCCA_1'] = pd.Series(map(lambda x:[x], kcca_x[1])).apply(lambda x:x[0])
	kcca_x = 0.5 * (kcca_x[0] + kcca_x[1])
	df['KCCA'] = pd.Series(map(lambda x:[x], kcca_x)).apply(lambda x:x[0])
	df['CCA'] = pd.Series(map(lambda x:[x], cca_x)).apply(lambda x:x[0])
	df['Concat'] = pd.Series(map(lambda x:[x], concat_x)).apply(lambda x:x[0])

	n_expts = 20
	seeds = random.sample(range(1, 1000), n_expts)
	kcca_precs = []
	kcca_f1s = []
	kcca_accs = []
	cca_precs = []
	cca_f1s = []
	cca_accs = []
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
		x_train, x_test, y_train, y_test = train_test_split(df, y, stratify=y, test_size=0.1, random_state=seeds[expt_id])

		# print('Experiment:', expt_id)
		cca_prec, cca_f1, cca_acc, cca_clf = train_classifier(np.asarray(x_train['CCA'].tolist()), y_train, np.asarray(x_test['CCA'].tolist()), y_test)
		cca_precs.append(cca_prec)
		cca_f1s.append(cca_f1)
		cca_accs.append(cca_acc)
		cca_correct_preds, _ = predicted_sentences(cca_clf, 'CCA', x_test, y_test)

		kcca_prec, kcca_f1, kcca_acc, kcca_clf = train_classifier(np.asarray(x_train['KCCA'].tolist()), y_train, np.asarray(x_test['KCCA'].tolist()), y_test)
		kcca_precs.append(kcca_prec)
		kcca_f1s.append(kcca_f1)
		kcca_accs.append(kcca_acc)
		kcca_correct_preds, _ = predicted_sentences(kcca_clf, 'KCCA', x_test, y_test)

		# Just CNN:
		cnn_prec, cnn_f1, cnn_acc, cnn_clf = train_classifier(np.asarray(x_train['CNN'].tolist()), y_train, np.asarray(x_test['CNN'].tolist()), y_test)
		cnn_precs.append(cnn_prec)
		cnn_f1s.append(cnn_f1)
		cnn_accs.append(cnn_acc)
		_, cnn_incorrect_preds = predicted_sentences(cnn_clf, 'CNN', x_test, y_test)
		
		# Just BERT without Finetuning:
		bert_prec, bert_f1, bert_acc, bert_clf = train_classifier(np.asarray(x_train['BERT'].tolist()), y_train, np.asarray(x_test['BERT'].tolist()), y_test)
		bert_precs.append(bert_prec)
		bert_f1s.append(bert_f1)
		bert_accs.append(bert_acc)
		_, bert_incorrect_preds = predicted_sentences(bert_clf, 'BERT', x_test, y_test)

		kcca_correct_bert_cnn_incorrect = kcca_correct_preds & bert_incorrect_preds & cnn_incorrect_preds
		kcca_predictions_analysis(kcca_correct_bert_cnn_incorrect, x_test)
		# print(kcca_correct_bert_cnn_incorrect)

		concat_prec, concat_f1, concat_acc, concat_clf = train_classifier(np.asarray(x_train['Concat'].tolist()), y_train, np.asarray(x_test['Concat'].tolist()), y_test)
		concat_precs.append(concat_prec)
		concat_f1s.append(concat_f1)
		concat_accs.append(concat_acc)
	
	print('\nSummary of {0} experiments:\n'.format(n_expts))
	print('CCA:\nPrecision: {0} +- {1}\nF1: {2} +- {3}\nAccuracy: {4} +- {5}\n'.format(np.mean(cca_precs), np.std(cca_precs), np.mean(cca_f1s), np.std(cca_f1s), np.mean(cca_accs), np.std(cca_accs)))
	print('KCCA:\nPrecision: {0} +- {1}\nF1: {2} +- {3}\nAccuracy: {4} +- {5}\n'.format(np.mean(kcca_precs), np.std(kcca_precs), np.mean(kcca_f1s), np.std(kcca_f1s), np.mean(kcca_accs), np.std(kcca_accs)))
	print('CNN:\nPrecision: {0} +- {1}\nF1: {2} +- {3}\nAccuracy: {4} +- {5}\n'.format(np.mean(cnn_precs), np.std(cnn_precs), np.mean(cnn_f1s), np.std(cnn_f1s), np.mean(cnn_accs), np.std(cnn_accs)))
	print('BERT:\nPrecision: {0} +- {1}\nF1: {2} +- {3}\nAccuracy: {4} +- {5}\n'.format(np.mean(bert_precs), np.std(bert_precs), np.mean(bert_f1s), np.std(bert_f1s), np.mean(bert_accs), np.std(bert_accs)))
	print('Concat:\nPrecision: {0} +- {1}\nF1: {2} +- {3}\nAccuracy: {4} +- {5}'.format(np.mean(concat_precs), np.std(concat_precs), np.mean(concat_f1s), np.std(concat_f1s), np.mean(concat_accs), np.std(concat_accs)))


if __name__ == "__main__":
	main()
