import csv
import random
import numpy as np
from sklearn.metrics import precision_score, f1_score, roc_auc_score, accuracy_score

testset_path = '/srv/home/rohit.sharma/dev/glue_data/IMDB/test.tsv'
predictions_path = '/srv/home/rohit.sharma/dev/bert_finetuning_out/imdb_test/test_results.tsv'

def read_tsv(input_file, label_idx, quotechar=None):
	"""Reads a tab separated value file."""
	with open(input_file, "r", encoding='utf-8') as f:
		reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
		lines = []
		for line in reader:
			lines.append(float(line[label_idx]))
		return lines

y_test = np.asarray(read_tsv(testset_path, 1))
y_pred = np.asarray(read_tsv(predictions_path, 1))
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0

prec = precision_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average ='binary')
auc = accuracy_score(y_test, y_pred)

print('BERT Finetuning:\nPrecision: {0}\nF1: {1}\nAccuracy: {2}\n'.format(prec, f1, auc))
