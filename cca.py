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

#f=open('kcca.pkl','wb')

train_data_size=1000
bert_dimension=768
cnn_dimension=384

bert_embeddings= np.loadtxt('data/sentiment/imdb_bert_embeddings.txt')
cnn_embeddings= np.loadtxt('data/sentiment/imdb_cnn_no_glove.txt')

y=np.zeros(train_data_size)
y[:500]=1


k = min(bert_dimension, cnn_dimension)

####### Linear CCA
cca = CCA(n_components=k,max_iter=200)
cca.fit(bert_embeddings,cnn_embeddings)
[train_vec1,train_vec2] = cca.transform(bert_embeddings,cnn_embeddings)
x=0.5*(train_vec1+train_vec2)
print("Done Linear CCA transform")

####### KCCA
# kcca = rcca.CCA(reg=0.01,numCC=k,kernelcca=True,ktype="gaussian",
#                   gausigma=2.5)
# cancomps=kcca.train([bert_embeddings,cnn_embeddings]).comps
# l_train = cancomps[0]
# g_train = cancomps[1]
# x=0.5*(l_train+g_train)
# print("Done KCCA transform")

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)

cca_precision=0
cca_f1 =0
cca_auc =0
cnn_precision=0
cnn_f1=0
cnn_auc=0
bert_precision=0
bert_f1=0
bert_auc=0
con_precision=0
con_f1=0
con_auc=0

itera=20
for iteration in range(itera):
	print(iteration)
	lr = LogisticRegression(penalty='l2',class_weight='balanced', fit_intercept=True,solver='newton-cg',multi_class='multinomial',max_iter=100)
	lr = lr.fit(x_train,y_train)
	y_pred = lr.predict(x_test)
	precision = precision_score(y_test,y_pred,average ='binary')
	f1 = f1_score(y_test,y_pred,average ='binary')
	auc = accuracy_score(y_test,y_pred)

	# print("CCA Transformed")
	# print(precision)
	# print(f1)
	# print(auc)
	cca_precision += precision
	cca_f1 += f1
	cca_auc += auc

	x_train, x_test, y_train, y_test = train_test_split(cnn_embeddings, y, stratify=y, test_size=0.2)
	lr = LogisticRegression(penalty='l2',class_weight='balanced', fit_intercept=True,solver='newton-cg',multi_class='multinomial',max_iter=100)
	lr = lr.fit(x_train,y_train)
	y_pred = lr.predict(x_test)
	precision = precision_score(y_test,y_pred,average ='binary')
	f1 = f1_score(y_test,y_pred,average ='binary')
	auc = accuracy_score(y_test,y_pred)

	# print("Cnn Embeddings")
	# print(precision)
	# print(f1)
	# print(auc)
	cnn_precision += precision
	cnn_f1 += f1
	cnn_auc += auc

	x_train, x_test, y_train, y_test = train_test_split(bert_embeddings, y, stratify=y, test_size=0.2)
	lr = LogisticRegression(penalty='l2',class_weight='balanced', fit_intercept=True,solver='newton-cg',multi_class='multinomial',max_iter=100)
	lr = lr.fit(x_train,y_train)
	y_pred = lr.predict(x_test)
	precision = precision_score(y_test,y_pred,average ='binary')
	f1 = f1_score(y_test,y_pred,average ='binary')
	auc = accuracy_score(y_test,y_pred)

	# print("BERT Embeddings")
	# print(precision)
	# print(f1)
	# print(auc)
	bert_precision += precision
	bert_f1 += f1
	bert_auc += auc

	x_train, x_test, y_train, y_test = train_test_split(np.concatenate((bert_embeddings,cnn_embeddings),axis=1), y, stratify=y, test_size=0.2)
	#print(np.shape(x_train))
	lr = LogisticRegression(penalty='l2',class_weight='balanced', fit_intercept=True,solver='newton-cg',multi_class='multinomial',max_iter=100)
	lr = lr.fit(x_train,y_train)
	y_pred = lr.predict(x_test)
	precision = precision_score(y_test,y_pred,average ='binary')
	f1 = f1_score(y_test,y_pred,average ='binary')
	auc = accuracy_score(y_test,y_pred)

	# print("Concat Embeddings")
	# print(precision)
	# print(f1)
	# print(auc)
	con_precision += precision
	con_f1 += f1
	con_auc += auc

print("Cnn Embeddings: ",cnn_precision/itera, cnn_f1/itera, cnn_auc/itera)
print("BERT Embeddings: ",bert_precision/itera, bert_f1/itera, bert_auc/itera)
print("CCA Embeddings: ",cca_precision/itera, cca_f1/itera, cca_auc/itera)
print("Concat Embeddings: ",con_precision/itera, con_f1/itera, con_auc/itera)