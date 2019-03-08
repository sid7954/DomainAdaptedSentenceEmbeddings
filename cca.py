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

f=open('kcca.pkl','wb')

train_data_size=1000
bert_dimension=768
cnn_dimension=384

bert_embeddings= np.loadtxt('data/sentiment/amazon_bert_embeddings.txt')
cnn_embeddings= np.loadtxt('data/sentiment/amazon_cnn_embeddings.txt')

y=np.zeros(train_data_size)
y[:500]=1


k = min(bert_dimension, cnn_dimension)
# cca = CCA(n_components=k,max_iter=200)
# cca.fit(bert_embeddings,cnn_embeddings)
#[train_vec1,train_vec2] = cca.transform(bert_embeddings,cnn_embeddings)
kcca = rcca.CCA(reg=0.01,numCC=k,kernelcca=True,ktype="gaussian",
                  gausigma=2.5)
cancomps=kcca.train([bert_embeddings,cnn_embeddings]).comps
l_train = cancomps[0]
g_train = cancomps[1]
x=0.5*(l_train+g_train)
print("Done KCCA transform")

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.1)


lr = LogisticRegression(penalty='l2',class_weight='balanced', fit_intercept=True,solver='newton-cg',multi_class='multinomial',max_iter=100)
lr = lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
precision = precision_score(y_test,y_pred,average ='binary')
f1 = f1_score(y_test,y_pred,average ='binary')
auc = accuracy_score(y_test,y_pred)

print("CCA Transformed")
print(precision)
print(f1)
print(auc)

x_train, x_test, y_train, y_test = train_test_split(cnn_embeddings, y, stratify=y, test_size=0.1)
lr = LogisticRegression(penalty='l2',class_weight='balanced', fit_intercept=True,solver='newton-cg',multi_class='multinomial',max_iter=100)
lr = lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
precision = precision_score(y_test,y_pred,average ='binary')
f1 = f1_score(y_test,y_pred,average ='binary')
auc = accuracy_score(y_test,y_pred)

print("Cnn Embeddings")
print(precision)
print(f1)
print(auc)

x_train, x_test, y_train, y_test = train_test_split(bert_embeddings, y, stratify=y, test_size=0.1)
lr = LogisticRegression(penalty='l2',class_weight='balanced', fit_intercept=True,solver='newton-cg',multi_class='multinomial',max_iter=100)
lr = lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
precision = precision_score(y_test,y_pred,average ='binary')
f1 = f1_score(y_test,y_pred,average ='binary')
auc = accuracy_score(y_test,y_pred)

print("BERT Embeddings")
print(precision)
print(f1)
print(auc)

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((bert_embeddings,cnn_embeddings),axis=1), y, stratify=y, test_size=0.1)
print(np.shape(x_train))
lr = LogisticRegression(penalty='l2',class_weight='balanced', fit_intercept=True,solver='newton-cg',multi_class='multinomial',max_iter=100)
lr = lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
precision = precision_score(y_test,y_pred,average ='binary')
f1 = f1_score(y_test,y_pred,average ='binary')
auc = accuracy_score(y_test,y_pred)

print("Concat Embeddings")
print(precision)
print(f1)
print(auc)

