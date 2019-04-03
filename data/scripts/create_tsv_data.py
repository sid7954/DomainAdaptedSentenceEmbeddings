import csv

dataset = 'imdb'

sentences = open('../' + dataset + '/'+ dataset + '_unlabelled_sorted.txt', "r", encoding='utf-8')
bert = open('../' + dataset + '/'+ dataset + '_bert_embeddings.txt', "r", encoding='utf-8')
cnn_no_glove = open('../' + dataset + '/'+ dataset + '_cnn_no_glove.txt', "r", encoding='utf-8')
cnn_glove_non_trainable = open('../' + dataset + '/'+ dataset + '_cnn_embeddings.txt', "r", encoding='utf-8')
cnn_glove_trainable = open('../' + dataset + '/'+ dataset + '_cnn_trainable_glove.txt', "r", encoding='utf-8')

i=0

with open('../' + dataset + '/'+ dataset + '.tsv', 'w', encoding='utf-8') as data_tsv:
    data_tsv.write('Review\tSentiment\tBERT\tCNN_no_glove\tCNN_glove_non_trainable\tCNN_glove_trainable\n')
    for a,b,c,d,e in zip(sentences, bert,cnn_no_glove, cnn_glove_non_trainable,cnn_glove_trainable):
    	if (i<500):
        	data_tsv.write(a[:-1]+ '\t1\t' + b[:-1] + '\t' + c[:-1] + '\t' + d[:-1]+'\t' +e[:-1] +'\n')
    	else:
    		data_tsv.write(a[:-1]+ '\t0\t' + b[:-1] + '\t' + c[:-1] + '\t' + d[:-1]+'\t' +e[:-1] +'\n')
    	i=i+1