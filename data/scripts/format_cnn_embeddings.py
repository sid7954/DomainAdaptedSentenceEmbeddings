import numpy as np 

g=open('../imdb_cnn_trainable_glove.txt')
f=open('imdb_cnn_trainable_glove.txt','w')


for line in g:
	line=line.replace("]","\n")
	line=line.replace("[","")
	f.write(line)
