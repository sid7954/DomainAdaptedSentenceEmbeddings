import numpy as np 

g=open('yelp_cnn_embeddings.txt')
f=open('yelp_cnn_embeddings2.txt','w')


for line in g:
	line=line.replace("]","\n")
	line=line.replace("[","")
	f.write(line)
