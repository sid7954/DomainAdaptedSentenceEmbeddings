import numpy as np 

g=open('amazon_cells_labelled.txt')
f=open('amazon_cells_labelled_sorted.txt','w')
pos=[]
neg=[]

for line in g:
	line=line.replace("\t"," ")
	line=line.replace("."," . ")
	line=line.replace("!"," ! ")
	line=line.replace("?"," ? ")
	line=line.replace(","," , ")
	_ = line.split(" ")
	if(_[-1] == '0\n' ):
		neg.append(line)
	else:
		pos.append(line)
for line in pos:
	f.write(line)
for line in neg:
	f.write(line)
