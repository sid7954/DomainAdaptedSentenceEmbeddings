import numpy as np 

g=open('imdb_labelled_sorted.txt')
f=open('imdb_vocab.txt',"w")

a=set()
for line in g:
	line=line.replace("\t"," ")
	line=line.replace("."," . ")
	line=line.replace("!"," ! ")
	line=line.replace("?"," ? ")
	line=line.replace(","," , ")
	_ = line.split(" ")
	_ = _[:-1]
	#print(_)
	for word in _:
		if word not in a:
			a.add(word)

for word in a:
	f.write(word + "\n")
print(len(a))