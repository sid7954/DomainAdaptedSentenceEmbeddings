import numpy as np 

# g=open('imdb_labelled_sorted.txt')
# f=open('imdb_vocab.txt',"w")

# a=set()
# for line in g:
# 	line=line.replace("\t"," ")
# 	line=line.replace("."," . ")
# 	line=line.replace("!"," ! ")
# 	line=line.replace("?"," ? ")
# 	line=line.replace(","," , ")
# 	_ = line.split(" ")
# 	_ = _[:-1]
# 	#print(_)
# 	for word in _:
# 		if word not in a:
# 			a.add(word)

# for word in a:
# 	f.write(word + "\n")
# print(len(a))

g1=open('../sst-2/train_all.tsv')
g2=open('../sst-2/dev_all.tsv')
g3=open('../sst-2/test_all.tsv')
f=open('../sst-2/sst2_vocab.txt',"w")

a=set()
i=0
for line in g1:
	if(i>0):
		_ = line.split("\t")
		# print(len(_))
		line = _[0]
		line=line.replace("\t"," ")
		line=line.replace("."," . ")
		line=line.replace("!"," ! ")
		line=line.replace("?"," ? ")
		line=line.replace(","," , ")
		_ = line.split(" ")
		for word in _:
			if word not in a:
				a.add(word)
	i=i+1

i=0
for line in g2:
	if(i>0):
		_ = line.split("\t")
		# print(len(_))
		line = _[0]
		line=line.replace("\t"," ")
		line=line.replace("."," . ")
		line=line.replace("!"," ! ")
		line=line.replace("?"," ? ")
		line=line.replace(","," , ")
		_ = line.split(" ")
		for word in _:
			if word not in a:
				a.add(word)
	i=i+1

i=0
for line in g3:
	if(i>0):
		_ = line.split("\t")
		# print(len(_))
		line = _[0]
		line=line.replace("\t"," ")
		line=line.replace("."," . ")
		line=line.replace("!"," ! ")
		line=line.replace("?"," ? ")
		line=line.replace(","," , ")
		_ = line.split(" ")
		for word in _:
			if word not in a:
				a.add(word)
	i=i+1

print(i)
for word in a:
	f.write(word + "\n")
print(len(a))