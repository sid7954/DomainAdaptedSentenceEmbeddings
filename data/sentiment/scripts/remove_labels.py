
dataset = 'yelp'

labelled_file = open('../' + dataset + '_labelled_sorted.txt', 'r', encoding='utf-8')
with open('../' + dataset + '_unlabelled_sorted.txt', 'w', encoding='utf-8') as unlabelled_file:
	for line in labelled_file:
		unlabeled_line = line[:-4]
		unlabelled_file.write(unlabeled_line + '\n')
labelled_file.close()
