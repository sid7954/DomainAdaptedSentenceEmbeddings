import csv

dataset = 'imdb'

with open('../' + dataset + '.tsv', 'w', encoding='utf-8') as data_tsv, open('../' + dataset + '_labelled.txt', "r", encoding='utf-8') as data_txt:
    data_tsv.write('Review	BERT Embeddings	CNN Embeddings	Sentiment\n')
    txt_reader = csv.reader(data_txt, delimiter="\t", quotechar=None)
    for line in txt_reader:
        data_tsv.write(line[0] + '			' + line[1] + '\n')
