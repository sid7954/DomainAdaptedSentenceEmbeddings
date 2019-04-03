import csv
import pandas as pd
# import nltk.data

def parse(path):
  g = open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

dataset = 'amazon'
path = '../' + dataset + 'tsv_bert_embeddings.txt'
out_path = '../' + dataset + '.tsv'

df = getDF(path)

reviews = []
labels = []

# print(df)		# df has embeddings even for the header row in case of custom embeddings
with open(out_path, 'r', encoding='utf-8') as emb_file:
	csv_reader = csv.reader(emb_file, delimiter="\t", quotechar=None)
	for row in csv_reader:
		reviews.append(row[0])
		labels.append(row[3])

# print(len(reviews))
# print(len(df['features']))
with open(out_path, 'w', encoding='utf-8') as emb_file:
	csv_writer = csv.writer(emb_file, delimiter='\t', quotechar=None)
	csv_writer.writerow(['Review', 'BERT Embeddings', 'CNN Embeddings', 'Sentiment'])
	idx = 0
	for example in df['features']:
		if idx > 0:
			feature_vec = str(example[0]['layers'][0]['values']).replace(',', '').replace('[', '').replace(']', '')
			# print(idx)
			csv_writer.writerow([reviews[idx], feature_vec, '', labels[idx]])
		idx += 1
