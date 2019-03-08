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

dataset = 'imdb'
path = '../' + dataset + '_bert_embeddings.txt'

df = getDF(path)
# print(df)
with open(path, 'w') as emb_file:
	for example in df['features']:
		feature_vec = str(example[0]['layers'][0]['values']).replace(',', '').replace('[', '').replace(']', '')
		emb_file.write(feature_vec + '\n')
