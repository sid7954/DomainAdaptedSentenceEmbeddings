import sys
import os
import pickle

import numpy as np

"""
Utility functions for handling dataset, embeddings and batches
"""

def convert_file(word_list, word_dict):
    #with open(filepath) as ifile:
    a=[word_dict.get(w, 0) for w in word_list]
    #print("yo")
    #print(a)
    return a


# def discover_dataset(wdict):
#     pos_dataset = []
#     neg_dataset = []
#     g=open("../data/sentiment/yelp_labelled_sorted.txt")
#     for line in g:
#         line=line.replace("\t"," ")
#         line=line.replace("."," . ")
#         line=line.replace("!"," ! ")
#         line=line.replace("?"," ? ")
#         line=line.replace(","," , ")
#         _ = line.split(" ")
#         x_ = _[:-1]
#         y_ = _[-1]
#         if(y_ == '0\n'):
#             neg_dataset.append(convert_file( x_ , wdict))
#         else:
#             pos_dataset.append(convert_file( x_ , wdict))
#     return pos_dataset, neg_dataset


def discover_dataset(wdict):
    X = []
    Y = []
    g=open("../data/sst-2/train_all.tsv")
    i=0
    count=0
    for line in g:
        if(i>0):
            line=line.split("\t")
            # print(len(line))
            x_ = line[0]
            y_ = line[1]
            X.append(convert_file( x_ , wdict))
            if(y_ == '0'):
                Y.append([0, 1])
                count+=1
            else:
                Y.append([1,0])
        i=i+1
    print("i ",i)
    print("count ",count)
    return X, Y

def pad_dataset(dataset, maxlen):
    return np.array(
        [np.pad(r, (0, maxlen-len(r)), mode='constant') if len(r) < maxlen else np.array(r[:maxlen])
         for r in dataset])


# Class for dataset related operations
class IMDBDataset():
    def __init__(self, dict_path, maxlen=128):
        # pos_path = os.path.join(path, 'pos')
        # neg_path = os.path.join(path, 'neg')

        with open(dict_path, 'rb') as dfile:
            wdict = pickle.load(dfile)

        self.X , self.Y = discover_dataset(wdict)
        self.X = pad_dataset(self.X, maxlen).astype('i')

    def __len__(self):
        return len(self.X)

    # def get_example(self, i):
    #     is_neg = i >= len(self.pos_dataset)
    #     dataset = self.neg_dataset if is_neg else self.pos_dataset
    #     idx = i - len(self.pos_dataset) if is_neg else i
    #     label = [0, 1] if is_neg else [1, 0]
        
    #     print (type(dataset[idx]))
    #     return (dataset[idx], np.array(label, dtype=np.int32))
    
    def load(self):
        print(len(self.X))        
        return self.X, np.array(self.Y, dtype=np.int32)


# Function for handling word embeddings
def load_embeddings(path, size, dimensions):
    
    embedding_matrix = np.zeros((size, dimensions), dtype=np.float32)

    # As embedding matrix could be quite big we 'stream' it into output file
    # chunk by chunk. One chunk shape could be [size // 10, dimensions].
    # So to load whole matrix we read the file until it's exhausted.
    size = os.stat(path).st_size
    with open(path, 'rb') as ifile:
        pos = 0
        idx = 0
        while pos < size:
            chunk = np.load(ifile)
            chunk_size = chunk.shape[0]
            embedding_matrix[idx:idx+chunk_size, :] = chunk
            idx += chunk_size
            pos = ifile.tell()
    return embedding_matrix

# Function for creating batches
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    print ("Generating batch iterator ...")
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

