import csv
import random

dataset_path = '/srv/home/rohit.sharma/dev/glue_data/Yelp/test.tsv'

def read_tsv(input_file, quotechar=None):
	"""Reads a tab separated value file."""
	with open(input_file, "r", encoding='utf-8') as f:
		reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
		lines = []
		for line in reader:
			lines.append(line)
		return lines

def create_examples(lines, set_type):
	"""Creates examples for the training and dev sets."""
	examples = []
	for (i, line) in enumerate(lines):
		guid = "%s-%s" % (set_type, i)
		text_a = line[0]
		if set_type == "test":
			label = "0"
		else:
			label = line[1]
			print(text_a, label)
	return examples

create_examples(read_tsv(dataset_path), "train")

# lines = open(dataset_path, encoding='utf-8').readlines()
# random.shuffle(lines)
# open(dataset_path, 'w', encoding='utf-8').writelines(lines)
