import json
import sys
import pickle
import os
from time import time
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

def size_mb(obj):
	return sys.getsizeof(obj)/ 1e6

def load_dict(dirpath, type, select_names, numb_objects = 0, verbose = True): 	
	print('Loading Data:')
	names_dict = defaultdict(list)
	object_count = 0
	with open( dirpath + 'dataset' + type + '.json') as fp:

		if numb_objects == 0:
			for line in fp:
				all_names = json.loads(line)
				for k in all_names:
					if k in select_names:
						names_dict[k].append(all_names[k])
				object_count += 1		
				if verbose == True and object_count % 1000 == 0:
					print('reading object %d ' % object_count, end='\r')
		else:
			line = fp.readline()
			while line and object_count != numb_objects:
				all_names = json.loads(line)
				for k in all_names:
					if k in select_names:
						names_dict[k].append(all_names[k])
				line = fp.readline()		
				object_count += 1
				if verbose == True and object_count % 1000 == 0:
					print('reading object %d ' % object_count, end='\r')
		print(' ', end = '\r')	
	return names_dict

def main(argv):

	if len(argv) < 3:
		if(os.path.isfile("100kreviews.pickle")):
			print("Try again with something like 'python3 dataLoaderNoVectorizer.py numObjects dataFolder dumpFileName'")
			sys.exit()

	numb_objects = int(argv[0])
	dirpath = str(argv[1])
	filename = ""

	if(len(argv) == 2):
		print("Setting number of objects to 100k due to incomplete command line args")
		print("Setting dump file name to: 100kreviews.pickle")
		filename = "100kreviews"
		numb_objects = 100000

	if(len(filename) == 0):
		filename = str(argv[2])

	filename = filename + ".pickle"

	reviews = load_dict(dirpath, 'review', ['text', 'stars'], numb_objects)
	print('%0.3fMB of data\n' % (size_mb(reviews['text']) + size_mb(reviews['stars'])))

	##avoid reviews which are less than 10 words.
	itemsToRemove = []
	for i in range(len(reviews['text'])):
		if(len(reviews['text'][i].split(" ")) < 10):
			itemsToRemove.append(i)

	diff = 0
	for i in itemsToRemove:
		if(i > 0):
			i = i - diff
		del reviews['stars'][i]
		del reviews['text'][i]
		diff += 1

	print("Reviews left: " + str(len(reviews['stars'])))
	print('Rounding star ratings to the nearest integer')
	rounded_stars = [round(elm) for elm in reviews['stars']]

	data_stars = (reviews['text'], rounded_stars)
	with open(filename,'wb') as fp:
		pickle.dump(data_stars,fp,protocol = 4)

if __name__ == "__main__":
	main(sys.argv[1:])