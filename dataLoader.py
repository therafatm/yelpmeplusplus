import json
import sys
import pickle
from time import time
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def size_mb(obj):
	return sys.getsizeof(obj)/ 1e6

def load_dict(type, select_names, numb_objects = 0, verbose = True): 	
	print('Loading Data:')
	names_dict = defaultdict(list)
	object_count = 0
	with open('dataset' + type + '.json') as fp:

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

	if len(argv) != 1:
		print('Specify the number of objects to load.')
		print('An input of 0 will load all availabe objects.')
		sys.exit()

	numb_objects = int(argv[0])
	reviews = load_dict('review', ['text', 'stars'], numb_objects)
	print('%0.3fMB of data\n' % (size_mb(reviews['text']) + size_mb(reviews['stars'])))

	vect = TfidfVectorizer(strip_accents = 'ascii', stop_words = 'english')

	data_train, data_test, target_train, target_test = train_test_split(
		reviews['text'], reviews['stars'], test_size=0.3, random_state=0
	)

	print('Vectoizing training data')
	data_train = vect.fit_transform(data_train)
	data_test = vect.transform(data_test)

	data_target = (data_train, data_test, target_train, target_test)

	with open('TfidfVectoridedData'+ str(numb_objects/1000) +'k.pickle','wb') as fp:
		pickle.dump(data_target,fp,protocol = 4)

if __name__ == "__main__":
	main(sys.argv[1:])