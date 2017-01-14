import json
import sys
import pickle
from time import time
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

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
	location = load_dict('business', ['city','latitude', 'longitude', 'stars'], numb_objects)

	with open('LocationData'+ str(numb_objects) +'.pickle','wb') as fp:
		pickle.dump(location,fp,protocol = 4)

if __name__ == "__main__":
	main(sys.argv[1:])