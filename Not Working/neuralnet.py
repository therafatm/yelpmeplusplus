import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPRegressor

def main(argv):

	if(len(argv) != 1):
		print("Defaulting to 100k reviews pickle")
		filetoread = '100kreviews.pickle'
	else:
		filetoread = str(argv[0])

	clf = MLPRegressor(hidden_layer_sizes=(5, 2), random_state=1)

	with open(filetoread,'rb') as fp:
		text, stars = pickle.load(fp)

	data_train, data_test, target_train, target_test = train_test_split(
		text, stars, test_size=0.4, random_state=0
	)

	vect = TfidfVectorizer(strip_accents = 'ascii', norm = 'l2' , stop_words = 'english', ngram_range = (1,3), use_idf = False, smooth_idf = False)
	print('Vectorizing training data')
	fittedAndTransformedTrainData = vect.fit_transform(data_train)
	print('Fitting vectorized data')
	clf.fit(fittedAndTransformedTrainData, target_train)

	testData = vect.transform(data_test)
	# print('Classification report')
	# print('_' * 40)
	print('Making some grate predictions.')
	predicted = clf.predict(testData)
	# print ('Accuracy score: %0.3f' % accuracy_score(target_test, predicted))
	# print('Classification report:')
	# print(classification_report(target_test, predicted))


	fp = open("MLPResults.txt", "w")
	for r,p, a in zip(data_test, predicted, target_test):
		review = "Review: " + str(r) + "\n"
		fp.write("%s" % review)
		predictedRating = "Predicted Rating: " + str(p) + "\n"
		fp.write("%s" % predictedRating)
		actualRating = "Actual Raiting:" + str(a) + "\n"
		fp.write("%s" % actualRating)
		seperator = "---------------" * 10 + "\n"
		fp.write("%s" % seperator)

	fp.close()

if __name__ == "__main__":
	main(sys.argv[1:])

