import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, explained_variance_score, mean_absolute_error, mean_squared_error

def main(argv):

	if(len(argv) != 1):
		print("Defaulting to 100k reviews pickle")
		filetoread = '100kreviews.pickle'
	else:
		filetoread = str(argv[0])

	clf = SGDClassifier(penalty='l2', alpha=1e-3, n_jobs= -1, n_iter=5, random_state=42) #loss='squared_epsilon_insensitive'

	with open(filetoread,'rb') as fp:
		data_train, data_test, target_train, target_test = pickle.load(fp)

	print('Fitting training data')
	clf.fit(data_train, target_train)

	print('Classification report')
	print('_' * 40)
	print('Predicting test data')
	predicted = clf.predict(data_test)
	print ('Accuracy score: %0.3f' % accuracy_score(target_test, predicted))
	print ('Variance: %0.3f' % explained_variance_score(target_test, predicted))
	print ('Mean absolute error: %0.3f' % mean_absolute_error(target_test, predicted))
	print ('Mean squared error: %0.3f' % mean_squared_error(target_test, predicted))
	print('Classification report:')
	print(classification_report(target_test, predicted))

	totalCorrect = 0
	totalSortOfCorrect = 0
	totalSortOfCorrectLess = 0
	totalSortOfCorrectGreater = 0

	for i in range(len(predicted)):
		if(predicted[i] == int(target_test[i])):
			totalCorrect += 1

		if(abs(predicted[i] - int(target_test[i])) == 1):
			totalSortOfCorrect += 1
			if( predicted[i] - int(target_test[i]) < 1):
				totalSortOfCorrectLess += 1
			else:
				totalSortOfCorrectGreater += 1

	totalSortOfCorrect += totalCorrect
	totalSortOfCorrectLess += totalCorrect
	totalSortOfCorrectGreater += totalCorrect

	print ('Off by 1 Accuracy: ' + str(totalSortOfCorrect*100/len(predicted)) + '%')
	print ('Off by 1 Greater Accuracy: ' + str(totalSortOfCorrectGreater*100/len(predicted)) + '%')
	print ('Off by 1 Less Accuracy: ' + str(totalSortOfCorrectLess*100/len(predicted)) + '%')

	print('Writing results to output file.')
	fp = open("SVCResults.txt", "w")
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
	print('Done.')

if __name__ == "__main__":
	main(sys.argv[1:])