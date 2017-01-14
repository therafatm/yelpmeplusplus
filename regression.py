import sys
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report
import math

def main(argv):

	if(len(argv) != 1):
		print("Defaulting to 100k reviews pickle")
		filetoread = '100kreviews.pickle'
	else:
		filetoread = str(argv[0])

	with open(filetoread,'rb') as fp:
		text, stars = pickle.load(fp)

	clf = linear_model.LinearRegression(copy_X = False, n_jobs = -1)

	data_train, data_test, target_train, target_test = train_test_split(
		text, stars, test_size=0.3, random_state=0
	)

	vect = TfidfVectorizer(strip_accents = 'ascii', stop_words = 'english', ngram_range = (1,2), use_idf = True, smooth_idf = True)
	print('Vectorizing training data')
	fittedAndTransformedTrainData = vect.fit_transform(data_train)
	print('Fitting training data')
	clf.fit(fittedAndTransformedTrainData, target_train)

	print('Transforming test data')
	testData = vect.transform(data_test)

	print('Predicting test data')
	predicted = clf.predict(testData)

	totalCorrect = 0
	totalSortOfCorrect = 0
	totalSortOfCorrectGreater = 0
	totalSortOfCorrectLess = 0

	for i in range(len(predicted)):
		if(predicted[i] < 5):
			diff = predicted[i] - int(predicted[i])
			if(diff < 0.5):
				predicted[i] = int(predicted[i])
			else:
				predicted[i] = int(predicted[i]) + 1

			if(predicted[i] == 0):
				predicted[i] = 1
		else:
			predicted[i] = math.floor(predicted[i])

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

	print ('Explained Variance Score: %0.3f' % explained_variance_score(target_test, predicted))
	print ('Mean Absolute Error: %0.3f' % mean_absolute_error(target_test, predicted))
	print ('Mean Squared Error: %0.3f' % mean_squared_error(target_test, predicted))
	print ('R2 Score: %0.3f' % r2_score(target_test, predicted))
	print ('Accuracy: ' + str(totalCorrect*100/len(predicted)) + '%')
	print ('Off by 1 Accuracy: ' + str(totalSortOfCorrect*100/len(predicted)) + '%')
	print ('Off by 1 Greater Accuracy: ' + str(totalSortOfCorrectGreater*100/len(predicted)) + '%')
	print ('Off by 1 Less Accuracy: ' + str(totalSortOfCorrectLess*100/len(predicted)) + '%')

	fp = open("RegressionResults.txt", "w")
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
