import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report, explained_variance_score
from sklearn.naive_bayes import MultinomialNB

def plot_learning_curve(train_sizes, train_scores, test_scores):

	plt.figure()
	plt.title("Learning Curves (Naive Bayes)")
	#plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	train_scores_mean + train_scores_std, alpha=0.1,color="r")

	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	test_scores_mean + test_scores_std, alpha=0.1, color="g")

	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
	label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
	label="Cross-validation score")

	plt.legend(loc="best")
	plt.show()

def main(argv):

	clf = MultinomialNB(alpha = 0.03)

	if(len(argv) != 1):
		print("Defaulting to 100k reviews pickle")
		filetoread = '100kreviews.pickle'
	else:
		filetoread = str(argv[0])
	
	with open(filetoread,'rb') as fp:
		text, stars = pickle.load(fp)

	data_train, data_test, target_train, target_test = train_test_split(
		text, stars, test_size=0.4, random_state=0
	)

	vect = TfidfVectorizer(strip_accents = 'ascii', norm = 'l2' , stop_words = 'english', ngram_range = (1,2), use_idf = False, smooth_idf = False)
	print('Vectorizing training data')
	fittedAndTransformedTrainData = vect.fit_transform(data_train)
	clf.fit(fittedAndTransformedTrainData, target_train)

	testData = vect.transform(data_test)

	print('Classification report')
	print('_' * 40)
	predicted = clf.predict(testData)
	print ('Accuracy score: %0.3f' % accuracy_score(target_test, predicted))
	print ('Variance: %0.3f' % explained_variance_score(target_test, predicted))
	print('Classification report:')
	print(classification_report(target_test, predicted))

	totalCorrect = 0
	totalSortOfCorrect = 0
	totalSortOfCorrectGreater = 0
	totalSortOfCorrectLess = 0

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

	fp = open("MNBClassificationResults.txt", "w")
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

	ratingsToSentiment = { '1' : "Utterly disgusted.\n", '2' : 'Meh\n', '3' : "Not bad at all.\n", '4' : 'That was pretty great!\n', '5' : 'Mind = Blown!\n'}
	while(True):
		rev = input("Hey yo! Let me tell you how you feel!\nOr say 'quit' if you want to quit\n------>:  ")
		if(rev == "quit"):
			print("Bye! :(")
			exit(0)

		if(rev == " "):
			print("Try again. And write something pls.\n")
			continue

		print(rev)
		transformed = vect.transform([rev])
		predicted = clf.predict(transformed)
		print( "*" * 40)
		print("You feel: " + str(predicted[0]) + "  " + "-" * 5 + "  " + ratingsToSentiment[str(predicted[0])])
		print( "*" * 40)		
		print("I might've been completely wrong. Try again?")

if __name__ == "__main__":
	main(sys.argv[1:])