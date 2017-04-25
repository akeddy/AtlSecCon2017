#!/usr/bin/env python

'''
	A script designed to investigate text feature creation and prediction on labeled test data
	from OSSEC for AtlSecCon 2017

	Developed By:	Alex Keddy
					April 23, 2017
'''

import sklearn
import time, os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score as accscore
from sklearn.metrics import roc_auc_score as auc
from sklearn import svm	
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as nn
from sklearn.model_selection import train_test_split

def addLabels( labels ):
	result = []
	for l in labels:
		l = int(l)
		if l == 0:
			result.append('safe')
		elif l == 1:
			result.append('mal')
	return result

def balAcc( confMat ):
	tp = confMat[0][0]
	fn = confMat[0][1]
	fp = confMat[1][0]
	tn = confMat[1][1]
	pos = 1
	neg = 1
	if tp+fp > 0:
		pos = float(tp) / (tp+fp)
	if tn+fn > 0:
		neg = float(tn) / (tn+fn)
	return ( pos + neg )*0.5
	
datfh  = os.getcwd() + os.sep +  "ossecRulesExample.csv"

dataset = np.loadtxt( datfh, delimiter=",", dtype=str, skiprows = 1 )

print "Loaded data with shape:"
print dataset.shape

descriptions = dataset[:,0]
realFlags = map(int,dataset[:,1])
scrambleFlags = map(int,dataset[:,2])
randomFlags = map(int,dataset[:,3])
	
# shuffle all of these properties in the same way
descriptions, realFlags, scrambleFlags, randomFlags = shuffle( descriptions, realFlags, scrambleFlags, randomFlags )

trainSize = int(len( descriptions ) * 0.70)


trainDesc, testDesc, trainFlags, testFlags = train_test_split( descriptions, realFlags, test_size = (1-trainSize), random_state = 0 )

stopWords = stopwords.words('english')

count_v = CountVectorizer(stop_words=stopWords, ngram_range=(1,2))
trainBag = count_v.fit_transform(trainDesc)
#	perform a tf-idf transformation because it's ballin
tfidf_transformer = TfidfTransformer()
trainTFIDF = tfidf_transformer.fit_transform(trainBag)
testBag = count_v.transform( testDesc )
testTFIDF = tfidf_transformer.transform( testBag )
clfs = [svm.SVC(kernel='linear'), nn(solver="lbfgs"),knn(n_neighbors = 10)]
headers  = ["SVM","NN","KNN"]
for index, clf in enumerate(clfs):
	print headers[index]
	start = time.time()
	clf.fit( trainTFIDF.toarray(), trainFlags)

	predicted = clf.predict(testTFIDF.toarray())

	print "\tTime: %s" % (time.time() - start)
	labels = ['safe','mal']
	labeledPredicted = addLabels(predicted)
	labeledTest = addLabels( testFlags )
 
	print "\tConfusion Matrix Scores: "
	confusion = cm( labeledTest, labeledPredicted, labels )
	print "\t\tPred0 Pred1\n\tTrue0\t%s\n\tTrue1\t%s" %(confusion[0],confusion[1])
	
	print "\tAccuracy: "
	score = accscore(testFlags,predicted)
	print "\t%.4f" % score
	
	print "\tBalanced Accuracy: "
	score = balAcc( confusion )
	print "\t%.4f" % score
	

	print "\tAUC of ROC:"
	aucScore = auc(np.array(testFlags), np.array(predicted) )
	print "\t%.4f" % aucScore
	
	print "\n"

