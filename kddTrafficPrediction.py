#!/usr/bin/env python

'''
	A script designed to investigate text feature creation and prediction on labeled test data
	from OSSEC for AtlSecCon 2017

	Developed By:	Alex Keddy
					April 23, 201y
'''

import sklearn
import time, os
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score as accscore
from sklearn.metrics import roc_auc_score as auc
from sklearn import svm	
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as nn

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
	tn = confMat[0][0]
	fp = confMat[0][1]
	fn = confMat[1][0]
	tp = confMat[1][1]
	pos = 1
	neg = 1
	if tp+fp > 0:
		pos = float(tp) / (tp+fp)
	if tn+fn > 0:
		neg = float(tn) / (tn+fn)
	return ( pos + neg )*0.5
	
datfh  = os.getcwd() + os.sep +  "kddcup.data_10_percent_corrected"

dataset = np.loadtxt( datfh, delimiter=",", dtype=str, skiprows = 0 )

print "Loaded data with shape:"
print dataset.shape
#discovery stage!
if False:
	print np.unique(dataset[:,-1])
	print np.unique(dataset[:,1])
	print np.unique(dataset[:,2])
# normal data
negData = dataset[np.where(dataset[:,-1] == 'normal.'),:][0]
print negData.shape
negData[:,-1] = int(0)
negData = negData[0:10000,:]
# throw in some noise!
negData[0:500,-1] = int(1)
# buffer overflow attacks
bufferO = dataset[np.where(dataset[:,-1] == 'buffer_overflow.'),:][0]
bufferO[:,-1] = int(1)
print bufferO.shape
# instances of rootkits
rootkit = dataset[np.where(dataset[:,-1] == 'rootkit.'),:][0]
rootkit[:,-1] = int(1)
print rootkit.shape
# spyware
spy = dataset[np.where(dataset[:,-1] == 'satan.'),:][0]
spy[:,-1] = int(1)
print spy.shape
# teardrop attacks
tear = dataset[np.where(dataset[:,-1] == 'teardrop.'),:][0]
tear[:,-1] = int(1)
print tear.shape

data = np.vstack((negData,bufferO,rootkit,spy,tear))

# this is kind of cheating. A classifier will try to find relationships between these numbers. Really we want them to be treated as categories
transport = {'tcp':0,'udp':1,'icmp':2}
protocol = {'http':0,'smtp':1,'finger':2,'domain':3,'auth':4,'telnet':5,'private':6,}

deleteIndex = []
for i, row in enumerate(data):
	if row[2] in protocol.keys():
		data[i][1] = transport[row[1]]	
		data[i][2] = protocol[row[2]]
	else:
		deleteIndex.append(i)

data = np.delete(data,deleteIndex,0)
#	remove the 4th column
data = np.delete(data,3,1)
data = data.astype(np.float)
print data.shape
# shuffle all of these properties in the same way
data = shuffle( data )

testSize = int(len( data ) * 0.70)


trainData, testData, trainFlags, testFlags = train_test_split( data[:,:-1],data[:,-1], test_size=testSize)

clfs = [svm.SVC(),nn(solver='lbfgs'),knn(n_neighbors = 200)]
header = ["SVM Predictor", "Neural Net Predictor","KNN Predictor"]

for index, clf in enumerate(clfs):	
	print header[index]
	start = time.time()
	clf.fit( trainData, trainFlags)


	predicted = clf.predict(testData)

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

