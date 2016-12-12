#### main.py
#### Name: 		Linyi Xia
#### SID:		504518020
#### Course:	CS 260
#### Last Mod:	3/11/2015
from readfile import importCSV
from sklearn import cross_validation,datasets,svm
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from scipy import stats
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from T_test import tTest
from random import choice
import singlePerceptron 
import numpy as np
import pylab as pl
import os.path
import struct
import mlp
import knn
import Performance

largest_filename = 54
fileCnt = 0
rawData_pos = np.empty((0,2), int)
rawData_neg = np.empty((0,2), int)
tTest_results = tTest_results_pos = tTest_results_neg = np.empty((0,2), int)
mlp_results = testResults = testTargets = np.empty((0,1), int)
third_result = np.empty((0,3), int)

class features:
	def __init__(self,target):
		self.target = 2
		self.min = self.max = self.var = self.med = self.std = self.avg = np.empty((0,2), int)
		self.dist = self.lbl = self.targetArray = self.diff = np.empty((0,1), int)
	def showSelf(self):
		if self.target > 0 :
			print "\n >>--------------------------------<<"
			print "\t Positive Class: \n"
		else:
			print "\n >>--------------------------------<<"
			print "\t Negative Class: \n"
		print "Average: \nDiastolic , Systolic \n" + str(self.avg)
		print "Standard Deviation: \nDiastolic , Systolic \n" + str(self.std)
		print "Median: \nDiastolic , Systolic \n" + str(self.med)
		print "Variance: \nDiastolic , Systolic \n" + str(self.var)
		print "Maximum: \n" + str(self.max)
		print "Diff: \n" + str(self.diff)
	def extraction(self, dataArray):
		self.avg = np.append(self.avg, np.mean(dataArray,axis=0).reshape(1,2),axis=0)
		self.std = np.append(self.std, np.std(dataArray,axis=0).reshape(1,2),axis=0)
		self.med = np.append(self.med, np.median(dataArray,axis=0).reshape(1,2),axis=0)
		self.var = np.append(self.var, np.var(dataArray,axis=0).reshape(1,2),axis=0)
		self.max = np.append(self.max, np.amax(dataArray,axis=0).reshape(1,2),axis=0)
		self.min = np.append(self.min, np.amin(dataArray,axis=0).reshape(1,2),axis=0)
		self.diff = np.append(self.diff,np.mean(np.diff(dataArray)).reshape(1,1),axis=0)
		self.targetArray = np.append(self.targetArray, np.asarray(self.target).reshape(1,1), axis=0)

features_pos = features(1)
features_pos.target = 1
features_neg = features(0)
features_neg.target = 0

#Go thru the folder of data
for x in range (1,largest_filename):
	name_1 = str(x)+'_1.csv'
	name_0 = str(x)+'_0.csv'
	if os.path.isfile(name_1):
		dataSet_pos = np.asarray(importCSV(name_1))
		rawData_pos = np.append(rawData_pos ,dataSet_pos,axis=0)
		features_pos.extraction(dataSet_pos)
		fileCnt +=1
	elif os.path.isfile(name_0):
		dataSet_neg = np.asarray(importCSV(name_0))
		rawData_neg = np.append(rawData_neg ,dataSet_neg,axis=0)
		features_neg.extraction(dataSet_neg)
		fileCnt +=1

# features_pos.showSelf()
# features_neg.showSelf()
# print "_____Average and Targets_______"
# print np.column_stack((features_pos.avg[0,1],features_pos.targetArray[0]))
# print type(features_pos.avg[0,1])

#T-test		--------------------------------	
print "_________________T-test Results_________________"		
pos_test = tTest(rawData_pos,1)
pos_test.test()
neg_test = tTest(rawData_neg,0)
neg_test.test()
pos_test.showTtest()
neg_test.showTtest()
pos_test.tTest_classify(pos_test.avg2SE, neg_test.avg2SE, features_pos)
# print "++POS Class T_test results: \t Results: \t Targets:"
tTest_results_pos = np.column_stack((features_pos.avg,features_pos.var , features_pos.lbl,features_pos.targetArray))
# print tTest_results_pos
neg_test.tTest_classify(pos_test.avg2SE ,neg_test.avg2SE, features_neg)
# print "++NEG Class T_test results: \t Results: \t Targets:"
tTest_results_neg = np.column_stack((features_neg.avg,features_neg.var,features_neg.lbl,features_neg.targetArray))
# print tTest_results_neg
#End T-test	--------------------------------
print "++Class T_test results: \t Results: \t Targets:"
tTest_results = np.append(tTest_results_neg,tTest_results_pos,axis = 0).reshape(fileCnt,6)
print tTest_results.shape
testResults = tTest_results[:,4]
testTargets = tTest_results[:,5]
print testResults
print testTargets
Performance.Performance(testResults, testTargets)



dataset = np.empty((0,2), int)
targetSet = np.empty((0,1), int)
dataSet = np.append(features_pos.std,features_neg.std,axis=0)
dataSet1 = np.append(features_pos.diff,features_neg.diff,axis=0)
dataSet2 = np.append(features_pos.var,features_neg.var,axis=0)
#dataSet = np.column_stack((np.append(features_pos.var,features_neg.var,axis=0),np.append(features_pos.avg,features_neg.avg,axis=0)))
testData = np.column_stack((dataSet,dataSet1,dataSet2))
targetSet = np.append(features_pos.targetArray,features_neg.targetArray, axis=0)
print "______________DATASET________Targets_______________\n"
print np.column_stack((testData,targetSet))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataSet, targetSet, test_size=0.15, random_state=0)

q = mlp.mlp(dataSet, targetSet,2)
q.mlptrain(dataSet, targetSet,0.25,1001)
mlp_results = q.confmat(dataSet, targetSet)
# print mlp_results

# print y_test
y_train = np.transpose(y_train)[0,:]
y_test = np.transpose(y_test)[0,:]
#print y_test

print "_________________MLP Results With T-test________"
third_result =np.column_stack((mlp_results,tTest_results[:,4:6]))
print "before: \n"
print third_result
for i in range(len(third_result)):
	if third_result[i,1]> 1: 
		third_result[i,1] = third_result[i,0]
print "after: \n"
print third_result
print "_________________MLP Results With T-test Performance________"
testResults = third_result[:,1]
testTargets = third_result[:,2]
print testResults
print testTargets
Performance.Performance(testResults, testTargets)
print "_________________END OF MLP Results With T-test Performance________"

# clf = svm.SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
# print clf.score(X_test, y_test) 

# # Run classifier
# #classifier = svm.SVC(kernel='linear', probability=True)
# probas_ = clf.predict_proba(X_test)
# print "probability"
# print probas_
# # Compute ROC curve and area the curve
# fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
# roc_auc = auc(fpr, tpr)
# print "Area under the ROC curve : %f" % roc_auc

# Plot ROC curve
# pl.clf()
# pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# pl.plot([0, 1], [0, 1], 'k--')
# pl.xlim([0.0, 1.0])
# pl.ylim([0.0, 1.0])
# pl.xlabel('False Positive Rate')
# pl.ylabel('True Positive Rate')
# pl.title('Receiver operating characteristic example')
# pl.legend(loc="lower right")
# pl.show()

# T-test results
# print np.column_stack((features_pos.avg,features_pos.targetArray))
# print features_pos.targetArray.shape

# print np.column_stack((features_neg.avg,features_neg.targetArray))
# print features_neg.targetArray.shape

# clf = SVC()
# print clf.fit(X_train,np.asarray(np.transpose(y_train)))
# print clf.predict(X_test)
# print y_test
# print clf.score(X_test,y_test)

####--------Single layer perceptron
print "_________________Single Layer Perceptron Results_________________"
SLP = singlePerceptron.sPerceptron(dataSet, targetSet)
SLP.activation()
SLP.forward()
#print SLP.results
testResults = SLP.results[:,0]
testTargets = SLP.results[:,1]
print testResults
print testTargets
print "_________________SLP Performance Results_________________________"
Performance.Performance(testResults, testTargets)
#mlp.confmat()

print "_________________K-Nearest Neighbors Results_____________________"
testResults = testTargets = np.empty((0,1), int)
knn_data = np.column_stack((dataSet,targetSet))
N = knn_data.shape[0]
print "Number of datafiles: \t" + str(N)
z = np.zeros((N-1,N))
for i in range(knn_data.shape[0]):
#i = np.random.randint(knn_data.shape[0])
	#print "\n Current iteration: \t" +str(i)
	dataSet_knn = np.append(dataSet[0:i,:],dataSet[i+1:(knn_data.shape[0]-1),:], axis =0)
	targetSet_knn = np.append(targetSet[0:i,:],targetSet[i+1:(knn_data.shape[0]-1),:], axis =0)
	knn_data = np.column_stack((dataSet_knn,targetSet_knn))
	#print "\ntest Point: \n" + str(dataSet[i,:])
	#print "\n Data Pool: \n" + str(dataSet_knn)
	nNeighbour = knn.findNeighbours(knn_data,dataSet[i,:],2)
	#print str(nNeighbour[0:3]) + "\t" + str(dataSet[i])+str(targetSet[i])
	testResults = np.append(testResults,int(nNeighbour[2]))
	testTargets = np.append(testTargets, int(targetSet[i]))
# print testResults
# print testTargets
print "_________________KNN Performance Results_____________________"
Performance.Performance(testResults, testTargets)


