##### Performance.py
import numpy as np

def Performance(results, targets):
	accuracy = precision = recall = sensitivity = specificity = F_measure = float(0)
	# print len(results), len(targets)
	if len(results) != len(targets):
		print "Sizes don't match"
	else: 
		size = len(results)
		TN = TP = FP = FN = diff = 0
		for i in range(size):
			diff = results[i]-targets[i]
			if diff < 0:
				FN += 1
			elif diff < 1:
				if results[i]<1:
					TN += 1
				else: 
					TP += 1
			elif diff<2:
				FP +=1

		print "True Positive: \t" + str(TP)
		print "True Negative: \t" + str(TN)
		print "False Positive: \t" + str(FP)
		print "False Negative: \t" + str(FN)
		TP = float(TP)
		TN = float(TN)
		FN = float(FN)
		FP = float(FP)
		accuracy = (TP+TN)/(TP+FP+TN+FN)
		precision = TP/(TP+FP)
		recall = TP/(TP+FN)
		sensitivity = TP/(TP+FN)
		specificity = TN/(TN+FP)
		F_measure = TP/(TP+(FN+FP)/2)
		
		print "accuracy: \t" + str(accuracy)
		print "precision: \t" + str(precision)
		print "recall: \t" + str(recall)
		print "sensitivity: \t" + str(sensitivity)
		print "specificity: \t" + str(specificity)
		print "F_measure: \t" + str(F_measure)

		print "Confusion Matrix:"
		print str(TP) + "\t"+str(FP)+"\n"+str(FN)+"\t"+str(TN)
