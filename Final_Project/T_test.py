#### T-tests
import numpy as np
from scipy import stats


class tTest(object):
	"""docstring for tTest"""
	def __init__(self, arg, target):
		super(tTest, self).__init__()
		self.arg = arg
		self.avg = np.empty((0,2), int)
		self.std = np.empty((0,2), int)
		self.se  = np.empty((0,2), int)
		self.avg2SD = np.empty((0,2), int)
		self.avg2SE = np.empty((0,2), int)
		self.avg2SD_hi = np.empty((0,2), int)
		self.avg2SD_lo = np.empty((0,2), int)
		self.avg2SE_hi = np.empty((0,2), int)
		self.avg2SE_lo = np.empty((0,2), int)
		self.diff =np.empty((0,1), int)
		self.lbl = target
	def test(self):
		diff = np.empty((0,2), int)
		diff = np.diff(self.arg,axis = 1)
		print "Raw Data diff: "
		print self.diff
		self.diff = np.mean(diff,axis = 0).reshape(1,1)
		print "Average Raw Data diff: "
		print self.diff
		self.avg = np.mean(self.arg, axis=0)
		self.std = np.std (self.arg, axis=0)
		self.ser = stats.sem(self.arg, axis=0)
		self.avg2SD = np.transpose(np.append((self.avg+2*self.std).reshape(1,2),(self.avg-2*self.std).reshape(1,2),axis=0))
		self.avg2SE = np.transpose(np.append((self.avg+2*self.ser).reshape(1,2),(self.avg-2*self.ser).reshape(1,2),axis=0))
	def showTtest(self):
		print "\n*************T-Test Results:*************"
		if self.lbl>0:
			print ">>>>>Positive Class<<<<<<"
		else:
			print ">>>>>Negative Class<<<<<<"
		print "Mean: "
		print self.avg
		print "\nStandard Deviation: "
		print self.std
		print "\nMean + - 2SD range"
		print self.avg2SD
		print "\nMean + - 2SE range"
		print self.avg2SE
		print "***************T-Test End***************\n"

	def tTest_classify(self, Range_pos,Range_neg, Patients):
		label = np.empty((0,1), int)
		print "Positive and Negative Class conditions: "
		print Range_pos
		print Range_neg
		for i in Patients.avg:
			if ((i[0] <Range_pos[0,0]) and (i[1] < Range_pos[1,0])):
				label = np.append(label,1)
				# if i[1] < Range_pos[1,0]:
				# 	label = np.append(label,1)
			elif ((i[0]>Range_neg[0,1]) and (i[1] > Range_neg[1,1])):
				label = np.append(label,0) 
			else:
				label = np.append(label,2)
		Patients.lbl = label
		return Patients

