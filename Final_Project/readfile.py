from __main__ import *
import csv
import numpy as np
def importCSV(name):
	#print name
	with open(name,'rb') as csvfile:
		csvReader = csv.reader(csvfile, delimiter=',', quotechar='|')
		"{0}".format(csvfile.readline().split())
		Diastolic = []
		Systolic = []
		for row in csvReader:
			Diastolic.append(int(row[1])) 
			Systolic.append(int(row[2]))
	dataSet = zip(Diastolic,Systolic)
	return dataSet
		