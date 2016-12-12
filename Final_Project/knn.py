#### K Nearest Neighbours
#### Name: 		Linyi Xia
#### SID:		504518020
#### Course:	CS 260
#### Last Mod:	3/11/2015

import numpy as np
import math

def calcDist(data1, data2):
	length = data1.shape[0]-1
	dist = 0
	for x in range(length):
		dist += pow((data1[x] - data2[x]), 2)
	return math.sqrt(dist)

def findNeighbours(data_target, tstInput,k):
	length = len(tstInput)-1
	index = 0
	neighbours = dist = np.empty((0,1), int)
	col = 0
	for i in range(len(data_target)):
		d = np.asarray(calcDist(data_target[i],tstInput)).reshape(1,1)
		# if d<0.0001:
		# 	print data_target[i]
		# 	print "here" + str(i)
		dist = np.append(dist,d,axis=0)
		dist = np.sort(dist,axis=0)
	#print "\nlist of dist: \n" + str(dist)
	for j in range(k):
		neighbours = np.append(neighbours,dist[j])

	index = findLabel(data_target, tstInput, dist[0])
	return data_target[index]

def findLabel(data_target, tstInput, dist):
	index = len(data_target)+1
	d = 0
	for i in range(len(data_target)):
		d = calcDist(data_target[i],tstInput)
		if (d-dist) < 0.0000001:
			return i
	return 0
