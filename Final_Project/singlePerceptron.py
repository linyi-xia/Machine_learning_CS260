#### Simple Single Layer perceptron
#### Name: 		Linyi Xia
#### SID:		504518020
#### Course:	CS 260
#### Last Mod:	3/11/2015

import numpy as np
from numpy import array, dot, random
from random import choice

class sPerceptron(object):
	"""docstring for sPerceptron"""
	def __init__(self, arg, target, th=0.4, eta=0.25, n=1000):
		super(sPerceptron, self).__init__()
		self.arg = arg
		self.arg = self.arg/np.amax(self.arg)
		self.th = th
		self.w = np.random.random_sample(self.arg.shape[1])
		self.target = target
		self.y = np.empty((0,1), int)
		self.exp = 0
		self.n = n
		self.size = len(arg)
		self.eta = eta
		self.results = np.empty((0,2), int)
		print "Input dataset count: \t" + str(self.size)
		print "Initial weight: \t" + str(self.w)
		print "Number of Iterations: \t" + str(n)
		print "Learning rate: \t \t" + str(eta)

	def activation(self):
		error = 0
		for j in range(self.n):
			#self.exp = np.append(self.exp, dot_product,axis=0)
			#print str(np.dot(self.w,self.arg).shape)
			i = np.random.randint(self.arg.shape[0])
			self.exp = np.dot(self.w, self.arg[i])
			error = self.target[i] - self.uniFcn(self.exp)
			# print "error: "
			# print error
			#self.y = np.append(self.y,self.uniFcn(self.exp),axis=0)
			self.w += self.eta*(error)*self.arg[i]		
		print "perceptron results: \t" + str(self.w)

	def forward(self):
		print "____forward test Results____"
		for i in self.arg:
			result = np.dot(self.w,i)
			result = np.asarray(self.uniFcn(result)).reshape(1,1)
			#print result
			self.y = np.append(self.y,result, axis =0)
			#print self.y.shape[0]
			# self.y.reshape(len(self.target),0)
		#print self.target
		self.results = np.column_stack((self.y,self.target))

	def uniFcn(self, expected):
		uniResult = 1
		if expected<self.th:
			uniResult= 0
		return uniResult


		