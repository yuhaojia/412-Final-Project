'''
Logistic Regression classifier class
'''
from __future__ import division
import numpy as np
import scipy
import math
from scipy.optimize import fmin_bfgs

class LogisticRegression:
	def __init__(self, data, labels, alpha = 1, num_iters = 100):

		self.num_iters = num_iters
		self.alpha = alpha
		assert(len(np.unique(labels))>=2)
		pass


	def train(self, data, Olabels, unique_classes):

		print('training the data')
		num_iters = self.num_iters
		m,n = data.shape

		labels = np.zeros(Olabels.shape)
		
		uniq_Olabel_names = np.unique(Olabels)

		uniq_label_list = range(len(uniq_Olabel_names))

		for each in zip(uniq_Olabel_names, uniq_label_list):
			o_label_name = each[0]
			new_label_name = each[1]
			labels[np.where(Olabels == o_label_name)] = new_label_name

		labels = labels.reshape((len(labels),1))
		num_classes = len(unique_classes)
		Init_Thetas = []
		Thetas = [] 
		Cost_Thetas = [] 
		Cost_History_Theta = [] 
		
		
		for eachInitTheta in range(num_classes):
				theta_init = np.zeros((n,1))
				Init_Thetas.append(theta_init)
				pass
		for eachClass in range(num_classes):

 			local_labels = np.zeros(labels.shape)

 			
 			local_labels[np.where(labels == eachClass)] = 1

 			assert(len(np.unique(local_labels)) == 2)
 			assert(len(local_labels) == len(labels))

 			init_theta = Init_Thetas[eachClass]

 			new_theta, final_cost = self.computeGradient(data, local_labels, init_theta)

 			Thetas.append(new_theta)
 			Cost_Thetas.append(final_cost)
			
		return Thetas, Cost_Thetas
	

	def classify(self, data, Thetas):
		assert(len(Thetas)>0)
		if(len(Thetas) > 1):
			mvals = []	
			for eachTheta in Thetas:
				mvals.append(self.sigmoidCalc(np.dot(data, eachTheta)))
				pass
			return mvals.index(max(mvals))+1
		elif(len(Thetas) == 1):
			cval = round(self.sigmoidCalc(np.dot(data, Thetas[0])))+1.0
			return cval
	
	def sigmoidCalc(self, data):
		data = np.array(data, dtype = np.longdouble)
		g = 1/(1+np.exp(-data))
		return g

	def computeCost(self,data, labels, init_theta):

		llambda = 1.0

		m,n = data.shape
		
		J = 0

		grad = np.zeros(init_theta.shape)

		theta2 = init_theta[range(1,init_theta.shape[0]),:]
		regularized_parameter = np.dot(llambda/(2*m), np.sum( theta2 * theta2))

		
		
		J = (-1.0/ m) * ( np.sum( np.log(self.sigmoidCalc( np.dot(data, init_theta))) * labels + ( np.log ( 1 - self.sigmoidCalc(np.dot(data, init_theta)) ) * ( 1 - labels ) )))
		
		J = J + regularized_parameter
		return J

	def computeGradient(self,data, labels, init_theta):
		alpha = self.alpha
		num_iters = self.num_iters
		m,n = data.shape

		llambda = 1
		
		for eachIteration in range(num_iters):
			cost = self.computeCost(data, labels, init_theta)
			print('iteration: ', eachIteration)
			print('cost: ', cost)
			B = self.sigmoidCalc(np.dot(data, init_theta) - labels)
			A = (1/m)*np.transpose(data)
			grad = np.dot(A,B)
			A = (self.sigmoidCalc(np.dot(data, init_theta)) - labels )
			B =  data[:,0].reshape((data.shape[0],1))
			grad[0] = (1/m) * np.sum(A*B)
			
			A = (self.sigmoidCalc(np.dot(data, init_theta)) - labels)
			B = (data[:,range(1,n)])
			
			for i in range(1, len(grad)):
				A = (self.sigmoidCalc(np.dot(data,init_theta)) - labels )
				B = (data[:,i].reshape((data[:,i].shape[0],1)))
				grad[i] = (1/m)*np.sum(A*B) + ((llambda/m)*init_theta[i])
		
			init_theta = init_theta - (np.dot((alpha/m), grad))
			
		return init_theta, cost
