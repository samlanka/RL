"""Numpy code for K-Bandit Problem
--Sameera Lanka
--www.sameera-lanka.com
--slanka@ncsu.edu"""

#This code implements the k-Armed Bandit situatution for stationary, noneffective and stochastic reward condition,
#using only numpy

from __future__ import division
import numpy as np 

class bandit:
"""Initialises a k-Armed Bandit stationary situation"""
	def __init__(self, k):
		self.k = k;
		#drawing true rewards from a normal probability dist, mean = 0 and var = 1 
		self.reward = np.random.normal(0, 1.0, self.k)  
		print(self.reward)
		

	def train_egreedy(self, timestep, epsilon=0):
		#E-greedy value estimation
		estimate_t = np.zeros(self.k)
		for i in range(timestep):			
	
			if np.random.sample() < self.epsilon:
				a_t = np.random.randint(self.k)
			else:
				a_t = np.argmax(estimate_t)
				
		#stochastic Gaussian reward for action action centred at true reward with var 1	
			r_t = np.random.normal(self.reward[a_t], 1.0)
			estimate_t[a_t] = estimate_t[a_t] + (1/(i+1))*(r_t - estimate_t[a_t])
		return estimate_t

	
	def train_ucb(self, timestep, c=1):
		#Upper confidence bound value estimation
		estimate_t = np.zeros(self.k)
		counter_t = np.zeros(self.k)
		for i in range(timestep):
			a_t = np.argmax(estimate_t + c * np.sqrt(np.log(i)/counter_t))
			counter_t[a_t] = counter_t[a_t] + 1;
			r_t = np.random.normal(self.reward[a_t], 1.0)
			estimate_t[a_t] = estimate_t[a_t] + (1/(i+1))*(r_t - estimate_t[a_t])
		return estimate_t



if __name__=="__main__":
	prob = bandit(5)  
	print prob.train_ucb(2000)
