#!/home/slanka/anaconda2/bin/python

#greedy vs non-greedy
#sample average vs learning rate
#initial values / optimal initial values

from __future__ import division
import numpy as np 

class bandit:

	def __init__(self, k):
		print("hello world")
		self.k = k;
		self.reward = np.random.normal(0, 1.0, self.k)
		print(self.reward)

	def train_egreedy(self, timestep, epsilon=0):
		estimate_t = np.zeros(self.k)

		for i in range(timestep):			
	
			if np.random.sample() < self.epsilon:
				a_t = np.random.randint(self.k)
			else:
				a_t = np.argmax(estimate_t)
			
			r_t = np.random.normal(self.reward[a_t], 1.0)
			estimate_t[a_t] = estimate_t[a_t] + (1/(i+1))*(r_t - estimate_t[a_t])
			
		return estimate_t

	def train_ucb(self, timestep, c=5):
		estimate_t = np.zeros(self.k)
		counter_t = np.zeros(self.k)

		for i in range(timestep):
			a_t = np.argmax(estimate_t + c * np.sqrt(np.log(i)/counter_t))
			counter_t[a_t] = counter_t[a_t] + 1;
			r_t = np.random.normal(self.reward[a_t], 1.0)
			estimate_t[a_t] = estimate_t[a_t] + (1/(i+1))*(r_t - estimate_t[a_t])
			#print counter_t
		return estimate_t



if __name__=="__main__":
	prob = bandit(5)
	print prob.train_ucb(2000)
