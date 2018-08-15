import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from activation_functions import *

clrs = ['red', 'green', 'blue', 'pink', 'orange']

class hiddenUnitsPool(object):
	
	def __init__(self, input_size, output_size, num_candidates):
		self.I = input_size
		self.O = output_size
		self.K = num_candidates

		weights_mean = 0
		weights_std_dev = np.sqrt(2.0 / float(self.I))
		weights_shape = (self.K, self.I)

		weights_val = 4 * np.sqrt(6.0 / (self.I + 1))

		self.weights = np.random.uniform(-weights_val, weights_val, (weights_shape))

		self.alpha = 0.4 # TODO
		self.eps = 0.001  # TODO
		self.f = sigmoid      # TODO
		self.max_iterations = 30 # TODO

	def forward(self, xs, index=None):		# index = the index of the neuron we want to forward
		if index is None:
			hs = np.dot(xs, self.weights.T)
			vs = self.f(hs)
		else:
			hs = np.dot(xs, self.weights[index, :].T)
			vs = self.f(hs)

		return hs, vs

	def get_correlation(self, xs, vs, es):
		
		v_term = vs - np.mean(vs, axis=0)
		e_term = es - np.mean(es, axis=0)
		corr = np.dot(v_term.T, e_term)
		ss = np.sum(np.abs(corr), axis=1)

		return ss, corr, e_term

	def backward(self, xs, vs, corr, e_term):

		dweights = np.zeros(self.weights.shape)
		
		for k in range(self.K):
			tmp1 = np.multiply(self.f(vs[:, k], True)[:, np.newaxis], xs).T
			tmp2 = np.dot(tmp1, e_term)
			tmp3 = np.dot(np.sign(corr[k, :]), tmp2.T)
			dweights[k, :] = tmp3
		
		return dweights

	def update_weights(self, dweights):
		self.weights += self.alpha * dweights
		self.alpha = 0.998 * self.alpha

	def plot(self, ss):

		if not hasattr(self, 'ss'):
			self.ss = np.array([ss])
			self.figure = plt.figure()

			plt.xlabel('Iteration')
			plt.ylabel('Correlation')

		else:
			new_ss = np.zeros((self.ss.shape[0] + 1, self.ss.shape[1]))
			new_ss[:-1, :] = self.ss
			new_ss[-1, :] = ss

			self.ss = new_ss

		plt.figure(self.figure.number)
		for i in range(self.K):
			plt.plot(self.ss[:, i], color=clrs[i % len(clrs)])
		plt.savefig('recruiter.png')

	def check_convergence(self, iteration):
		if self.ss.shape[0] < 3:
			self.converged = False

		elif iteration == self.max_iterations:
			self.converged = True

		else:
			diff = np.abs(self.ss[-1, :] - self.ss[-2, :])
			
			if np.mean(diff) < self.eps:
				self.converged = True
			else:
				self.converged = False

	def train(self, xs, es):

		self.converged = False
		
		iteration = 0
		while not self.converged:

			hs, vs = self.forward(xs)
			ss, corr, e_term = self.get_correlation(xs, vs, es)
			dweights = self.backward(xs, vs, corr, e_term)

			self.update_weights(dweights)
			self.plot(ss)
			self.check_convergence(iteration)

			iteration += 1


		ss, _, _ = self.get_correlation(xs, vs, es)
		self.best_candidate_idx = np.argmax(ss)

		print('Hidden unit candidates convergence after %d iterations. \
			Best candidate is %d' % (iteration, self.best_candidate_idx))

	def get_best_candidate_values(self, xs):
		hs, vs = self.forward(xs, index=self.best_candidate_idx)
		return vs