import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from activation_functions import *
from hidden_units_pool import hiddenUnitsPool

colors = ['red', 'green', 'pink', 'blue', 'orange', 'magenta', 'olive']

class CasCorNet(object):
	def __init__(self, input_size, output_size, args):

		# I = input size, O = output size
		self.I = input_size
		self.O = output_size

		# initialize the weights
		self.weights = self.init_weights()

		self.hidden_units = []

		# hyperparameters
		self.f = args.activation_func
		self.alpha = args.learning_rate
		self.mb_sz = args.minibatch_size
		self.eps = args.patience
		self.output_file = args.output_file

		# TODO
		self.max_iterations_io = 50
		self.max_iterations = 30
		self.eval_every = 10

		self.train_loss = np.array([])
		self.test_loss = np.array([])
		self.loss_figure = plt.figure()
		self.accuracy_figure = plt.figure()
		self.cm_figure = plt.figure()
		self.limit_points_xs = []
		self.limit_points_ys = []

	def set_data(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

	def init_weights(self):
		val = 4 * np.sqrt(6 / (self.O + self.I))
		weights = np.random.uniform(-val, val, (self.O, self.I))
		return weights

	def accuracy(self, xs, ts):
		_, ys = self.forward(xs)
		acc = 1.0 * np.sum(np.argmax(ts, axis=1) == np.argmax(ys, axis=1)) / len(xs)
		return acc

	def forward(self, xs):
		hs = np.dot(xs, self.weights[:,:len(xs[0])].T)
		ys = self.f(hs)

		return hs, ys

	def backward(self, xs, ts, ys):

		# initialize the gradient with zeros
		dweights = np.zeros(self.weights.shape)
		
		# add the gradients for all examples
		for i in range(len(ts)):
			delta = -(ts[i] - ys[i]) * self.f(ys[i], True)
			dweights += np.outer(delta, xs[i])

		# take the average of the gradients for all examples
		dweights /= self.mb_sz

		return dweights

	def get_loss(self, ts, ys, return_sum):
		if return_sum == True:
			return np.sum(0.5 * (ts - ys)**2) / len(ts)
		else:
			return 0.5 * (ts - ys)**2 / len(ts)

	def update_weights(self, dweights, loss):
		self.weights -= self.alpha * dweights

	def plot_loss(self, loss_val, loss_name):

		if loss_name == 'train':
			loss_arr = self.train_loss
		else:
			loss_arr = self.test_loss

		new_loss = np.zeros((loss_arr.shape[0] + 1))
		new_loss[:-1] = loss_arr
		new_loss[-1] = loss_val

		if loss_name == 'train':
			self.train_loss = new_loss
		else:
			self.test_loss = new_loss
		
		plt.figure(self.loss_figure.number)	
		plt.xlabel('Iteration')
		plt.ylabel('Loss')		
		h1, = plt.plot(self.train_loss, color='red', label='Training loss')
		h2, = plt.plot(np.arange(0, self.eval_every * len(self.test_loss), self.eval_every), self.test_loss, color='blue', label='Test loss')
		plt.scatter(self.limit_points_xs, self.limit_points_ys, color='black', label='New hidden unit recruited')
		plt.legend(handles=[h1, h2])
		plt.savefig('training_loss.png')
		plt.clf()

	def plot_accuracy(self, train_accuracy, test_accuracy):

		if not hasattr(self, 'train_accuracies'):
			self.train_accuracies = [train_accuracy]
			self.test_accuracies = [test_accuracy]
		else:
			self.train_accuracies.append(train_accuracy)
			self.test_accuracies.append(test_accuracy)

		plt.figure(self.accuracy_figure.number)
		h1, = plt.plot(self.train_accuracies, color='red', label='Train accuracy')
		h2, = plt.plot(self.test_accuracies, color='blue', label='Test accuracy')
		plt.legend(handles=[h1, h2])
		plt.savefig('accuracy.png')
		plt.clf()

	def check_io_convergence(self, iteration):
		if iteration == self.max_iterations_io:
			self.converged = True
		elif len(self.train_loss) >= 2 and \
			 abs(self.train_loss[-1] - self.train_loss[-2]) < self.eps:
			self.converged = True

	def eval_network(self, xs_test, ts_test, ts_idx_test):

		print('learning rate', self.alpha)

		for hidden_unit in self.hidden_units:
			vs = hidden_unit.get_best_candidate_values(xs_test)
			print('xs shape before', xs_test.shape)
			xs_test = self.augment_input(xs_test, vs)
			print('xs shape after', xs_test.shape)

		self.hidden_units = []

		hs, ys = self.forward(xs_test)
		test_loss = self.get_loss(ts_test, ys, True)

		# make predictions
		predictions = np.argmax(ys, axis=1)

		accuracy = 0.0
		confusion_matrix = np.zeros((self.O, self.O))
		for i in range(len(xs_test)):
			confusion_matrix[ts_idx_test[i], predictions[i]] += 1.0
			if ts_idx_test[i] == predictions[i]:
				accuracy += 1

		accuracy /= len(xs_test)
		print('Accuracy on test data: %.3f' % (accuracy))

		plt.figure(self.cm_figure.number)
		plt.imshow(confusion_matrix)
		plt.savefig('confusion_matrix.png')
		plt.clf()

		self.plot_loss(test_loss, 'test')

		return xs_test

	def train_io(self, xs, ts, xs_test, ts_test, ts_idx_test):

		iteration = 0
		
		while not self.converged:

			shuffled_range = range(len(xs))
			np.random.shuffle(shuffled_range)

			total_loss = 0.0

			# get minibatches of data
			for i in range(len(xs) // self.mb_sz):

				indices = shuffled_range[i * self.mb_sz:(i+1) * self.mb_sz]
				mini_xs = xs[indices]
				mini_ts = ts[indices]

				# forward pass
				mini_hs, mini_ys = self.forward(mini_xs)

				# compute total loss on this minibatch
				loss = self.get_loss(mini_ts, mini_ys, True)
				total_loss += loss

				# backward pass
				dweights = self.backward(mini_xs, mini_ts, mini_ys)

				# update the weights using delta rule
				self.update_weights(dweights, loss)
				
			train_accuracy = self.accuracy(xs, ts)
			test_accuracy = self.accuracy(xs_test, ts_test)

			print('TRAIN ACC=', train_accuracy)
			print('TEST ACC=', test_accuracy)
			
			self.plot_loss(total_loss / (len(xs) / self.mb_sz), 'train')
			self.plot_accuracy(train_accuracy, test_accuracy)
			self.check_io_convergence(iteration)


			if (len(self.train_loss) - 1) % self.eval_every == 0:
				xs_test = self.eval_network(xs_test, ts_test, ts_idx_test)

			iteration += 1

		self.limit_points_xs.append(len(self.train_loss))
		self.limit_points_ys.append(self.train_loss[-1])

		print('Input-output convergence after %d iterations' % iteration)

		return xs_test

	def train(self):

		xs = self.X_train
		ts_idx = self.y_train
		xs_test = self.X_test
		ts_idx_test = self.y_test

		N, M = len(xs), len(xs_test)

		# create target arrays with 1 for the correct class
		ts = np.zeros((N, self.O))
		ts[np.arange(N), ts_idx] = 1
		ts_test = np.zeros((M, self.O))
		ts_test[np.arange(M), ts_idx_test] = 1

		iteration = 0
		acceptable_loss = 0.01
		max_iterations = 60

		train_acc = []
		test_acc = []

		while True:
			# train the input-output connections until convergence
			self.converged = False

			xs_test = self.train_io(xs, ts, xs_test, ts_test, ts_idx_test)
			self.X_test = xs_test

			# measure loss on data and stop if it's acceptable
			_, ys = self.forward(xs)
			losses = self.get_loss(ts, ys, False)

			loss = np.sum(losses)
			if loss < acceptable_loss:
				break

			if iteration == max_iterations:
				break

			# otherwise add a new hidden unit
			xs = self.add_hidden_unit(xs, ts, losses)
			self.X_train = xs

			iteration += 1

			if iteration % 5 == 0:
				with open(self.output_file, 'wb') as f:
					pickle.dump(self, f)

	def augment_input(self, xs, vs):
		new_xs = np.zeros((xs.shape[0], xs.shape[1] + 1))
		new_xs[:, :-1] = xs
		new_xs[:, -1] = vs

		return new_xs

	def add_hidden_unit(self, xs, ts, losses):
		
		# initialize a pool of 10 candidates # TODO: make param instead of 10
		candidates_pool = hiddenUnitsPool(self.I, self.O, 5)
		candidates_pool.train(xs, losses)		
		vs = candidates_pool.get_best_candidate_values(xs)
		xs = self.augment_input(xs, vs)

		self.hidden_units.append(candidates_pool)
		self.I += 1 	# just added one more element for each input, so the size fo the input has increased

		new_weights = self.init_weights()
		new_weights[:, :-1] = self.weights
		self.weights = new_weights

		return xs