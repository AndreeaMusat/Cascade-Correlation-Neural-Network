import sys, os
import argparse
import random
import numpy as np
import pickle
sys.path.append(
	os.path.join(
		os.path.dirname(__file__), 
		'..', 
		'utils'
	)
)
import mnist_reader
from cascade_correlation_network import CasCorNet
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from activation_functions import *


np.set_printoptions(threshold=np.nan)

def add_bias(X):
	X_bias_incl = np.ones((X.shape[0], X.shape[1] + 1))
	X_bias_incl[:, :-1] = X

	return X_bias_incl

def load_and_preprocess_data():

	X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
	X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')

	# X_train = 1.0 * X_train / 255
	# X_test = 1.0 * X_test / 255

	# subtract the mean image
	mean_img = np.mean(X_train, axis=0)
	stddev_img = np.std(X_train, axis=0)

	# normalize the data
	X_train = (X_train - mean_img) / stddev_img
	X_test = (X_test - mean_img) / stddev_img

	return add_bias(X_train), y_train, add_bias(X_test), y_test

def main(args):

	X_train, y_train, X_test, y_test = load_and_preprocess_data()

	if args.time == 'train':
		
		input_size = len(X_train[0])
		output_size = len(np.unique(y_train))

		net = CasCorNet(input_size, output_size, args)
		net.set_data(X_train, y_train, X_test, y_test)
		net.train()

	elif args.time == 'test':
		
		try:
			net = pickle.load(open(args.resume, 'rb'))
		except:
			exit('Cascade correlation net could not be loaded from file, bye')

		net.train()

def exit(msg):
	print(msg)
	sys.exit(-1)


def sanity_checks(args):
	try:
		assert args.learning_rate >= 0
		assert args.learning_rate <= 1
	except: 
		exit('Learning rate should be between 0 and 1')

	try:
		assert args.patience < 0.1
	except:
		exit('Patience should be a small float')

	try:
		assert args.activation_func in func_dict.keys()
		args.activation_func = func_dict[args.activation_func]
	except:
		exit('Activation function should be in:' + str(func_dict.keys()))

	try:
		assert args.output_file is not None
	except:
		exit('Output file unknown.')

	try:
		assert args.time in ['train', 'test']
	except:
		exit('Time should be train|test')

	try:
		if args.time == 'test':
			assert args.resume is not None
	except:
		exit('Resume should be the name of the file in which the saved model is found if time=test')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001)
	parser.add_argument('--minibatch-size', type=int, dest='minibatch_size', default=1)
	parser.add_argument('--patience', type=float, dest='patience', default=0.0001)
	parser.add_argument('--activation', type=str, dest='activation_func', default='sigmoid')
	parser.add_argument('--time', type=str, dest='time', default='train')
	parser.add_argument('--candidates', type=int, dest='num_candidates', default=5)
	parser.add_argument('--output-file', type=str, dest='output_file')
	parser.add_argument('--resume', type=str, dest='resume', default=None)

	args, unknown = parser.parse_known_args()
	sanity_checks(args)
	print(args)

	main(args)
