import numpy as np

def sigmoid(x, derivative=False):
	return 1 / (1 + np.exp(-x)) if not derivative \
	 							else x * (1 - x)

def tanh(x, derivative=False):
	return np.tanh(x) if not derivative \
					  else 1 - x * x

def relu(x, derivate = False):
    if not derivate:
        output = np.copy(x)
        output[output < 0] = 0.05
        return output

    if derivate:
        output = np.copy(x)
        output[output > 0] = 1
        output[output <= 0] = 0.05
        return output

func_dict = {}
func_dict['sigmoid'] = sigmoid
func_dict['tanh'] = tanh
func_dict['relu'] = relu