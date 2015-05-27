import math
import random
import pickle

ALPHA = 1

def sigmoid(x):
	return 1/(1 + math.pow(math.e, -x))

def linear(x):
	return x

"""NEURONS"""
class Neuron(object):
	def __init__(self):
		self.sum = 0

	def add(self, value):
		pass

	def clear(self):
		self.sum = 0

	def output(self):
		raise NotImplementedError("Please Implement this method")



class InputNeuron(Neuron):
	def set(self, _input):
		self.input = _input

	def output(self):
		return self.input

class HiddenNeuron(Neuron):
	def __init__(self, fn = sigmoid):
		self.fn = fn 
		
	def output(self):
		return self.fn(self.sum)

	def add(self, value):
		self.sum += value

class OutputNeuron(HiddenNeuron):
	def __init__(self, fn = sigmoid):
		self.fn = fn 	

	def result(self):
		return self.output()

class BiasNeuron(InputNeuron):
	def __init__(self):
		self.set(1)

class Weights(object):
	def __init__(self, parent, child):
		self.parent = parent
		self.child = child
		self.weights = {}
		for n1 in parent.neurons:
			for n2 in child.neurons:
				self.weights[(n1, n2)] = 0

	def randomize(self):
		for key, val in self.weights.items():
			self.weights[key] = random.randint(-2, 2)

	def get(self, key):
		return self.weights[key]

	def set(self, key, val):
		self.weights[key] = val

	def set_matrix(self, matrix):
		for i in range(0, len(self.parent.neurons)):
			n1 = self.parent.neurons[i]
			for j in range(0, len(self.child.neurons)):
				n2 = self.child.neurons[j]
				if not isinstance(n2, BiasNeuron): 
					self.weights[(n1, n2)] = matrix[i][j]

	def items(self):
		return self.weights.items()

"""LAYERS"""
class Layer(object):
	def __init__(self, size):
		self.child = None
		self.parent = None

	def connect(self, layer):
		self._set_child(layer)
		layer._set_parent(self)
		self.weights = Weights(self, layer)

	def clean(self):
		for n in self.neurons:
			n.clear()
		self.errors = {}

	def forward_prop(self):
		self.child.clean()
		for n1 in self.neurons:
			for n2 in self.child.neurons:
				if not isinstance(n2, BiasNeuron): 
					n2.add(n1.output() * self.weights.get((n1, n2)))

	def back_prop(self, output, expected):
		pass

	def _set_parent(self, layer):
		self.parent = layer

	def _set_child(self, layer):
		self.child = layer

	def weight_update(self):
		weights = self.parent.weights
		for n1 in self.parent.neurons:
			for n2 in self.neurons:
				weights.set((n1, n2), weights.get((n1, n2)) + ALPHA * self.errors[n2] * n1.output())

class HiddenLayer(Layer):
	def __init__(self, size):
		self.child = None
		self.parent = None
		self.neurons = [ HiddenNeuron() for i in range(0, size)]
		self.neurons.append(BiasNeuron())
		self.errors = {}

	def forward_prop(self):
		super().forward_prop()
		self.child.forward_prop()

	def back_prop(self):
		for n1 in self.neurons:
			_sum = sum([ self.child.errors[n2] * self.weights.get((n1, n2)) for n2 in self.child.neurons])
			self.errors[n1] = n1.output() * (1 - n1.output()) * _sum
				

		self.parent.back_prop()

	def weight_update(self):
		super().weight_update()
		self.parent.weight_update()


class InputLayer(Layer):
	def __init__(self, size):
		self.child = None
		self.parent = None
		self.neurons = [ InputNeuron() for i in range(0, size)]
		self.neurons.append(BiasNeuron())

	def forward_prop(self, inputs):
		#Set inputs
		for i in range(0, len(inputs)):
			self.neurons[i].set(inputs[i])

		super().forward_prop()
		self.child.forward_prop()

	def _set_parent(self, layer):
		raise NotImplementedError("You cannot use this method")

	def back_prop(self):
		pass

	def weight_update(self):
		pass

class OutputLayer(Layer):
	def __init__(self, size):
		self.child = None
		self.parent = None
		self.neurons = [ OutputNeuron() for i in range(0, size)]
		self.errors = {}

	def _set_child(self, layer):
		raise NotImplementedError("You cannot use this method")

	def forward_prop(self):
		pass

	def result(self):
		return [ neuron.result() for neuron in self.neurons ]

	def back_prop(self, output, expected):
		for i in range(0, len(self.neurons)):
			neuron = self.neurons[i]
			self.errors[neuron] = output[i] * (expected[i] - output[i]) * (1 - output[i])
		self.parent.back_prop()

	def weight_update(self):
		super().weight_update()
		self.parent.weight_update()


class Network(object):
	def __init__(self, _in, _out, hidden):
		self.layers = []
		#create layers
		parentLayer = InputLayer(_in)
		self.inputLayer = parentLayer
		self.layers.append(parentLayer)
		for h in hidden:
			newLayer = HiddenLayer(h)
			parentLayer.connect(newLayer)
			parentLayer.weights.randomize()
			self.layers.append(newLayer)
			parentLayer = newLayer

		newLayer = OutputLayer(_out)
		parentLayer.connect(newLayer)
		parentLayer.weights.randomize()
		self.layers.append(newLayer)
		self.outputLayer = newLayer

	def train(self, inputs, expected):
		self.inputLayer.forward_prop(inputs)
		result = self.outputLayer.result()
		error = self.calc_error(result, expected)
		self.outputLayer.back_prop(result, expected)
		self.outputLayer.weight_update()
		return error


	def calc_error(self, result, expected):
		return sum([ math.pow(result[i] - expected[i], 2) for i in range(0, len(result))])



	def recognize(self, inputs):
		self.inputLayer.forward_prop(inputs)
		return self.outputLayer.result()
	
	
	def save(self, fileName = "network.bin"):
		with open(fileName, "wb") as f:
			pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)
			
	@classmethod		
	def load(cls, fileName = "network.bin"):
		with open(fileName, "rb") as f:
			return pickle.load(f)
