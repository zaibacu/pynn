import unittest
from nn import *

ALPHA = 1
class NNTests(unittest.TestCase):
	def setUp(self):
		pass

	def test_create_input_layer(self):
		inputLayer = InputLayer(2)
		self.assertEqual(len(inputLayer.neurons), 3) #2 normal and 1 bias

		hiddenLayer = HiddenLayer(1)
		inputLayer.connect(hiddenLayer)
		self.assertTrue(inputLayer.child == hiddenLayer)
		self.assertTrue(hiddenLayer.parent == inputLayer)

	def test_forward_prop(self):
		inputLayer = InputLayer(2)

		outputLayer = OutputLayer(1)
		inputLayer.connect(outputLayer)

		for key, val in inputLayer.weights.items():
			inputLayer.weights.set(key, 1)

		results = inputLayer.forward_prop([1, 1])
		self.assertEqual(list(map(lambda x: round(x, 3), results)), [ 0.953 ])

	def test_output(self):
		inputLayer = InputLayer(2)
		hiddenLayer = HiddenLayer(2)
		inputLayer.connect(hiddenLayer)
		outputLayer = OutputLayer(2)
		hiddenLayer.connect(outputLayer)

		inputLayer.weights.set_matrix([
				[-1, 2],
				[1, 0],
				[-2, -1]
			])

		hiddenLayer.weights.set_matrix([
				[2, -2],
				[1, -3],
				[1, 0]
			])

		results = inputLayer.forward_prop([1, 2])
		self.assertEqual(list(map(lambda x: round(x, 3), results)), [ 0.906, 0.061 ])

	def test_backprop(self):
		inputLayer = InputLayer(2)
		hiddenLayer = HiddenLayer(2)
		inputLayer.connect(hiddenLayer)
		outputLayer = OutputLayer(2)
		hiddenLayer.connect(outputLayer)

		inputLayer.weights.set_matrix([
				[-1, 2],
				[1, 0],
				[-2, -1]
			])

		hiddenLayer.weights.set_matrix([
				[2, -2],
				[1, -3],
				[1, 0]
			])

		results = inputLayer.forward_prop([1, 2])
		outputLayer.back_prop(results, [1, 0])

		self.assertEqual(round(outputLayer.errors[outputLayer.neurons[0]], 3), 0.008)
		self.assertEqual(round(outputLayer.errors[outputLayer.neurons[1]], 4), -0.0035)

		self.assertEqual(round(hiddenLayer.errors[hiddenLayer.neurons[0]], 4), 0.0045)
		self.assertEqual(round(hiddenLayer.errors[hiddenLayer.neurons[1]], 4), 0.0036)

	def test_weight_update(self):
		inputLayer = InputLayer(2)
		hiddenLayer = HiddenLayer(2)
		inputLayer.connect(hiddenLayer)
		outputLayer = OutputLayer(2)
		hiddenLayer.connect(outputLayer)

		inputLayer.weights.set_matrix([
				[-1, 2],
				[1, 0],
				[-2, -1]
			])

		hiddenLayer.weights.set_matrix([
				[2, -2],
				[1, -3],
				[1, 0]
			])

		inputLayer.forward_prop([1, 2])

		outputLayer.errors[outputLayer.neurons[0]] = 0.008
		outputLayer.errors[outputLayer.neurons[1]] = -0.0035

		hiddenLayer.errors[hiddenLayer.neurons[0]] = 0.0045
		hiddenLayer.errors[hiddenLayer.neurons[1]] = 0.0036
		hiddenLayer.errors[hiddenLayer.neurons[2]] = 0

		outputLayer.weight_update()

		self.assertEqual(round(hiddenLayer.weights.get((hiddenLayer.neurons[0], outputLayer.neurons[0])), 3), 2.002)
		self.assertEqual(round(hiddenLayer.weights.get((hiddenLayer.neurons[0], outputLayer.neurons[1])), 4), -2.0009)






if __name__ == '__main__':
	unittest.main()