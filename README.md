PyNN
----
Yet another Neural Network implementation with Python

Usage
-----
1) New network:

```Python
from nn import Network
	
nn = Network(5, 1, [ 10 10 10 ]) -- Creates network with 5 input neurons, 1 output neuron and 3 hidden layers, 10 neurons each

```
2) Save network
```Python
nn.save("network.bin")
```
3) Load network
```Python
nn = Network.load("network.bin")
```
4) Train
```Python
nn.train([1 0 1 0 0], [1])
```
5) Predict
```Python
nn.predict([1 0 1 0 0])
```
