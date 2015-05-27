from nn import Network
import math

def run(iterations):
	nn = Network(10, 1, [ 10, 10, 10 ])
	_input = [ 1 ] * 10
	[ nn.train(_input, [1]) for i in range(0, int(iterations))]
	
if __name__ == "__main__":
	from timeit import timeit
	with open("bench_results.txt", "w") as f:
		for i in range(1, 10):
			k = pow(2, i)
			cmd = "run({0})".format(k)
			result = timeit(stmt=cmd, number=100, setup="from __main__ import run")
			f.write("{0},{1}\n".format(k, result))