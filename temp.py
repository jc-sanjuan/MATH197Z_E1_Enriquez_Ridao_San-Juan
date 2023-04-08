"""
Steepest Descent Method with Armijo Rule and Backtracking
"""
import numpy as np
from numoptim import steepest_descent

def himmelblau(x):
	"""
	Parameter
	---------
# 		x: array
			input 2d vector

	Returns
	-------
		float
	"""

	a = 100.0*(x[1]-x[0]**2)**2
	b = (1-x[0])**2
    
	return a+b

def grad_himmelblau(x):
	"""
	Parameter
	---------
		x:array
		  input 2d vector

	Returns
	-------
		2d vector
	"""

	
	dx0 = 400.0*x[0]**3 - 400.0*x[0]*x[1] + 2*x[0] - 2
	dx1 = 200.0*(x[1]-x[0]**2)
	return np.array([dx0,dx1])


if __name__ == "__main__":
	x = [0.,0.]				#np.array([0.,0.])
	x, grad_norm, it = steepest_descent(himmelblau, x, grad_himmelblau)
	print("Approximate Minimizer: {}" .format(x))
	print("Gradient Norm 		: {}" .format(grad_norm))
	print("Number of Iterations	: {}" .format(it))
	print("Function Value		: {}" .format(himmelblau(x)))