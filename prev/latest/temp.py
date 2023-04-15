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
    dx = np.zeros(100)	
    i = 1
    dx[0] = 400.0*x[0]**3 - 400.0*x[0]*x[1] + 2*x[0] - 2
    #for i in range(99):
    while i < 99:
        #dx[i] = 400.0*x[i]**3 - 400.0*x[i]*x[i+1] + 2*x[i] - 2
        #dx[i+1] = 200.0*(x[i+1]-x[i]**2)
        dx[i] = 400.0*x[i]**3 - 400.0*x[i+1]*x[i] + 2*x[i] - 2 + 200.0*(x[i]-x[i-1]**2)
        i = i+1
    #dx[i] = 400.0*x[i]**3 -400*x[i] - 2*x[i] - 2 + 200.0*(x[i]-x[i-1]**2)
    dx[99] = 200.0*(x[99]-(x[98]**2))
    return dx
"""
    dx=np.array([0],[0])
    dx[0] = 400.0*x[0]**3 - 400.0*x[0]*x[1] + 2*x[0] - 2
    dx[1] = 200.0*(x[1]-x[0]**2)
    
    return dx
"""

if __name__ == "__main__":
	x = np.zeros(100)			#np.array([0.,0.])
	x, grad_norm, it = steepest_descent(himmelblau, x, grad_himmelblau)
	print("Approximate Minimizer: {}" .format(x))
	print("Gradient Norm 		: {}" .format(grad_norm))
	print("Number of Iterations	: {}" .format(it))
	print("Function Value		: {}" .format(himmelblau(x)))