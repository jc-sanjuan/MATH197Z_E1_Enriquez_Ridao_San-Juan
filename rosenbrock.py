"""
MATH 197 Z Exercise 1
Name: Deangelo Enriquez
Raphaell Ridao
Jan Catherine San Juan
Date: 16 April 2023
"""
import numpy as np
from optrosen import steepest_descent

def rosenbrock(x):
    """
	Parameter
	---------
 		x: array
			input n dimensional array

	Returns
	-------
		a:float
            rosenbrock function
	"""

    a = 0
    i = 0
    while i<99:
        a = a + 100.0*(x[i+1]-x[i]**2)**2 + (1-x[i])**2
        i=i+1

    return a

def grad_rosenbrock(x):
    """
	Parameter
	---------
		x:array
		  input n dimensional array

	Returns
	-------
		dx:2d vector
          gradient
	"""
    dx = np.zeros(100)	

    i = 1
    dx[0] = 400.0*x[0]**3 - 400.0*x[0]*x[1] + 2*x[0] - 2
    
    while i < 99:
        dx[i] = dx[i] + 400.0*x[i]**3 - 400.0*x[i+1]*x[i] + 2*x[i] - 2 + 200.0*(x[i]-x[i-1]**2)
        i = i+1
    
    dx[i]= 200.0*(x[i]-x[i-1]**2)
    
    return dx


if __name__ == "__main__":
	x = np.zeros(100)			
	x, grad_norm, it = steepest_descent(rosenbrock, x, grad_rosenbrock)
	print("Approximate Minimizer: {}" .format(x))
	print("Gradient Norm 		: {}" .format(grad_norm))
	print("Number of Iterations	: {}" .format(it))
	print("Function Value		: {}" .format(rosenbrock(x)))
