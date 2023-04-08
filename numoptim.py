"""
Pyython Module for Math 197 - numerical Optimization

Name: Raphaell Ridao
2nd Semester AY 2022-2023
"""

import numpy as np 

def armijo_backtrack(fun, d, x, fx, gx, c1=1e-4, rho=0.5, alpha_in=1.0, maxback=30):
	"""
	Parameters
	----------
		d:array
		  current direction
		x:array
		  current point
		fx:float
		  function value at x
		gx:array
		  gradient at x
		c1:float
		  armijo parameter (default value is 1e-4)
		rho:float 
		  backtracking parameter (default value is 0.5)
		alpha_in:float
		  initial step length (default value is 1.0) 
		maxback:int 
			max number of backtracking iterations (default is 30)

	Returns
	-------
		float
			steplength satisfying the Armijo rule or the last steplength
	"""
	alpha = alpha_in
	q = np.dot(gx, d)
	j = 0
	while fun(x + alpha*d)>fx+c1*alpha*q and j<= maxback:
		alpha = rho*alpha
		j = j+1 						# j+=1

	return alpha 


def steepest_descent(fun, x, gradfun, tol=1e-10, maxit=1000):
	"""
	Parameters
	----------
		fun:callable
			objective function
		x:array
			initial point
		gradfun:callable
			gradient of the objective function
		tol:float
			tolerance of the method (default is 1e-10)
		maxit:int
			maximum number of iterationd

	Returns
	-------
		tuple(x,grad_norm,it)
			x:array
				approximate minimizer or last iteration
			grad_norm:float
				norm of the gradient at x
			it:int
				number of iteration
	"""
	it = 0
	grad_norm = np.linalg.norm(gradfun(x))
	while grad_norm>tol and it<=maxit:
		d = -gradfun(x)
		fx = fun(x)
		alpha = armijo_backtrack(fun,d,x,fx,-d)
		x = x + alpha*d
		grad_norm = np.linalg.norm(gradfun(x))
		it = it + 1
	return x,grad_norm,it