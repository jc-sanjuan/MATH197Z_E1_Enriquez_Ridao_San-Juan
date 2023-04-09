"""
Python Module for Math 197 - numerical Optimization

Name: Raphaell Ridao
2nd Semester AY 2022-2023
"""

import numpy as np 
from numpy import *
import math

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

def wolfe_conditions(fun, d, x, fx, gx, g, c1=1e-4, rho=0.5, alpha_in=1.0, maxback=30):
	j = 0
	alpha = alpha_in
	q = np.dot(gx, d)
	while fun(x+alpha*d)>fx+c1*alpha*q or np.dot(g(x + alpha*d), d)<0.9*q and j<= maxback:
		alpha = rho*alpha
		j = j+1 
	return alpha

def strongWolfe_conditions(fun, d, x, fx, gx, g, c1=1e-4, rho=0.5, alpha_in=1.0, maxback=30):
    j = 0
    alpha = alpha_in
    q = np.dot(gx, d)    
    while fun(x+alpha*d)>fx+c1*alpha*q or abs(np.dot(g(x + alpha*d), d)>0.9*q) and j<= maxback:
        alpha = rho*alpha
        j = j+1 
    return alpha


def goldstein_conditions(fun, d, x, fx, gx, rho=0.5, alpha_in=1.0, maxback=30):
	j = 0
	alpha = alpha_in
	q = np.dot(gx, d)
	while fun(x + alpha*d) > fx + 0.25*alpha*q or (fun(x + alpha*d) >= fx + 0.75*alpha*q and j <= maxback):
		alpha = rho*alpha
		j = j+1 
	return alpha

def steepest_descent(fun, x, gradfun, tol=1e-10, maxit=50000):

    operator = input("Choose stepsize selection criterion:\n 1. Armijo\n 2. Wolfe\n 3. StrongWolfe\n 4. Goldstein\n 5. Polynomial-Armijo\n 6. Polynomial-Wolfe\n 7. Polynomial-StrongWolfe\n 8. Polynomial-Goldstein\n Input name: ")
    print(operator)
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
        if operator == 'Armijo' or operator == '1':
            alpha = armijo_backtrack(fun,d,x,fx,-d)
        elif operator == 'Wolfe' or operator == '2':
            alpha = wolfe_conditions(fun,d,x,fx,-d, gradfun)
        elif operator == 'StrongWolfe' or operator == '3':
            alpha = strongWolfe_conditions(fun,d,x,fx,-d, gradfun)
        elif operator == 'Goldstein' or operator == '4':
            alpha = goldstein_conditions(fun,d,x,fx,-d)
        elif operator == 'Polynomial-Armijo' or operator == '5':
            alpha = polynomial_armijo(fun,d,x,fx,-d, gradfun)
        elif operator == 'Polynomial-Wolfe' or operator == '6':
            alpha = polynomial_wolfe(fun,d,x,fx,-d, gradfun)
        elif operator == 'Polynomial-StrongWolfe' or operator == '7':
            alpha = polynomial_strongwolfe(fun,d,x,fx,-d, gradfun)
        elif operator == 'Polynomial-Goldstein' or operator == '8':
            alpha = polynomial_goldstein(fun,d,x,fx,-d, gradfun)

        x = x + alpha*d
        grad_norm = np.linalg.norm(gradfun(x))
        it = it + 1
        
    return x,grad_norm,it

def polynomial_armijo(fun, d, x, fx, gx, g, c1=1e-4, rholo=0.1, rhohi=0.5, alpha_in=1.0, tol=1e-10, maxpol=30):
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
    fj = fun(x + alpha*d)
    j = 0
    alphatemp = alpha
    ftemp = fj
    
    while fj > fx+c1*alpha*q and j <= maxpol:
        
        if j == 0:
            alphastar = ((-q)*alpha**2)/(2*(fj-fx-(alpha*q)))
            
        else:
            A = np.array([[np.float64(alphatemp**2), np.float64(alphatemp**3)],[np.float64(alpha**2), np.float64(alpha**3)]])
            B = np.array([(np.float64(ftemp-fx-(alphatemp*q))),(np.float64(fj-fx-(alpha*q)))])
            
            C = np.linalg.solve(A,B)
            
            if (C[0]**2)-3*C[1]*q >= 0 and np.fabs(C[1]) > 10e-10:
                alphastar = (-(C[0])+ np.sqrt((C[0]**2) - 3*C[1]*q))/(3*C[1])
            else:
                alphastar = alpha
            
        alphatemp = alpha
        alpha = max((rholo*alpha),min(alphastar, rhohi*alpha))
        ftemp = fj
        fj = fun(x + alpha*d)
        j = j+1
        
    return alpha 

def polynomial_wolfe(fun, d, x, fx, gx, g, c1=1e-4, rholo=0.1, rhohi=0.5, alpha_in=1.0, tol=1e-10, maxpol=30):
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
    fj = fun(x + alpha*d)
    j = 0
    alphatemp = alpha
    ftemp = fj
    
    while fj > fx+c1*alpha*q or np.dot(g(x + alpha*d), d)<0.9*q and j <= maxpol:

        if j == 0:
            alphastar = ((-q)*alpha**2)/(2*(fj-fx-(alpha*q)))
            
        else:
            A = np.array([[np.float64(alphatemp**2), np.float64(alphatemp**3)],[np.float64(alpha**2), np.float64(alpha**3)]])
            B = np.array([(np.float64(ftemp-fx-(alphatemp*q))),(np.float64(fj-fx-(alpha*q)))])
            
            C = np.linalg.solve(A,B)
            
            if (C[0]**2)-3*C[1]*q >= 0 and np.fabs(C[1]) > 10e-10:
                alphastar = (-(C[0])+ np.sqrt((C[0]**2) - 3*C[1]*q))/(3*C[1])
            else:
                alphastar = alpha
            
        alphatemp = alpha
        alpha = max((rholo*alpha),min(alphastar, rhohi*alpha))
        ftemp = fj
        fj = fun(x + alpha*d)
        j = j+1
        
    return alpha 

def polynomial_strongwolfe(fun, d, x, fx, gx, g, c1=1e-4, rholo=0.1, rhohi=0.5, alpha_in=1.0, tol=1e-10, maxpol=30):
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
    fj = fun(x + alpha*d)
    j = 0
    alphatemp = alpha
    ftemp = fj
    
    while fj > fx+c1*alpha*q or abs(np.dot(g(x + alpha*d), d))>-0.9*q and j <= maxpol:

        if j == 0:
            alphastar = ((-q)*alpha**2)/(2*(fj-fx-(alpha*q)))
            
        else:
            A = np.array([[np.float64(alphatemp**2), np.float64(alphatemp**3)],[np.float64(alpha**2), np.float64(alpha**3)]])
            B = np.array([(np.float64(ftemp-fx-(alphatemp*q))),(np.float64(fj-fx-(alpha*q)))])
            
            C = np.linalg.solve(A,B)
            
            if (C[0]**2)-3*C[1]*q >= 0 and np.fabs(C[1]) > 10e-10:
                alphastar = (-(C[0])+ np.sqrt((C[0]**2) - 3*C[1]*q))/(3*C[1])
            else:
                alphastar = alpha
            
        alphatemp = alpha
        alpha = max((rholo*alpha),min(alphastar, rhohi*alpha))
        ftemp = fj
        fj = fun(x + alpha*d)
        j = j+1
        
    return alpha 

def polynomial_goldstein(fun, d, x, fx, gx, g, c1=1e-4, rholo=0.1, rhohi=0.5, alpha_in=1.0, tol=1e-10, maxpol=30):
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
    fj = fun(x + alpha*d)
    j = 0
    alphatemp = alpha
    ftemp = fj
    
    while fx+0.75*alpha*q>fj or fj>fx+0.25*alpha*q and j <= maxpol:
        if j == 0:
            alphastar = ((-q)*alpha**2)/(2*(fj-fx-(alpha*q)))
            
        else:
            A = np.array([[np.float64(alphatemp**2), np.float64(alphatemp**3)],[np.float64(alpha**2), np.float64(alpha**3)]])
            B = np.array([(np.float64(ftemp-fx-(alphatemp*q))),(np.float64(fj-fx-(alpha*q)))])
            
            C = np.linalg.solve(A,B)
            
            if (C[0]**2)-3*C[1]*q >= 0 and np.fabs(C[1]) > 10e-10:
                alphastar = (-(C[0])+ np.sqrt((C[0]**2) - 3*C[1]*q))/(3*C[1])
            else:
                alphastar = alpha
            
        alphatemp = alpha
        alpha = max((rholo*alpha),min(alphastar, rhohi*alpha))
        ftemp = fj
        fj = fun(x + alpha*d)
        j = j+1
        
    return alpha 