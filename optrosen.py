"""
MATH 197 Z Exercise 1
Name: Deangelo Enriquez
Raphaell Ridao
Jan Catherine San Juan
Date: 16 April 2023
"""

import numpy as np 
from numpy import *
import math
import sys

def backtracking(fun, d, x, fx, gx, g, ssc, c1=1e-4, rho=0.5, alpha_in=1.0, maxback=30):
    """
	Parameters
	----------
        fun: callable
          objective function
		d:array
		  current direction
		x:array
		  current point
		fx:float
		  function value at x
		gx:array
		  gradient at x
        g:callable
		  gradient of the objective function
        ssc:string/int
          chosen stepsize criterion
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
		alpha: float
			steplength satisfying the chosen rule or the last steplength
	"""
    alpha = alpha_in
    q = np.dot(gx, d)
    j = 0
    condition = conditions(fun, d, x, fx, gx, g, ssc, c1, rho, alpha, q)
    
    
    while condition and j<= maxback:
        alpha = rho*alpha
        j = j+1 						# j+=1
        condition = conditions(fun, d, x, fx, gx, g, ssc, c1, rho, alpha, q)
    return alpha 

def conditions(fun, d, x, fx, gx, g, ssc, c1, rho, alpha, q):
    """
	Parameters
	----------
        fun: callable
          objective function
		d:array
		  current direction
		x:array
		  current point
		fx:float
		  function value at x
		gx:array
		  gradient at x
        g:callable
		  gradient of the objective function
        ssc:string/int
          chosen stepsize criterion
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
		alpha: float
			steplength satisfying the chosen rule or the last steplength
	"""
    
    if ssc == 'Armijo' or ssc == '1':
        condition = fun(x + alpha*d)>fx+c1*alpha*q
    elif ssc == 'Wolfe' or ssc == '2':
        condition = fun(x+alpha*d)>fx+c1*alpha*q or np.dot(g(x + alpha*d), d)<0.9*q
    elif ssc == 'StrongWolfe' or ssc == '3':
        condition = fun(x+alpha*d)>fx+c1*alpha*q or abs(np.dot(g(x + alpha*d), d))>-0.9*q
    elif ssc == 'Goldstein' or ssc == '4':
        condition = fun(x + alpha*d) > fx + 0.25*alpha*q or (fun(x + alpha*d) < fx + 0.75*alpha*q)
    else:
        print("Please input a valid number or the exact criterion name.")
        sys.exit()
    return condition
    

def steepest_descent(fun, x, gradfun, tol=1e-10, maxit=50000):
    ssm = input("Choose method:\n 1. Backtracking\n 2. Polynomial Interpolation\n Input name/number: ")
    ssc = input("Choose stepsize selection criterion:\n 1. Armijo\n 2. Wolfe\n 3. StrongWolfe\n 4. Goldstein\n Input name/number: ")
    
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
        if ssm == 'Backtracking' or ssm == '1':
            alpha = backtracking(fun,d,x,fx,-d, gradfun, ssc)
        elif ssm == 'Polynomial Interpolation' or ssm == '2':
            alpha = polynomial(fun,d,x,fx,-d, gradfun,ssc)
        else:
            print("Please input a valid number or the exact method name.")
            sys.exit()
       
        x = x + alpha*d
        grad_norm = np.linalg.norm(gradfun(x))
        it = it + 1
        
    return x,grad_norm,it

def polynomial(fun, d, x, fx, gx, g, ssc, c1=1e-4, rholo=0.1, rhohi=0.5, alpha_in=1.0, tol=1e-10, maxpol=30):
    """
	Parameters
	----------
        fun: callable
          objective function
		d:array
		  current direction
		x:array
		  current point
		fx:float
		  function value at x
		gx:array
		  gradient at x
        g:callable
		  gradient of the objective function
        ssc:string/int
          chosen stepsize criterion
		c1:float
		  armijo parameter (default value is 1e-4)
		rholo:float 
		  polynomial interpolation parameter (default value is 0.1)
        rhohi:float 
  		  polynomial interpolation parameter (default value is 0.5)
		alpha_in:float
		  initial step length (default value is 1.0) 
		maxpol:int 
			max number of polynomial interpolation iterations (default is 30)

	Returns
	-------
		alpha: float
			steplength satisfying the chosen rule or the last steplength
	"""
    alpha = alpha_in
    q = np.dot(gx, d)
    fj = fun(x + alpha*d)
    j = 0
    alphatemp = alpha
    ftemp = fj
    condition = conditions(fun, d, x, fx, gx, g, ssc, c1, rhohi, alpha, q)
    while condition and j <= maxpol:
        
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
        #fj = fun(x + alpha*d)
        condition = conditions(fun, d, x, fx, gx, g, ssc, c1, rhohi, alpha, q)
        j = j+1
        
    return alpha 

