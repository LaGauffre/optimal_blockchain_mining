# -*- coding: utf-8 -*-
"""
Simulation of jump Levy processes with drift
title: levy_simulation.py
author: Pierre-O Goffard
mail: goffard@unistra.fr
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import math as ma
from scipy import signal
import itertools
from mpmath import *
from numba import jit
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError
from joblib import Parallel, delayed

@overload(np.heaviside)
def np_heaviside(x1, x2):
    @register_jitable
    def heaviside_impl(x1, x2):
        if x1 < 0:
            return 0.0
        elif x1 > 0:
            return 1.0
        else:
            return x2

    return heaviside_impl

@jit(nopython = True)
def g_fun(x, j):
    return np.exp(x) * x**j / ma.gamma(j + 1) * np.heaviside(x, 1)

@jit(nopython = True)
def G_fun(x, j):
    if j % 2 == 0:
        res = sum([(-1)**i * g_fun(x, i) * ma.gamma(j + 1)  for i in range(j+1)]) + (-1)**(j+1)*ma.gamma(j + 1)
    else:
        res = sum([(-1)**(i + 1) * g_fun(x, i) * ma.gamma(j + 1)  for i in range(j+1)]) + (-1)**(j+1) * ma.gamma(j + 1)
    
    return res  * np.heaviside(x, 1) / ma.gamma(j + 1)


@jit(nopython = True)
def barG_fun(x, j):
    if j % 2 == 0:
        res = sum([(-1)**i * G_fun(x, i)  for i in range(j+1)]) + x * (-1)**(j+1)
    else:
        res = sum([(-1)**(i + 1) * G_fun(x, i) for i in range(j+1)]) + x * (-1)**(j+1)
    
    return res  * np.heaviside(x, 1) 

# def g_fun(x, j):
#     return(np.exp(x) * x**j / ma.factorial(j))
# def G_fun(x, j):
#     f = lambda x: np.exp(x) * x**j* np.heaviside(x, 0) 
#     result, _ = sc.integrate.quad(f, 0, x)
#     return result / ma.factorial(j)
# def barG_fun(z, j):
#     f = lambda y, x: np.exp(y) * y**j * np.heaviside(x, 0) 
#     result, _ = sc.integrate.dblquad(f, 0, z, lambda x: 0, lambda x: x)
#     return(result / ma.factorial(j))

@jit(nopython = True)
def multinomial_coefficient(n, ns):
    denominator = 1
    for ni in ns:
        denominator *=  ma.gamma(ni+1)
    return  ma.gamma(n+1) // denominator

class wealth_process:
    def __init__(self, l, b, c, n, wk = None):
        """
        Parameters
        ----------
        l : num
            Block arrival process intensity.
        b : num
            Block reward.
        c : num
            Operational cost
        n : num
            Number of mining pool
        wk : array
            Hash power distribution among the mining pools.   
        Returns
        -------
        Object of type wealth process.

        """
        if n >0:
            self.l, self.b, self.c, self.n, self.wk = l, b, c, n, wk
            self.bk = b
            self.muk =  l * wk
            self.pk = self.muk / sum(self.muk)
        if n == 0:# Solo mining or mining in only one mining pool
            self.l, self.b, self.c, self.n = l, b, c, n
    
    def sample(self, t):
        l, b, c, n = self.l, self.b, self.c, self.n
        if n >0:
            muk, bk, pk = self.muk, self.bk, self.pk

        if n ==0:
            Nt = np.random.poisson(l * t)
            Bi = np.append(0,np.ones(Nt) * b)
        else:
            Nt = np.random.poisson(sum(muk) * t)
            Bi = np.append(0,np.random.multinomial(1, pk, size=Nt)@bk)
        Ti = np.append(0, np.sort(np.random.uniform(0, t, Nt)))
        self.data = {'t': t, 'Nt': Nt, 'Ti': Ti, 'Bi': Bi}

    def Xt(self, x):
        Nt, Ti, Bi = self.data['Nt'], self.data['Ti'], self.data['Bi']
        def traj(s):
            return( x + 
                  sum(
                      [np.heaviside(min(s - Ti[k], 0), Bi[k]) 
                       for k in range(Nt+1)]
                      ) - 
                  self.c * s)
        return(traj)
    
    def sup_Xt(self, x):
        t, Ti, Bi, c = self.data['t'], self.data['Ti'], self.data['Bi'], self.c
        traj_Xt = self.Xt(x)
        XTi = [max(traj_Xt(Ti[i]), 0 ) for i in range(len(Ti))]
        max_Xts = [max(XTi[:i+1]) for i in range(len(Ti))]
        ts = np.append(Ti, t)  
        def traj(s):
            res = sum([
                np.heaviside(int(ts[k]<= s < ts[k+1]),0)*max_Xts[k] 
                for k in range(len(max_Xts))])
            return(res)
        return(traj)
    #     
    def Lt(self, x, a):
        traj_sup_Xt = self.sup_Xt(x)
        def traj(s):
            return(max(a, traj_sup_Xt(s)) - a)
        return(traj)
    def Yt(self, x):
        traj_Xt, traj_sup_Xt = self.Xt(x), self.sup_Xt(x)
        def traj(s):
            return(max(x, traj_sup_Xt(s)) - traj_Xt(s))
        return(traj)
    def Ut(self, x, a):
        traj_Xt, traj_Lt = self.Xt(x), self.Lt(x, a)
        def traj(s):
            return(traj_Xt(s) - traj_Lt(s))
        return(traj)
        
    def E(self, x):
        l, b, c, n = self.l, self.b, self.c, self.n
        if n >0:
            muk, bk, pk = self.muk, self.bk, self.pk

            E = x - c + sum(muk) * bk@pk
        else:
            E = x - c + l * b
    
        return(E)
    
    def Var(self):
        l, b, c, n = self.l, self.b, self.c, self.n
        if n >0:
            muk, bk, pk = self.muk, self.bk, self.pk

            Var = sum(muk) * bk**2@pk
        else:
            Var = l * b**2
        return(Var)

    def ruin_proba(self, x):
        l, b, c, n = self.l, self.b, self.c, self.n
        if n >0:
            muk, bk, pk = self.muk, self.bk, self.pk
            psi = lambda th : th * c + sum(muk) * (np.exp(-th * bk)@pk - 1)
            def phi(q):
                sol = sc.optimize.root_scalar(lambda theta: psi(theta)- q, bracket=[0.001, 1000], method='bisect').root
                return(sol)
        else: 
            def phi(q):
                return(np.real(sc.special.lambertw(-l * b / c *np.exp(- (q + l) * b / c)) / b  + (q + l) / c))
        return(np.exp(-phi(0) * x))
    
    def scale_functions(self):
        l, b, c, n = self.l, self.b, self.c, self.n

        if n >0:
            wk, muk, bk, pk = self.wk, self.muk, self.bk, self.pk
            psi = lambda th : th * c + sum(muk) * (np.exp(-th * bk)@pk - 1)
            phi = lambda q : sc.optimize.root_scalar(lambda theta: psi(theta)- q, bracket=[0.0001, 10000], method='bisect').root
            def Wq(x, q):
                mu = sum(muk)
                exponent = int(x / np.min(bk))
                res = 0

                for ik in itertools.product(range(exponent + 1), repeat=len(bk)):
                    ib = ik@bk
                    j = sum(ik)
                    prob = np.prod(pk**np.array(ik))
                    if ib <= x and j <= exponent and prob >0:
                        res += multinomial_coefficient(j, ik) * \
                        prob * (-mu / (mu +q))**j * g_fun((mu+q) / c * (x - ik@bk), j)  / \
                        c
                return(res)
            def Zq(x, q):
                mu = sum(muk)
                exponent = int(x / np.min(bk))
                res = 0
                exponent
                for ik in itertools.product(range(exponent + 1), repeat=len(bk)):
                    ib = ik@bk
                    #print(iw, ik)
                    j = sum(ik)
                    prob = np.prod(pk**np.array(ik))
                    if ib <= x and j <= exponent and prob >0:
                        # print(j)
                        res += multinomial_coefficient(j, ik) * \
                        prob * (-mu)**j / (mu +q)**(j+1) * G_fun((mu+q) / c * (x - ik@bk), j) 
                return(1 + q * res)
            def barZq(x, q):
                mu = sum(muk)
                exponent = int(x / np.min(bk))
                res = 0
                for ik in itertools.product(range(exponent + 1), repeat=len(bk)):
                    ib = ik@bk
                    #print(iw, ik)
                    j = sum(ik)
                    prob = np.prod(pk**np.array(ik))
                    if ib <= x and j <= exponent and prob >0:
                        res +=  multinomial_coefficient(j, ik) * \
                        prob * (-mu)**j / (mu +q)**(j+2) * barG_fun((mu+q) / c * (x - ik@bk), j) 
                return( x + q * c* res)
            psi_prime = lambda th : c - sum(muk) * (bk * np.exp(-th * bk))@pk
            
            def kap(x, q): 
                return(barZq(x,q) - Zq(x,q) / phi(q) + psi_prime(0) / q)

        else:
            psi = lambda θ : θ * c + l * (np.exp(-b * θ) - 1)
            phi = lambda q : np.real(sc.special.lambertw(-l * b / c *np.exp(- (q + l) * b / c)) / b  + (q + l) / c)
            def Wq(x, q):
                return(np.heaviside(x, 1) * sum([(-1)**k * (l / (l+q))**k * g_fun((l + q) /c * (x - k * b) , k) for k in range(int(x / b) + 1)]) / c)
            def Zq(x, q):
                return(1 + q * sum([(-l)**k / (l+q)**(k +1) * G_fun((l + q) /c * (x - k * b) , k) for k in range(int(x / b) + 1)]))
            def barZq(x, q):
                return(x + q * c * sum([(-l)**k / (l+q)**(k +2) * barG_fun((l + q) /c * (x - k * b) , k) for k in range(int(x / b) + 1)]))
            psi_prime = lambda θ : c - b * l * np.exp(-b * θ)
            def kap(x,q):
                return(barZq(x,q) - Zq(x,q) / phi(q) + psi_prime(0) / q)
        
        self.psi, self.phi, self.psi_prime  = psi, phi, psi_prime
        self.Wq, self.Zq, self.barZq, self.kap = Wq, Zq, barZq, kap
        
            
        
    
    def V(self, x, q):
        l, b, c, n = self.l, self.b, self.c, self.n
        if n >0:
            wk, muk, bk, pk = self.wk, self.muk, self.bk, self.pk

            psi = lambda th : th * c + sum(muk) * (np.exp(-th * bk)@pk - 1)
            phi_q = sc.optimize.root_scalar(lambda theta: psi(theta)- q, bracket=[0.0001, 10000], method='bisect').root
            # def phi(q):
            #     sol = sc.optimize.root_scalar(lambda theta: psi(theta)- q, bracket=[0.0001, 10000], method='bisect').root
            #     return(sol)
            def Wq(x, q):
                mu = sum(muk)
                exponent = int(x / np.min(bk))
                res = 0

                for ik in itertools.product(range(exponent + 1), repeat=len(bk)):
                    ib = ik@bk
                    j = sum(ik)
                    prob = np.prod(pk**np.array(ik))
                    if ib <= x and j <= exponent and prob >0:
                        res += multinomial_coefficient(j, ik) * \
                        prob * (-mu / (mu +q))**j * g_fun((mu+q) / c * (x - ik@bk), j)  / \
                        c
                        return(res)
            def Zq(x, q):
                mu = sum(muk)
                exponent = int(x / np.min(bk))
                res = 0
                exponent
                for ik in itertools.product(range(exponent + 1), repeat=len(bk)):
                    ib = ik@bk
                    #print(iw, ik)
                    j = sum(ik)
                    prob = np.prod(pk**np.array(ik))
                    if ib <= x and j <= exponent and prob >0:
                        # print(j)
                        res += multinomial_coefficient(j, ik) * \
                        prob * (-mu)**j / (mu +q)**(j+1) * G_fun((mu+q) / c * (x - ik@bk), j) 
                return(1 + q * res)
            def barZq(x, q):
                mu = sum(muk)
                exponent = int(x / np.min(bk))
                res = 0
                for ik in itertools.product(range(exponent + 1), repeat=len(bk)):
                    ib = ik@bk
                    #print(iw, ik)
                    j = sum(ik)
                    prob = np.prod(pk**np.array(ik))
                    if ib <= x and j <= exponent and prob >0:
                        res +=  multinomial_coefficient(j, ik) * \
                        prob * (-mu)**j / (mu +q)**(j+2) * barG_fun((mu+q) / c * (x - ik@bk), j) 
                return( x + q * c* res)
            psi_prime = lambda th : c - sum(muk) * (bk * np.exp(-th * bk))@pk
            psi_prime_0 = psi_prime(0)
            kap = lambda y: barZq(y,q) - Zq(y,q) / phi_q + psi_prime_0 / q
            def v(x, a, q):
                return(- kap(a-x) + Zq(a-x, q) / Zq(a, q) * kap(a))
            obj = lambda a: barZq(a, q)+ psi_prime_0 / q
            a1, ϵ = 0, 1
            while obj(a1) < 0:
                a1 += ϵ
            a_ast = sc.optimize.root_scalar(obj, bracket=[0, a1], method='brentq').root
        else:
            psi = lambda θ : θ * c + l * (np.exp(-b * θ) - 1)
            phi = lambda q : np.real(sc.special.lambertw(-l * b / c *np.exp(- (q + l) * b / c)) / b  + (q + l) / c)
            def Wq(x, q):
                return(np.heaviside(x, 1) * sum([(-1)**k * (l / (l+q))**k * g_fun((l + q) /c * (x - k * b) , k) for k in range(int(x / b) + 1)]) / c)
            def Zq(x, q):
                return(1 + q * sum([(-l)**k / (l+q)**(k +1) * G_fun((l + q) /c * (x - k * b) , k) for k in range(int(x / b) + 1)]))
            def barZq(x, q):
                return(x + q * c * sum([(-l)**k / (l+q)**(k +2) * barG_fun((l + q) /c * (x - k * b) , k) for k in range(int(x / b) + 1)]))
            psi_prime = lambda θ : c - b * l * np.exp(-b * θ)
            kap = lambda y: barZq(y,q) - Zq(y,q) / phi(q) + psi_prime(0) / q
            def v(x, a, q):
                return(- kap(a-x) + Zq(a-x, q) / Zq(a, q) * kap(a))
            obj = lambda a: barZq(a, q)+ psi_prime(0) / q
            a1, ϵ = 0, 1
            while obj(a1) < 0:
                a1 += ϵ
            a_ast = sc.optimize.root_scalar(obj, bracket=[0, a1], method='brentq').root
            
            
        return(a_ast, v(x, a_ast, q))