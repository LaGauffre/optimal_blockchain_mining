# -*- coding: utf-8 -*-
"""
Perform computing resources allocation optimization via various algorithms.
title: optim.py
author: Pierre-O Goffard
mail: goffard@unistra.fr
"""

from wealth_process_V_opt import *
import matplotlib.pyplot as plt

# Optimization in the presence of only two pools
def max_scalar(X, x, q):
    λ, b, c, n = X.l, X.b, X.c, X.n
    def objective_function(w0):
        w = np.array([w0, 1-w0])
        X_mixed = wealth_process(λ, b, c, n, w)
        a_ast, V_ast = X_mixed.V(x, q)
        return(-V_ast)

    res = sc.optimize.minimize_scalar(objective_function, bounds=(0, 1), method='bounded')
    best_position = {'position': np.array([res.x, 1-res.x]) , 'value': -res.fun}                 
    return(best_position)

# Sequential Least Square programming algorithm
def SLSP(X, x, q):
    λ, b, c, n = X.l, X.b, X.c, X.n
    initial_cloud = []
    for ik in itertools.product(range(2), repeat = n+1):
            if sum(ik) > 0:
                initial_cloud.append(np.array(ik)/ sum(ik))

    def constraint_sum_to_one(w):
        """Constraint function: ensure that the sum of components of w is equal to one."""
        return np.sum(w) - 1.0
    bounds = [(0, 1)] * (n+1)
    def objective_function(w):
        X_mixed = wealth_process(λ, b, c, n, w)
        _, V_ast = X_mixed.V(x, q)
        return(-V_ast)
        # Define equality constraint: sum of components of w equals one
    constraints = [{'type': 'eq', 'fun': constraint_sum_to_one}]
    res_SLSP = []
    for w_init in initial_cloud:
        res = sc.optimize.minimize(objective_function, w_init, bounds=bounds, constraints=constraints)
        res_SLSP.append({'position':res.x, 'value':-res.fun})
    best_res = max(res_SLSP, key=lambda item: item['value'])
    
    return(best_res)
    

# Nelder-Mead algorithm
def NLM(X, x, q):
    λ, b, c, n = X.l, X.b, X.c, X.n

    initial_simplex = np.concatenate([np.eye(n+1), np.ones((1, n+1)) / (n + 1)], axis=0)
    def penalty(w):
            """Penalty term for deviation from the unit simplex."""
            return 1000 * abs(np.sum(w) - 1) + 1000 * np.sum(np.maximum(-w, 0))  # Penalize sum deviations and negative values

    def objective_function(w):
        
        X_mixed = wealth_process(λ, b, c, n, w)
        a_ast, V_ast = X_mixed.V(x, q)
        return -V_ast + penalty(w)  # Minimize negative value with penalty term

    # Nelder-Mead optimization with custom simplex
    res = sc.optimize.minimize(objective_function, x0 = initial_simplex[0], method='Nelder-Mead',
                            options={'initial_simplex': initial_simplex})

    return({'position':res.x, 'value':-res.fun})

def project_to_simplex(w):
            """Projects a point w onto the unit simplex."""
            sorted_w = np.sort(w)[::-1]
            cumsum_w = np.cumsum(sorted_w)
            rho = np.where(sorted_w - (cumsum_w - 1) / (np.arange(len(w)) + 1) > 0)[0][-1]
            theta = (cumsum_w[rho] - 1) / (rho + 1)
            return np.maximum(w - theta, 0)

def PSO(X, x, q, K, max_iterations = 100, c1  = 0.7, c2 = 1.5, c3 = 1.5, tol = 0.01, verbose = True, paralell = False, nproc = 4):
    λ, b, c, n= X.l, X.b, X.c, X.n

    clouds, fitnesses, velocities =[], [], []
    initial_cloud = []
    for ik in itertools.product(range(2), repeat = n+1):
        if sum(ik) > 0:
            initial_cloud.append(np.array(ik)/ sum(ik))
    initial_cloud = np.concatenate([np.array(initial_cloud), np.random.dirichlet(np.ones(n+1), size=K - len(initial_cloud))], axis = 0)
    clouds.append(initial_cloud)

    def objective_function(w):

        X_mixed = wealth_process(λ, b, c, n, w)
        a_ast, V_ast = X_mixed.V(x, q)
        return(V_ast)

    if paralell:
        compute_fitness = lambda cloud: np.array(Parallel(n_jobs=nproc)(delayed(objective_function)(i) for i in cloud))
    else: 
        compute_fitness = lambda cloud: np.array([objective_function(i) for i in cloud])

    velocities.append(np.zeros((K, n+1)))
    fitnesses.append(compute_fitness(initial_cloud))
    cloud, fitness, velocity = clouds[-1], fitnesses[-1], velocities[-1]
    best_position_all = {'position': cloud[np.argmax(fitness),:] , 'value': max(fitness)}
    best_position_particle = {'positions': cloud , 'values': fitness}

    for i in range(max_iterations):
        if verbose:
            print("Iteration #" + str(i+1) + 
                  " the velocity is " + str(np.linalg.norm(velocities[-1], axis=1).max()) + 
                  " and the best particle is " + 
                  str(best_position_all['position']) + " with fitness " + str(best_position_all['value']))
        # # Update the position of the clouds
        cloud, fitness, velocity = clouds[-1], fitnesses[-1], velocities[-1]
        new_velocity = c1 * velocity + c2 * np.random.rand(K, n+1) * (np.tile(best_position_all['position'], (K, 1)) - cloud) + c3 * np.random.rand(K, n+1) * (best_position_particle['positions'] - cloud)
        new_velocity = np.apply_along_axis(lambda v: v - np.mean(v), 1, new_velocity)
        velocities.append(new_velocity)
        new_cloud = cloud + new_velocity
        new_cloud_after_projection = np.apply_along_axis(project_to_simplex, 1, new_cloud)
        clouds.append(new_cloud_after_projection)
        fitnesses.append(compute_fitness(clouds[-1]))

        # Update best_position_all if a fitter particle is found
        if max(fitnesses[-1]) > best_position_all['value']:
            best_position_all = {'position': new_cloud_after_projection[np.argmax(fitnesses[-1]),:], 'value': max(fitnesses[-1])}
        # BEGIN: Update best_position_particle
        for j in range(K):
            if fitnesses[-1][j] > best_position_particle['values'][j]:
                best_position_particle['positions'][j] = new_cloud_after_projection[j]
                best_position_particle['values'][j] = fitnesses[-1][j]
        # END: Update best_position_particle
        if np.linalg.norm(velocities[-1], axis=1).max() < tol:
            break
    return best_position_all




