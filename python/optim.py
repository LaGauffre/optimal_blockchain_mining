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




# Meta optimization functions





# def selection(fitness, cloud, s = 2):
#     K = len(fitness)
#     p_rank =  np.flip((s - (2 * s - 2) * (np.arange(1, K+1, 1)-1) / (K-1)) / K)[fitness.argsort().argsort()]
#     return cloud[np.random.choice(range(K), size=K, replace=True, p=p_rank)]

# def cross_over(parents):
#     # K = len(parents)
#     K, n_col = np.shape(parents)
#     n = n_col - 1 
#     offspring = []
#     for _ in range(K):
#         parent1, parent2 = np.random.choice(range(K), size=2, replace=False)
#         crossover_point = np.random.randint(1, n+1)
#         child = np.concatenate((parents[parent1][:crossover_point], parents[parent2][crossover_point:]))
#         while np.sum(child) ==0:
#             parent1, parent2 = np.random.choice(range(K), size=2, replace=False)
#             crossover_point = np.random.randint(1, n+1)
#             child = np.concatenate((parents[parent1][:crossover_point], parents[parent2][crossover_point:]))
#         offspring.append(child / np.sum(child))
#     return(np.array(offspring))

# def mutation(offspring, mutation_rate = 0.1):
#     K, n_col = np.shape(offspring)

#     mutation_mask = np.random.random(size=(K, n_col)) < mutation_rate
#     mutation_mask
#     offspring[mutation_mask] = np.random.dirichlet(np.ones(n_col), size= K)[mutation_mask]  
#     offspring /= np.sum(offspring, axis=1, keepdims=True)
#     return(offspring)

# def genetic_optimization(X, x, q, K, max_generations, mutation_rate =0.1, s = 2, tol = 0.1, verbose = True, paralell = True, nproc = 4):
#     λ, b, c, n = X.l, X.b, X.c, X.n
#     clouds, fitnesses =[], []
#     initial_cloud = []
#     for ik in itertools.product(range(2), repeat = n+1):
#         if sum(ik) > 0:
#             initial_cloud.append(np.array(ik)/ sum(ik))
#     initial_cloud = np.array(initial_cloud)
#     initial_cloud = np.concatenate([np.array(initial_cloud), np.random.dirichlet(np.ones(n+1), size=K - len(initial_cloud))], axis = 0)
#     clouds.append(initial_cloud)
    
#     def objective_function(w):
#         X_mixed = wealth_process(λ, b, c, n, w)
#         a_ast, V_ast = X_mixed.V(x, q)
#         return(V_ast)
    
#     # compute_fitness = lambda cloud: np.array(Parallel(n_jobs=4)(delayed(objective_function)(i) for i in cloud))
#     if paralell:
#         compute_fitness = lambda cloud: np.array(Parallel(n_jobs=nproc)(delayed(objective_function)(i) for i in cloud))
#     else: 
#         compute_fitness = lambda cloud: np.array([objective_function(i) for i in cloud])
#     cloud = clouds[-1]
#     fitness = compute_fitness(cloud)
#     fitnesses.append(fitness)
#     for gen in range(max_generations):
#         if verbose:
#             print("This is generation #"+str(gen))
#         cloud = clouds[-1]
#         fitness = compute_fitness(cloud)
#         fitnesses.append(fitness)
#         parents = selection(fitness, cloud, s)
#         offspring = cross_over(parents)
#         offspring_new = mutation(offspring, mutation_rate)
#         clouds.append(offspring_new)
#         if verbose:
#             fitness_maxs = np.array([max(fitness) for fitness in fitnesses ])
#             best_position = clouds[np.argmax(fitness_maxs)][np.argmax(fitnesses[np.argmax(fitness_maxs)])]
#             print({'position': best_position , 'value': max(fitness_maxs)})
#         if np.all(np.std(offspring_new , axis = 0) < tol):
#             break
#     fitness_maxs = np.array([max(fitness) for fitness in fitnesses ])
#     best_position = clouds[np.argmax(fitness_maxs)][np.argmax(fitnesses[np.argmax(fitness_maxs)])]
#     best_position_all = {'position': best_position , 'value': max(fitness_maxs)}
#     return(clouds, fitnesses, best_position_all)

# Plot histogram of marginal distributions
# def plot_marginal_histograms(samples):
#     num_samples, num_dimensions = samples.shape
    
#     fig, axs = plt.subplots(num_dimensions, 1, figsize=(8, 6), sharex=True)
#     fig.suptitle('Marginal Distributions')

#     for i in range(num_dimensions):
#         axs[i].hist(samples[:, i], bins=20, density=False, alpha=0.7)
#         axs[i].set_title(f'Dimension {i+1}')

#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.tight_layout()
#     plt.show()
    
# Grid search functions

# def probability_grid(values, n):
#     values = set(values)
#     # Check if we can extend the probability distribution with zeros
#     with_zero = 0. in values
#     values.discard(0.)
#     if not values:
#         raise StopIteration
#     values = list(values)
#     for p in _probability_grid_rec(values, n, [], 0.):
#         if with_zero:
#             # Add necessary zeros
#             p += (0.,) * (n - len(p))
#         if len(p) == n:
#             yield from set(itertools.permutations(p))  # faster: more_itertools.distinct_permutations(p)

# def _probability_grid_rec(values, n, current, current_sum, eps=1e-10):
#     if not values or n <= 0:
#         if abs(current_sum - 1.) <= eps:
#             yield tuple(current)
#     else:
#         value, *values = values
#         inv = 1. / value
#         # Skip this value
#         yield from _probability_grid_rec(
#             values, n, current, current_sum, eps)
#         # Add copies of this value
#         precision = round(-ma.log10(eps))
#         adds = int(round((1. - current_sum) / value, precision))
#         for i in range(adds):
#             current.append(value)
#             current_sum += value
#             n -= 1
#             yield from _probability_grid_rec(
#                 values, n, current, current_sum, eps)
#         # Remove copies of this value
#         if adds > 0:
#             del current[-adds:]


# def grid_search_optimization(X, x, q, grid_size, max_size = 1000, paralell = True, nproc = 4):

#     λ, b, c, n = X.l, X.b, X.c, X.n
#     def objective_function(w):
#         X_mixed = wealth_process(λ, b, c, n, w)
#         a_ast, V_ast = X_mixed.V(x, q)
#         return(V_ast)
#     if paralell: 
#         compute_fitness = lambda cloud: np.array(Parallel(n_jobs=nproc)(delayed(objective_function)(i) for i in cloud))
#     else:
#         compute_fitness = lambda cloud: np.apply_along_axis(objective_function, 1, cloud)

#     values = np.linspace(0,1,grid_size)
#     grid_points = np.array(list(probability_grid(values, n+1)))
#     if len(grid_points) > max_size:
#         print("The grid is too large")
#     else:
#         print(grid_points)
#         fitnesses = compute_fitness(grid_points)
#         return(grid_points[np.argmax(fitnesses)], max(fitnesses))
    


# def pso_dc(X, x, q, K, max_iterations, c1, c2, c3, tol, verbose = True, paralell = True, nproc = 4):
#     λ, b, c, n= X.l, X.b, X.c, X.n
#     best_positions_pso = []
#     for m in np.arange(2, n+2, 1):
#         for comb in itertools.combinations(range(n+1), m):
#             λ_comb, b_comb = λ[np.array(comb)], b[np.array(comb)]
#             if verbose:
#                 print(comb)
#             w = np.random.dirichlet(np.ones(len(comb)), size=1)
#             X = wealth_process(λ_comb, b_comb, c, m-1, w)
#             clouds_pso, fitnesses_pso, best_position_all_pso = pso(X, x, q, K, max_iterations, c1, c2, c3, tol, False, paralell, nproc)
#             best_position_all_pso['comb'] = comb
#             best_positions_pso.append(best_position_all_pso)
#             if verbose:
#                 print(best_position_all_pso)
#     fitnesses = [best_positions_pso[j]['value'] for j in range(len(best_positions_pso))]
#     return(best_positions_pso[np.argmax(fitnesses)])


# def ga_dc(X, x, q, K, max_generations, mutation_rate, s, tol, verbose = True, paralell = True, nproc = 4):
#     λ, b, c, n= X.l, X.b, X.c, X.n
#     best_positions_ga = []
#     for m in np.arange(2, n+2, 1):
#         for comb in itertools.combinations(range(n+1), m):
#             f, δ = fk[np.array(comb)], δk[np.array(comb)]
#             if verbose: 
#                 print(comb)
#             w = np.random.dirichlet(np.ones(len(comb)), size=1)
#             X = wealth_process(λ, b, c, m-1, w)
#             clouds_ga, fitnesses_ga, best_position_all_ga = genetic_optimization(X, x, q, K, max_generations, mutation_rate, s, tol, False, paralell, nproc)
#             best_position_all_ga['comb'] = comb
#             best_positions_ga.append(best_position_all_ga)
#             print(best_position_all_ga)
#     fitnesses = [best_positions_ga[j]['value'] for j in range(len(best_positions_ga))]
#     return(best_positions_ga[np.argmax(fitnesses)])


# def pso_search(X, x, q, K, max_iterations, c1, c2, c3, tol, verbose = True, paralell = True, nproc = -1):
#     λ, b, c, n = X.l, X.b, X.c, X.n
#     best_positions_pso = []
#     for m in np.arange(2, n+2, 1):
#         for comb in itertools.combinations(range(n+1), m):
#             f, δ = fk[np.array(comb)], δk[np.array(comb)]
#             if verbose:
#                 print(comb)
#             w = np.random.dirichlet(np.ones(len(comb)), size=1)
#             X = wealth_process(λ, b, c, m-1, w)
#             clouds_pso, fitnesses_pso, best_position_all_pso = pso(X, x, q, K, max_iterations, c1, c2, c3, tol, False, paralell, nproc)
#             best_position_all_pso['comb'] = comb
#             best_positions_pso.append(best_position_all_pso)
#             if verbose:
#                 print(best_position_all_pso)
#     fitnesses = [best_positions_pso[j]['value'] for j in range(len(best_positions_pso))]
#     return(best_positions_pso)