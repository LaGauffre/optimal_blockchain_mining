
from wealth_process_V_opt import *
from optim import *
from decimal import *
import math as ma
import pandas as pd
import matplotlib.pyplot as plt

import warnings
import time

def optimal_allocation_mean_variance(lam_l, b, f_n, d_n, gam_l, method = 'efficient frontier'):
    mu_ln, b_n = lam_l / d_n, (1-f_n) * b * d_n
    n = len(f_n)-1
    res = {}
    if method == "efficient frontier":
        target_sig_2 = min(b_n**2 * mu_ln) * gam_l + max(b_n**2 * mu_ln) * (1 - gam_l)
        for comb in itertools.combinations(range(n+1), 2):

            l_j, l_i = lam_l / d_n[comb[0]], lam_l / d_n[comb[1]]
            b_j, b_i = (1 - f_n[comb[0]]) * b * d_n[comb[0]], (1 - f_n[comb[1]]) * b * d_n[comb[1]]
            if l_i * b_i**2 < target_sig_2 and l_j * b_j**2 > target_sig_2:
                wi = (l_j * b_j**2 - target_sig_2)/ (l_j * b_j**2 - l_i * b_i**2)
                # print(wi)
                res[comb] = {'position': np.array([wi, 1 - wi]), 'value': wi * l_i * b_i + (1 - wi) * l_j * b_j} 
                # Keep only the item in res associated with the maximum 'value'
                # max_value_key = max(res, key=lambda k: res[k]['value'])
                # res = {max_value_key: res[max_value_key]}
    else:
        k_ast = np.argmax(lam_l * (b * (1 - f_n) * d_n - gam_l * (b * (1 - f_n) * d_n)**2))
        V_ast = np.max(lam_l * (b * (1 - f_n) * d_n - gam_l * (b * (1 - f_n) * d_n)**2))
        res[(k_ast,)] = {'position': np.array([1]), 'value': V_ast}
    return(res)

def optimal_allocation_dividend(c_l, lam_l, x_l, b, f_n, d_n, q, prec, n_comb_max = 3):
    mu_k, b_k = lam_l / d_n, (1-f_n) * b * d_n
    n = len(f_n)-1
    res = {}
    for l in range(n_comb_max):
        for comb in itertools.combinations(range(n+1), l+1):
            

            if (l + 1) == 1:
                X = wealth_process(mu_k[comb[0]], b_k[comb[0]], c_l, l) 
                X.scale_functions()
                res[comb] = {'position':np.array([1]), 'value':X.V(x_l,q)[1]}
            elif (l + 1) == 2:
                w = np.array([1 / 2, 1 / 2])
                X_mixed = wealth_process(mu_k[np.array(comb)], b_k[np.array(comb)], c_l, l, w)
                res_max_scalar = max_scalar(X_mixed, x_l, q)
                res_max_scalar['position'][res_max_scalar['position'] > 1 - prec] = 1
                res_max_scalar['position'][res_max_scalar['position'] < prec] = 0
                res[comb] = res_max_scalar
            else: 
                w = np.ones(l+1) / (l + 1)
                X_mixed = wealth_process(mu_k[np.array(comb)], b_k[np.array(comb)], c_l, l, w)
        
                if (l+1) < 5:
                    K = 10
                else:
                    K = 50
                res_PSO = PSO(X_mixed, x_l, q, K, tol = 0.1, verbose = False)
                res_PSO['position'][res_PSO['position'] > 1 - prec] = 1
                res_PSO['position'][res_PSO['position'] < prec] = 0
                res[comb] = res_PSO
    return(res)


def miner_hashpower_allocate(c_m, lam_m, x_m, gam_m, b, f_n, d_n, q, m, criteria='mean variance frontier'):
  

    mining_network = {}

    for l in range(m):
        lam_l, gam_l = lam_m[l], gam_m[l]
        if criteria == 'mean variance frontier':
            res  = optimal_allocation_mean_variance(lam_l, b, f_n, d_n, gam_l, method = 'efficient frontier')
        elif criteria == 'mean variance utility':
            res  = optimal_allocation_mean_variance(lam_l, b, f_n, d_n, gam_l, method = 'not efficient frontier')
        elif criteria == 'dividend':
            c_l, x_l = c_m[l], x_m[l]
            res  = optimal_allocation_dividend(c_l, lam_l, x_l, b, f_n, d_n, q, prec = 0.001, n_comb_max = 2)
        else:
            raise ValueError("Criteria not implemented. Use 'mean variance frontier' or 'dividend' or 'mean variance utility'.")
        max_res = max(res.items(), key=lambda item: item[1]['value'])
        mining_network[l] = {'comb': max_res[0], 'weight': max_res[1]['position'], 'value': max_res[1]['value'], 'hashpower': lam_m[l]}
    return mining_network

def mining_pool_hashpower_distribute(mining_network, n):

    
    # Initialize an empty dictionary to store the hashpower distribution
    hashpower_distribution = {}
    
    # Initialize all mining pools with zero hashpower
    for pool in range(n + 1):
        if pool not in hashpower_distribution:
            hashpower_distribution[pool] = 0
    
    # Iterate through the mining network to distribute the hashpower
    for miner, data in mining_network.items():
        comb = data['comb']
        weight = data['weight']
        hashpower = data['hashpower']
        
        for pool, w in zip(comb, weight):
            if pool not in hashpower_distribution:
                hashpower_distribution[pool] = 0
            hashpower_distribution[pool] += hashpower * w

    # Convert the dictionary to a DataFrame
    hashpower_df = pd.DataFrame(list(hashpower_distribution.items()), columns=['Mining Pool', 'Hashpower Proportion'])

    # Normalize the hashpower proportion to sum to 1
    hashpower_df['Hashpower Proportion'] /= hashpower_df['Hashpower Proportion'].sum()
    
    return hashpower_df


def hashpower_pie_plot(hashpower_df):
    """
    Plots a pie chart of the hashpower distribution among mining pools.
    
    Parameters:
    hashpower_df (DataFrame): DataFrame containing 'Mining Pool' and 'Hashpower Proportion'.
    """
    
    # Ensure the DataFrame is not empty
    if hashpower_df.empty:
        print("No data to plot.")
        return
    
    # Set the figure size
    plt.figure(figsize=(10, 7))
    # Define colors for each slice
    colors = plt.cm.tab20c.colors
    # Filter out rows with zero hashpower proportion
    hashpower_df = hashpower_df[hashpower_df['Hashpower Proportion'] > 0]
    # Plot the pie chart with different colors and improved text placement
    plt.figure(figsize=(10, 7))
    wedges, texts, autotexts = plt.pie(hashpower_df['Hashpower Proportion'], labels=hashpower_df['Mining Pool'], autopct='%1.1f%%', startangle=140, colors=colors)

    # Improve text placement
    for text in texts + autotexts:
        text.set_fontsize(20)
        text.set_color('black')

    # plt.title('Hashpower Proportion by Mining Pool')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.show()