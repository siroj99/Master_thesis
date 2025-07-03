import array
from matplotlib.pylab import LinAlgError
import numpy as np
from copy import deepcopy
import dionysus as dio
import scipy
import scipy.stats as ss
import pandas as pd
from sympy import use
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gudhi as gd

import time
import torch
from torch.nn import functional as F
torch.set_default_dtype(torch.float64)

def compute_boundary_matrices(f: dio.Filtration, weight_fun, device = "cpu"):
    t_old = 0
    relevant_times = []
    n_simplicies_seen_per_time = []
    n_simplicies_seen_total = []

    for s in f:
        if s.data != t_old:
            relevant_times.append(t_old)
            n_simplicies_seen_per_time.append(deepcopy(n_simplicies_seen_total))
            t_old = s.data

        q = s.dimension()
        while q >= len(n_simplicies_seen_total):
            n_simplicies_seen_total.append(0)
        n_simplicies_seen_total[q] += 1

    # To be safe add a q+1 entry, in case of cycles appearing in q+1, but not boundaries.
    n_simplicies_seen_total.append(0)

    relevant_times.append(s.data)
    n_simplicies_seen_per_time.append(deepcopy(n_simplicies_seen_total))
    maxq = len(n_simplicies_seen_per_time[-1])
    for qi in range(len(n_simplicies_seen_per_time)):
        n_simplicies_seen_per_time[qi] = n_simplicies_seen_per_time[qi] + [0]*(maxq-len(n_simplicies_seen_per_time[qi]))

    def simplices_at_time(t):
        # i = 0
        if t == torch.inf:
            return n_simplicies_seen_per_time[-1]
        
        i = np.searchsorted(relevant_times, t) 
        # while relevant_times[i] < t:
        #     i += 1
        #     if i >= len(relevant_times) - 1: # Changed this to >= from ==.
        #         return n_simplicies_seen_per_time[-1]
        return n_simplicies_seen_per_time[i]
    
    relevant_times = np.array(relevant_times)

    simplices_at_end = simplices_at_time(torch.inf)

    # boundary_matrices = [0] + [torch.zeros((simplices_at_end[q-1], simplices_at_end[q])) for q in range(1, maxq)]
    boundary_matrices = [torch.zeros((simplices_at_end[max(0,q-1)], simplices_at_end[q]), device = device) for q in range(maxq)]
    name_to_idx = [{} for _ in range(maxq)]
    simplex_bar = tqdm(range(len(f)), leave=False, desc=f"Computing boundary matrices")
    for s in f:
        simplex_bar.update(1)
        q = s.dimension()

        name = str(s).split("<")[1].split(">")[0]
        idx = len(name_to_idx[q])
        name_to_idx[q][name] = idx

        if q > 0:
            for i, bdry_simplex in enumerate(s.boundary()):
                name_bdry_simplex = str(bdry_simplex).split("<")[1].split(">")[0]  
                idx_bdry_simplex = name_to_idx[q-1][name_bdry_simplex]

                # Boundaries as described in OG (Dawson) paper
                boundary_matrices[q][idx_bdry_simplex, idx] = weight_fun(name)/weight_fun(name_bdry_simplex)*(-1)**i

                # Assuming product weights
                # for v in s:
                #     if v not in bdry_simplex:
                #         name_bdry_simplex = str(v)
                #         break
                # boundary_matrices[q][idx_bdry_simplex, idx] = weight_fun(name_bdry_simplex)*(-1)**i

    # print("Computing boundary matrices done.")

    return boundary_matrices, name_to_idx, simplices_at_time, relevant_times

def combinatorial_Laplacian(boundary_matrices, q, t, simplices_at_time, device = "cpu"):
    """
    Computes the combinatorial Laplacian at time t.
    """
    
    Bq = boundary_matrices[q][:simplices_at_time(t)[q-1], :simplices_at_time(t)[q]]
    Bqp1 = boundary_matrices[q+1][:simplices_at_time(t)[q], :simplices_at_time(t)[q+1]]

    return Bqp1 @ Bqp1.T + Bq.T @ Bq 

def compute_combinatorial_Laplacian(f, weight_fun, q, t, device="cpu"):
    """
    Computes the combinatorial Laplacian at time t for dimension q.
    """
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun, device=device)

    return combinatorial_Laplacian(boundary_matrices, q, t, simplices_at_time, device=device)

def combinatorial_Laplacian_eigenvalues(f, weight_fun, max_dim=1, device="cpu"):
    """
    Computes the eigenvalues of the combinatorial Laplacian at all times.
    """
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun, device=device)

    eigenvalues = {}
    for q in range(max_dim+1):
        eigenvalues[q] = {}
        for t in tqdm(relevant_times, leave=False, desc=f"Computing eigenvalues for q={q}"):
            L = combinatorial_Laplacian(boundary_matrices, q, t, simplices_at_time, device=device)
            if L.shape[0] == 0:
                eigenvalues[q][t] = np.array([])
                continue
            try:
                eigvals = torch.linalg.eigvalsh(L)
                eigenvalues[q][t] = eigvals.cpu().numpy()
            except LinAlgError:
                print(f"LinAlgError at time {t} for q={q}.")
                eigenvalues[q][t] = np.array([])

    return eigenvalues, name_to_idx, simplices_at_time, relevant_times

def plot_combinatorial_eigenvalues(eigenvalues, relevant_times, eval_fun = lambda x: np.min(x), max_dim=1, labels=None):
    """
    Plots the eigenvalues of the combinatorial Laplacian at all times.
    """

    fig, ax = plt.subplots(max_dim+1, 1, figsize=(10, 5*(max_dim+1)), sharex=True)
    fig.tight_layout()
    ax = ax.flatten() if max_dim > 0 else [ax]

    if labels is None:
        labels = [lambda q : fr'$\lambda_{q}$', lambda q : fr'$\beta_{q}$']

    for q in range(max_dim+1):
        zero_evals = []
        pos_evals = []
        x_axis = []
        y_lap = []
        y_betti = []

        for t_i, t in enumerate(relevant_times):
            cur_evals = eigenvalues[q][t]
            mesh_zero = cur_evals < 1e-10

            x_axis.append(t)
            cur_lap = eval_fun(cur_evals[~mesh_zero]) if len(cur_evals[~mesh_zero]) > 0 else 0
            cur_betti = len(cur_evals[mesh_zero])
            y_lap.append(cur_lap)
            y_betti.append(cur_betti)

            if t_i < len(relevant_times) - 1:
                x_axis.append(relevant_times[t_i+1])
                y_lap.append(cur_lap)
                y_betti.append(cur_betti)
                
            # zero_evals.append(cur_evals[mesh_zero])
            # pos_evals.append(cur_evals[~mesh_zero])
        
        ax_betti = ax[q].twinx()
        p1, = ax[q].plot(x_axis, y_lap, label=labels[0](q), color='blue')
        p2, = ax_betti.plot(x_axis, y_betti, label=labels[1](q), color='red', linestyle='--')
        
        # ax[q].plot(relevant_times, [eval_fun(evals) if len(evals) > 0 else None for evals in pos_evals], label=f'q={q} (laplacian)', color='blue', linestyle='--')
        # ax_betti.plot(relevant_times, [len(evals) for evals in zero_evals], label=f'q={q} (betti)', color='orange')
        
        ax[q].set_title(f"q={q}")
        ax[q].set_ylim(0, np.max(y_lap)*1.1)
        ax[q].set_ylabel(labels[0](q), fontsize=20)
        ax[q].yaxis.label.set_color(p1.get_color())

        ax_betti.set_ylabel(labels[1](q), fontsize=20)
        ax_betti.set_ylim(0, np.max(y_betti)+1)
        ax_betti.set_yticks(range(0, np.max(y_betti)+2))
        ax_betti.yaxis.label.set_color(p2.get_color())

        ax[q].legend(handles=[p1, p2], loc='upper left')
        # ax_betti.legend()

    ax[-1].set_xlabel('Time', fontsize=20)


    fig.show()
    # plt.xlabel('Time')
    # plt.ylabel('Eigenvalues')
    # plt.title('Eigenvalues of Combinatorial Laplacian')
    # plt.legend()
    # plt.grid()
    # plt.show()

def combinatorial_laplacian_filtration(f, weight_fun, max_dim=1, device="cpu", eval_fun=lambda x: np.min(x), labels=None):
    eigenvalues, name_to_idx, simplices_at_time, relevant_times = combinatorial_Laplacian_eigenvalues(f, weight_fun, max_dim=max_dim, device=device)
    plot_combinatorial_eigenvalues(eigenvalues, relevant_times, eval_fun=eval_fun, max_dim=max_dim, labels=labels)