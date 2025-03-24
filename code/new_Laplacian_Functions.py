from multiprocessing import Value
import numpy as np
from copy import deepcopy
import dionysus as d
import scipy
import scipy.linalg
import scipy.stats as ss
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gudhi as gd
from Laplacian_Functions import *


def cross_Laplacian_cor10(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=False):
    t, s = relevant_times[t_i], relevant_times[s_i]
    tm1 = relevant_times[t_i-1]

    # If no new q+1 simplices, then it is just 0
    if simplices_at_time(t)[q+1] == simplices_at_time(tm1)[q+1]:
        return np.zeros((simplices_at_time(s)[q], simplices_at_time(s)[q]))

    if verb:
        print(f"Bqplus1:\n{boundary_matrices[q+1]}")
        print(f"n_q_t:{simplices_at_time(t)}, n_q_s: {simplices_at_time(s)}")
    if s_i > 0:
        sm1 = relevant_times[s_i-1]
        B12_sm1t = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
        if B12_sm1t.shape[1] == 0 or B12_sm1t.shape[0] == 0:
            B12_sm1t = np.zeros((max(simplices_at_time(t)[q]-simplices_at_time(sm1)[q],1), 1))

        B22_sm1t = boundary_matrices[q+1][simplices_at_time(sm1)[q]:simplices_at_time(t)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]
        if B22_sm1t.shape[0] == 0:
            # NOTE: max cannot be required as something happens between t and t-1!
            B22_sm1t = np.zeros((1,max(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1],1)))

        B22_sm1tm1 = boundary_matrices[q+1][simplices_at_time(sm1)[q]:simplices_at_time(tm1)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(tm1)[q+1]]
        if B22_sm1tm1.shape[0] == 0:
            B22_sm1tm1 = np.zeros((1,simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1]))
            # ker_B22_sm1tm1 = scipy.linalg.null_space(B22_sm1tm1)
    
    B12_st = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
    if B12_st.shape[1] == 0 or B12_st.shape[0] == 0:
        B12_st = np.zeros((max(simplices_at_time(t)[q]-simplices_at_time(s)[q],1), 1))

    B22_st = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
    if B22_st.shape[0] == 0:
        B22_st = np.zeros((1,simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1]))

    B22_stm1 = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(tm1)[q], simplices_at_time(s)[q+1]:simplices_at_time(tm1)[q+1]]
    if B22_stm1.shape[0] == 0:
        B22_stm1 = np.zeros((1,simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]))
    
    if verb:
        print(f"B12_st:\n{B12_st}")
        print(f"B22_st:\n{B22_st}")
        print(f"B22_stm1:\n{B22_stm1}")
        if s_i > 0:
            print(f"B22_sm1t:\n{B22_sm1t}")
            print(f"B22_sm1tm1:\n{B22_sm1tm1}")
    # print("ker_B22_stm1:", ker_B22_stm1.shape)

    if s_i > 0:
        ker_B22_sm1t = scipy.linalg.null_space(B22_sm1t)
        if ker_B22_sm1t.shape[1] == 0:
            ker_B22_sm1t = np.zeros((ker_B22_sm1t.shape[0],1))
        if verb:
            print(f"ker_B22_sm1t:\n{ker_B22_sm1t}")

        Projector = np.eye(B22_sm1t.shape[1])
        Projector[:B22_sm1tm1.shape[1], :B22_sm1tm1.shape[1]] = np.linalg.pinv(B22_sm1tm1)@B22_sm1tm1
        if verb:
            print(f"Projector sm1tm1:\n{Projector}")
        projection_sm1 = scipy.linalg.orth(np.round(Projector@ker_B22_sm1t, 15)).T
        if verb:
            print(f"initial projection:\n{projection_sm1}")
            print(f"norms initial projection:\n{np.linalg.norm(projection_sm1, axis=0)}")

        # part_sm1 = B12_sm1t@projection_sm1
        # if verb:
        #     print(f"part_sm1:\n{part_sm1}")
        Projector_sm1 = np.eye(simplices_at_time(t)[q+1]- simplices_at_time(sm1)[q+1]) - np.linalg.pinv(projection_sm1)@(projection_sm1)
        Projector_sm1 = Projector_sm1[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):]
    else:
        Projector_sm1 = np.eye(B22_st.shape[1])
    if verb:
        print(f"Projector_sm1:\n{Projector_sm1}")
    

    ker_B22_st = scipy.linalg.null_space(B22_st)
    if ker_B22_st.shape[1] == 0:
        ker_B22_st = np.zeros((ker_B22_st.shape[0],1))
        return np.zeros((simplices_at_time(s)[q],simplices_at_time(s)[q]))
    if verb:
        print(f"ker_B22_st:\n{ker_B22_st}")

    Projector = np.eye(B22_st.shape[1])
    Projector[:B22_stm1.shape[1], :B22_stm1.shape[1]] = np.linalg.pinv(B22_stm1)@B22_stm1
    if verb:
        print(f"Projector no identity:\n{np.linalg.pinv(B22_stm1)@B22_stm1}")
        print(f"Projector:\n{Projector}")

    projection_s = Projector@ker_B22_st
    # part_s = B12_st@projection_s
    if verb:
        print(f"projection_s:\n{projection_s}")
    if verb and s_i>0:
        print("Part of projector:", (simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]))
    # V = scipy.linalg.orth(np.round(Projector_sm1[(Projector_sm1.shape[0]-ker_B22_st.shape[0]):,(Projector_sm1.shape[0]-ker_B22_st.shape[0]):]@projection_s,15))
    V = scipy.linalg.orth(np.round(Projector_sm1@projection_s,15))
    if verb:
        print(f"V:\n{V}")
    V = V@(V.T)
    
    return B12_st@V@(B12_st.T)

def cross_Laplacian_cor10_eigenvalues(f: d.Filtration, weight_fun, max_dim = 1):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)), leave=False)
        for t_i in t_i_bar:
            for s_i in range(t_i):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                try:
                    Lap = cross_Laplacian_cor10(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times=relevant_times)
                except:
                    print(s,t)
                    Lap = cross_Laplacian_cor10(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times=relevant_times, verb=True)
                    raise ValueError
                
                eigenvalues[q][s][t] = np.linalg.eigvalsh(Lap)
    return eigenvalues, relevant_times
    

def calc_cor10(f: d.Filtration, q, s, t, weight_fun = lambda x: 1, verb=False):
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    relevant_times = np.array(relevant_times)
    t_i = np.argmin(np.abs(relevant_times - t))
    s_i = np.argmin(np.abs(relevant_times - s))
    return cross_Laplacian_cor10(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=verb)

def plot_Laplacian_cor10_eigenvalues(f: d.Filtration, weight_fun, max_dim = 1, plot_types = "all", 
                     plot_args_mesh = {}, 
                     plot_args_diag = {},
                     plot_args_line = {},
                     plot_type_to_fun = {}):
    """
    lapalcian_type: "persistent" for normal laplacian, or "cross" for cross laplacian, "cross_cor10" for cross Laplacian in two directions.
    """
    eigenvalues, relevant_times = cross_Laplacian_cor10_eigenvalues(f, weight_fun, max_dim=max_dim)

    fig, ax = plot_eigenvalues(eigenvalues, relevant_times, plot_types=plot_types, filtration=f,
                               plot_args_mesh = plot_args_mesh, plot_args_diag=plot_args_diag,
                               plot_args_line=plot_args_line, plot_type_to_fun=plot_type_to_fun)
    return eigenvalues, relevant_times, fig, ax

