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
from Laplacian_Functions import persistent_Laplacian_filtration

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

def cross_Laplacian(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=False, Laplacian_fun = None, device="cpu"):
    t, s = relevant_times[t_i], relevant_times[s_i]
    if t_i == 0:
        return torch.from_numpy(persistent_Laplacian_filtration(q, [b.numpy() for b in boundary_matrices], s, t, simplices_at_time, verb=verb)).to(device)
    tm1 = relevant_times[t_i-1]

    if verb:
        print(f"Bqplus1:\n{boundary_matrices[q+1]}")
        print(f"n_q_t:{simplices_at_time(t)}, n_q_tm1:{simplices_at_time(tm1)}, n_q_s: {simplices_at_time(s)}")
    if s_i > 0:
        sm1 = relevant_times[s_i-1]

        B22_sm1t = boundary_matrices[q+1][simplices_at_time(sm1)[q]:simplices_at_time(t)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]

        B22_sm1tm1 = boundary_matrices[q+1][simplices_at_time(sm1)[q]:simplices_at_time(tm1)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(tm1)[q+1]]
        
        A_matrix = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]

    B12_st = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]

    B22_st = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]

    B22_stm1 = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(tm1)[q], simplices_at_time(s)[q+1]:simplices_at_time(tm1)[q+1]]
    
    if verb:
        print(f"B12_st:\n{B12_st}")
        print(f"B22_st:\n{B22_st}")
        print(f"B22_stm1:\n{B22_stm1}")
        if s_i > 0:
            print(f"B22_sm1t:\n{B22_sm1t}")
            print(f"B22_sm1tm1:\n{B22_sm1tm1}")
    
    if s_i > 0:
        eye_sm1t = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1], device = device)
        B22_sm1t_full = torch.linalg.pinv(B22_sm1t)@B22_sm1t

        B22 = torch.linalg.pinv(B22_st)
        if verb:
            print(f"B22_st^dagger:{(B22<1e-13).sum()}/{B22.shape[0]*B22.shape[1]}={(B22<1e-13).sum()/(B22.shape[0]*B22.shape[1])}")
        # return
        B22_st_full = torch.zeros_like(eye_sm1t, device = device)
        B22_st_full[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = B22@B22_st

        B22_stm1_full = torch.zeros_like(eye_sm1t, device = device)
        B22_stm1_partial_full = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], device = device)
        B22_stm1_partial_full[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = torch.linalg.pinv(B22_stm1)@B22_stm1
        B22_stm1_full[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = B22_stm1_partial_full

        B22_sm1tm1_full = deepcopy(eye_sm1t)
        B22_sm1tm1_full[:(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1])] = torch.linalg.pinv(B22_sm1tm1)@B22_sm1tm1

        if verb:
            print(f"A_matrix:\n{A_matrix}")
            print(f"B22_st_full:\n{torch.round(B22_st_full,decimals=5)}")
            print(f"B22_stm1_full:\n{torch.round(B22_stm1_full,decimals=5)}")
            print(f"B22_sm1t_full:\n{torch.round(B22_sm1t_full,decimals=5)}")
            print(f"B22_sm1tm1_full:\n{torch.round(B22_sm1tm1_full,decimals=5)}")
            print(f"B22_stm1_full@B22_sm1t_full:\n{torch.round(B22_stm1_full@B22_sm1t_full,decimals=5)}")
            print(f"A_matrix@B22_stm1_full@B22_sm1t_full@A_matrix^T:\n{torch.round(A_matrix@(B22_stm1_full@B22_sm1t_full)@A_matrix.T, decimals=5)}")
            print(f"A_matrix@B22_sm1t_full@B22_stm1_full@A_matrix^T:\n{torch.round(A_matrix@(B22_sm1t_full@B22_stm1_full)@A_matrix.T, decimals=5)}")

        if verb:
            # VS = scipy.linalg.null_space(B22_st_full)
            # print(f"VS ({VS.shape}):\n{VS}")
            # G = VS.T@QR_matrix.T@QR_matrix@VS
            # print(f"G:\n{torch.round(G, decimals=5)}")
            # P1 = VS.T@B22_stm1_full@VS
            # P2 = VS.T@B22_sm1t_full@VS

            # Test_m1 = G@P1@P2
            Test_m1 = torch.round(B22_stm1_full@(eye_sm1t-B22_st_full), decimals=6)
            print(f"Test Matrix 1:\n{torch.round(Test_m1, decimals=4)}")
            evals, evecs = torch.linalg.eigh(torch.round(Test_m1, decimals=4))
            ker_B22 = []
            # for i, eval in enumerate(evals.real):
            #     if eval > 1e-8:
            #         ker_B22.append(evecs[:,i].real)
            #     # if torch.abs(eval) > 1e-8:
            #     print("eval:", eval)
            #     evec = evecs[:,i].real
            #     print("normalized evec", torch.round(evec, decimals=4))
            #     print("denormalized evec",torch.round(evec/torch.min(torch.abs(evec[torch.abs(evec) > 1e-8])),decimals=4))
            #     print()
            
            # for v in ker_B22:
            #     print("v:", torch.round(v,decimals=4))
            #     print("Av:", torch.round(A_matrix@v,decimals=4))
            #     print("AP^s-1,tv:", torch.round(A_matrix@B22_sm1t_full@v,decimals=4))
            #     print()
                # print("ori calc:", torch.linalg.norm(QR_matrix@B22_sm1tm1_full@v)**2 - torch.linalg.norm(QR_matrix@B22_sm1t_full@v)**2 - torch.linalg.norm(QR_matrix@B22_stm1_full@v)**2)
                # print("new calc:", (QR_matrix@B22_stm1_full@v).T@(QR_matrix@B22_sm1t_full@v))
                # print("newest calc:",0.5*(torch.linalg.norm(QR_matrix@(B22_sm1t_full + B22_stm1_full)@v)**2 - torch.linalg.norm(QR_matrix@B22_sm1t_full@v)**2 - torch.linalg.norm(QR_matrix@B22_stm1_full@v)**2))
                # print("final calc:",0.5*(-1*torch.linalg.norm(QR_matrix@(B22_sm1t_full - B22_stm1_full)@v)**2 + torch.linalg.norm(QR_matrix@B22_sm1t_full@v)**2 + torch.linalg.norm(QR_matrix@B22_stm1_full@v)**2))

                # print("left side:", torch.round(torch.linalg.norm(QR_matrix@B22_sm1tm1_full@v)**2 - 0.5*torch.linalg.norm(QR_matrix@(B22_sm1t_full - B22_stm1_full)@v)**2, decimals=5))
                # print("right side:", torch.round(0.5*(torch.linalg.norm(QR_matrix@B22_sm1t_full@v)**2 + torch.linalg.norm(QR_matrix@B22_stm1_full@v)**2), decimals=5))
                # print("sm1tm1:", torch.round(torch.linalg.norm(QR_matrix@B22_sm1tm1_full@v)**2, decimals=5))
                # print("sm1t:", torch.round(torch.linalg.norm(QR_matrix@B22_sm1t_full@v)**2, decimals=5))
                # print("stm1:", torch.round(torch.linalg.norm(QR_matrix@B22_stm1_full@v)**2, decimals=5))
                # print("sm1 - tm1:", torch.round(0.5*torch.linalg.norm(QR_matrix@(B22_sm1t_full - B22_stm1_full)@v)**2, decimals=5))
                # print("sm1 + tm1:", torch.round(0.5*torch.linalg.norm(QR_matrix@(B22_sm1t_full + B22_stm1_full)@v)**2, decimals=5))
            # Test_m2 = B22_sm1t_full@(eye_sm1t-B22_st_full)@B22_sm1t_full@B22_stm1_full
            # print(f"Test Matrix 2:\n{torch.round(Test_m2, decimals=5)}")
            # evals, evecs = torch.linalg.eig(torch.round(Test_m2, decimals=5))
            # for i, eval in enumerate(evals):
            #     # if torch.abs(eval) > 1e-6:
            #     print("eval:", eval)
            #     evec = evecs[:,i]
            #     print("normalized evec", torch.round(evec, decimals=5))
            #     print("denormalized evec",torch.round(evec/torch.min(torch.abs(evec[torch.abs(evec) > 1e-6])),decimals=5))
            #     print()

            # Test_m3 = B22_stm1_full@B22_sm1t_full@(eye_sm1t-B22_st_full)@B22_sm1t_full
            # print(f"Test Matrix:\n{torch.round(Test_m3, decimals=5)}")
            # evals, evecs = torch.linalg.eig(torch.round(Test_m3, decimals=5))
            # for i, eval in enumerate(evals):
            #     # if torch.abs(eval) > 1e-6:
            #     print("eval:", eval)
            #     evec = evecs[:,i]
            #     print("normalized evec", torch.round(evec, decimals=5))
            #     print("denormalized evec",torch.round(evec/torch.min(torch.abs(evec[torch.abs(evec) > 1e-6])),decimals=5))
            #     print()



        if Laplacian_fun is not None:
            return A_matrix@Laplacian_fun(B22_st_full, B22_stm1_full, B22_sm1t_full, B22_sm1tm1_full, eye_sm1t)@A_matrix.T
        return A_matrix@(B22_sm1t_full@B22_stm1_full@(eye_sm1t-B22_st_full)@B22_stm1_full)@A_matrix.T

    eye_st = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], device = device)

    B22_st_full = torch.linalg.pinv(B22_st)@B22_st

    B22_stm1_full = deepcopy(eye_st)
    B22_stm1_full[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = torch.linalg.pinv(B22_stm1)@B22_stm1

    return B12_st@(eye_st-B22_st_full)@B22_stm1_full@B12_st.T

def calc_cross(f: dio.Filtration, q, s, t, weight_fun = lambda x: 1, verb=False, Laplacian_fun = None, device = "cpu"):
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun, device=device)
    # relevant_times = torch.tensor(relevant_times, dtype=torch.float64, device = device)
    t_i = np.argmin(np.abs(relevant_times - t))
    s_i = np.argmin(np.abs(relevant_times - s))
    return cross_Laplacian(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=verb, Laplacian_fun= Laplacian_fun, device=device)

def calc_extended_up_Laplacian(f, q, s, t, weight_fun=lambda _ : 1, verb=False, device = "cpu"):
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun, device=device)
    t_i = np.argmin(np.abs(relevant_times - t))
    s_i = np.argmin(np.abs(relevant_times - s))
    if s_i == 0:
        return 0

    sm1 = relevant_times[s_i-1]
    tm1 = relevant_times[t_i-1]
    
    B_sm1 = boundary_matrices[q+1][:simplices_at_time(sm1)[q], :simplices_at_time(sm1)[q+1]]
    up_Lap_sm1 = B_sm1@B_sm1.T

    if verb:
        print(f"up_Lap_sm1:\n{up_Lap_sm1}")

    B22_sm1t = boundary_matrices[q+1][simplices_at_time(sm1)[q]:simplices_at_time(t)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]
    A_matrix = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]

    B22_sm1t_full = torch.linalg.pinv(B22_sm1t)@B22_sm1t

    second_part = A_matrix@(torch.eye(B22_sm1t_full.shape[0], device=device) - B22_sm1t_full)@A_matrix.T
    if verb:
        print(f"second_part:\n{second_part}")
    first_part = torch.zeros_like(second_part, device=device)
    first_part[:simplices_at_time(sm1)[q], :simplices_at_time(sm1)[q]] = up_Lap_sm1
    if verb:
        print(f"first_part:\n{first_part}")

    return first_part + second_part



def cross_Laplacian_new_eigenvalues(f: dio.Filtration, weight_fun, max_dim = 1, Laplacian_fun = None, device = "cpu"):
    f.sort()

    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun, device=device)
    eigenvalues = {q: {s: {t: torch.tensor([], dtype=torch.float64, device=device) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)), leave=False)
        for t_i in t_i_bar:
            s_i_bar = range(t_i)
            for s_i in s_i_bar:
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                try:
                    Lap = cross_Laplacian(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times=relevant_times, Laplacian_fun = Laplacian_fun, device=device)
                except:
                    print(s,t)
                    Lap = cross_Laplacian(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times=relevant_times, Laplacian_fun = Laplacian_fun, device = device, verb=True)
                    raise ValueError
                
                eigenvalues[q][s][t] = torch.linalg.eigvals(Lap).cpu().numpy().real
                # if torch.any(eigenvalues[q][s][t].real != eigenvalues[q][s][t]):
                #     print(f"q: {q}, s: {s}, t: {t}, evals: {eigenvalues[q][s][t]}, Lap:\n{Lap}")
    return eigenvalues, relevant_times

def cross_Laplaican_eigenvalues_less_memory(f: dio.Filtration, weight_fun = lambda x: 1, min_dim = 0, max_dim = 1, 
                                            Laplacian_fun = None, device = "cpu",
                                            compute_only_trace = False):
    f.sort()
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun, device=device)
    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(min_dim, max_dim+1)}
    # eigenvalues = {q: {s: {relevant_times[0]: []} if s==relevant_times[0] else {} for s in relevant_times} for q in range(max_dim+1)}
    # projection_matrices = {q: {s: {t: torch.tensor([], dtype=torch.float64) for t in range(len(relevant_times))} for s in  range(len(relevant_times))} for q in range(max_dim+1)}
    
    if Laplacian_fun is None:
        Laplacian_fun = lambda B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye: B22_sm1t@B22_stm1@(eye-B22_st)@B22_stm1@B22_sm1t
    
    for q in range(min_dim, max_dim+1):
        projection_matrices = {"sm1": {t: None for t in range(len(relevant_times))}, "s": {t: None for t in range(len(relevant_times))}}
        s_i_bar = tqdm(range(len(relevant_times)-1), leave=False)
        pinv_total_time = 0
        greville_total_time = 0
        lap_total_time = 0
        eig_total_time = 0
        save_total_time = 0

        for s_i in s_i_bar:
            s = relevant_times[s_i]

            # NOTE: if n_q^K==n_q^L, then the persistent up-laplacian is just the combinatorial up-laplacian in L. Therefore, we can let B22 be 0.
            # NOTE: if n_q+1^K==n_q+1^L, then the persistent up-laplacian is just the combinatorial up-laplacian in K. Therefore, we can let B22 be the identity.
            obtained_B22 = False
            for t_i in range(s_i, len(relevant_times)):
                s_i_bar.set_description(f"t_i: {t_i}/{len(relevant_times)}({int(obtained_B22)}), pinv: {pinv_total_time:.2f}, lap: {lap_total_time:.2f}, grev: {greville_total_time:.2f}, eig: {eig_total_time:.2f}, save: {save_total_time:.2f}")
                t = relevant_times[t_i]
                # tm1 = relevant_times[t_i-1]

                # Get B22^(-1)@B22
                stime = time.time()
                if not obtained_B22:
                    if simplices_at_time(t)[q+1] == simplices_at_time(s)[q+1]:
                        cur_B22 = torch.zeros((0,0), device = device)
                    elif simplices_at_time(t)[q] == simplices_at_time(s)[q]:
                        cur_B22 = 0#torch.zeros((simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1]), device = device)

                        # Think this is okay
                        # eigenvalues[q][s][t] = np.array([0])
                        # continue
                    else:
                        A_km1 = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
                        A_km1_pinv = torch.linalg.pinv(A_km1)
                        cur_B22 = A_km1_pinv@A_km1
                        t_A_km1 = t
                        obtained_B22 = True
                else:
                    # changed_B22 = False
                    # if simplices_at_time(t)[q] > simplices_at_time(t_A_km1)[q]:
                    #     # NOTE: Not necessary, can also just use boundary matrices to get it. But this makes it more clear.
                    #     # A_km1_new = torch.zeros((simplices_at_time(t)[q]-simplices_at_time(s)[q], simplices_at_time(t_A_km1)[q+1]-simplices_at_time(s)[q+1]), device = device)
                    #     # A_km1_new[:A_km1.shape[0], :A_km1.shape[1]] = A_km1
                    #     # A_km1 = A_km1_new

                    #     A_km1 = F.pad(A_km1, (0, 0, 0, simplices_at_time(t)[q]-simplices_at_time(t_A_km1)[q]), "constant", 0)
                        

                    #     # A_km1_pinv_new = torch.zeros((simplices_at_time(t_A_km1)[q+1]-simplices_at_time(s)[q+1], simplices_at_time(t)[q]-simplices_at_time(s)[q]), device = device)
                    #     # A_km1_pinv_new[:A_km1_pinv.shape[0], :A_km1_pinv.shape[1]] = A_km1_pinv
                    #     # A_km1_pinv = A_km1_pinv_new

                    #     A_km1_pinv = F.pad(A_km1_pinv, (0, simplices_at_time(t)[q]-simplices_at_time(t_A_km1)[q], 0, 0), "constant", 0)

                    #     changed_B22 = True
                    
                    if simplices_at_time(t)[q+1] > simplices_at_time(t_A_km1)[q+1]:
                        # Use the update rule from Greville.

                        if simplices_at_time(t)[q] > simplices_at_time(t_A_km1)[q]:
                            A_km1 = F.pad(A_km1, (0, 0, 0, simplices_at_time(t)[q]-simplices_at_time(t_A_km1)[q]), "constant", 0)
                            A_km1_pinv = F.pad(A_km1_pinv, (0, simplices_at_time(t)[q]-simplices_at_time(t_A_km1)[q], 0, 0), "constant", 0)
                        
                        stime2 = time.time()
                        for col in range(simplices_at_time(t_A_km1)[q+1], simplices_at_time(t)[q+1]):
                            a_k = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], [col]]

                            d_k = A_km1_pinv@a_k
                            c_k = a_k - A_km1@d_k
                            
                            c_k_norm = torch.linalg.norm(c_k)
                            if c_k_norm > 1e-8:
                                b_k = torch.linalg.pinv(c_k)
                                # b_k = c_k.T/c_k_norm
                            else:
                                b_k = (1 + d_k.T@d_k)**(-1)*d_k.T@A_km1_pinv
                                # print("c_k=0 norm c_k:", torch.linalg.norm(c_k))
                            
                            pinv_top_part = A_km1_pinv-d_k@b_k

                            
                            # This instead of the full matrix multiplication
                            added_row = b_k@A_km1
                            b_ka_k = (b_k@a_k)
                            added_row = (d_k*(1-b_ka_k))
                            cur_B22 = cur_B22 - d_k@(added_row.T)
                            cur_B22 = torch.block_diag(cur_B22, b_ka_k)

                            cur_B22[[-1], :-1] = added_row.T
                            cur_B22[:-1, [-1]] = added_row
                            # -------------------------------------------------
                            # Also instead of full matrix multiplication, but now removed cur_B22 in algorithm, which allows the whole computation to be written as a subtraction.
                            # Is a bit slower, than above method though.
                            # b_ka_k = (b_k@a_k).item()
                            # d_k_ext = torch.vstack([d_k, torch.tensor([[-1]])])
                            # sub_matrix = (1-b_ka_k)*d_k_ext@(d_k_ext.T)
                            
                            # sub_matrix[-1, -1] = -1*b_ka_k
                            # cur_B22 = F.pad(cur_B22, (0,1,0,1)) - sub_matrix
                            # -------------------------------------------------


                            A_km1_pinv = torch.vstack((pinv_top_part, b_k))
                            A_km1 = torch.hstack((A_km1, a_k))



                        t_A_km1 = t
                        # cur_B22 = A_km1_pinv@A_km1

                        greville_total_time += time.time() - stime2
                    
                    # Else matrix stays the same, however A_km1 can still differ. Don't think we need to update howerer as the updated A_km1 is only used when n_q+1^t>N_q+1^t_A_km1.
                pinv_total_time += time.time() - stime
                stime = time.time()


                # Calculate Laplacian and eigenvalues
                if s_i != t_i and t_i > 0:
                    tm1 = relevant_times[t_i-1]

                    # Otherwsie, we now that there are only 0 eigenvalues
                    if simplices_at_time(t)[q+1] > simplices_at_time(tm1)[q+1]:
                        if s_i > 0:
                            sm1 = relevant_times[s_i-1]

                            # Otherwsie, we now that there are only 0 eigenvalues
                            if simplices_at_time(s)[q] > simplices_at_time(sm1)[q]:
                                A_matrix = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]

                                eye = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1], device = device)
                                B22_sm1t = projection_matrices["sm1"][t_i]

                                B22_st = torch.zeros_like(eye, device = device)
                                B22_st[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = cur_B22

                                B22_stm1 = torch.zeros_like(eye, device = device)
                                B22_stm1_partial = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], device = device)
                                B22_stm1_partial[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = projection_matrices["s"][t_i-1]
                                B22_stm1[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = B22_stm1_partial

                                B22_sm1tm1 = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1], device = device)
                                B22_sm1tm1[:(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1])] = projection_matrices["sm1"][t_i-1]

                                cross_Lap = A_matrix@Laplacian_fun(B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye)@A_matrix.T

                                lap_total_time += time.time() - stime
                                stime = time.time()
                                # cross_Lap = Laplacian_fun(B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye)
                                if not compute_only_trace:
                                    eigenvalues[q][s][t] = torch.linalg.eigvals(cross_Lap).cpu().numpy().real
                                else:
                                    eigenvalues[q][s][t] = torch.trace(cross_Lap).cpu().numpy().reshape((1))
                                eig_total_time += time.time() - stime

                            else:
                                lap_total_time += time.time() - stime
                                stime = time.time()

                                # eigenvalues[q][s][t] = np.array([0])
                                eig_total_time += time.time() - stime
                        
                        else:
                            B12_st = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]

                            eye = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], device = device)

                            B22_st = cur_B22

                            B22_stm1 = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], device = device)
                            B22_stm1[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = projection_matrices["sm1"][t_i-1]

                            cross_Lap = B12_st@(B22_stm1 - B22_st)@B12_st.T

                            lap_total_time += time.time() - stime
                            stime = time.time()

                            if not compute_only_trace:
                                eigenvalues[q][s][t] = torch.linalg.eigvals(cross_Lap).cpu().numpy().real
                            else:
                                eigenvalues[q][s][t] = torch.trace(cross_Lap).cpu().numpy().reshape((1))
                            eig_total_time += time.time() - stime
                    else:
                        lap_total_time += time.time() - stime
                        stime = time.time()
                        
                        # eigenvalues[q][s][t] = np.array([0])

                        eig_total_time += time.time() - stime
                    
                stime = time.time()
                # Saving and removing the right matrices
                if s_i > 0:
                    projection_matrices["s"][t_i] = cur_B22
                    if t_i != s_i:
                        projection_matrices["sm1"][t_i-1] = projection_matrices["s"][t_i-1]
                    else:
                        projection_matrices["sm1"][t_i-1] = None
                    projection_matrices["s"][t_i-1] = None

                    
                else:
                    projection_matrices["sm1"][t_i] = cur_B22

                if t_i == len(relevant_times)-1:
                    projection_matrices["sm1"][t_i] = cur_B22


                save_total_time += time.time() - stime

    return eigenvalues, relevant_times



def cross_Laplaican_eigenvalues_fast(f: dio.Filtration, weight_fun = lambda x: 1, max_dim = 1, Laplacian_fun = None, device = "cpu", use_greville = False):
    f.sort()
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun, device=device)
    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    projection_matrices = {q: {s: {t: torch.tensor([], dtype=torch.float64) for t in range(len(relevant_times))} for s in  range(len(relevant_times))} for q in range(max_dim+1)}

    if Laplacian_fun is None:
        Laplacian_fun = lambda B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye: B22_sm1t@B22_stm1@(eye-B22_st)@B22_stm1@B22_sm1t
    
    print("Calculating projection matrices...")
    # Normal way
    if not use_greville:
        for q in range(max_dim+1):
            t_i_bar = tqdm(range(len(relevant_times)), leave=False)
            for t_i in t_i_bar:
                for s_i in range(t_i+1):
                    t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                    s, t = relevant_times[s_i], relevant_times[t_i]
                    B22 = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
                    
                    try:
                        projection_matrices[q][s_i][t_i] = torch.linalg.pinv(B22)@B22
                    except LinAlgError:
                        print("RAN INTO ERROR !!!")
                        print(f"s: {s}, t: {t}")

                        projection_matrices[q][s_i][t_i] = torch.eye(simplices_at_time(t)[q+1] - simplices_at_time(s)[q+1], device = device)
                        # np.savetxt("Error.txt", B22)
                        # return

    # Greville way
    else:
        for q in range(max_dim+1):
            
            s_i_bar = tqdm(range(len(relevant_times)), leave=False)
            for s_i in s_i_bar:
                s = relevant_times[s_i]

                # NOTE: if n_q^K==n_q^L, then the persistent up-laplacian is just the combinatorial up-laplacian in L. Therefore, we can let B22 be 0.
                # NOTE: if n_q+1^K==n_q+1^L, then the persistent up-laplacian is just the combinatorial up-laplacian in K. Therefore, we can let B22 be the identity.
                obtained_B22 = False
                for t_i in range(s_i, len(relevant_times)):
                    s_i_bar.set_description(f"t_i: {t_i}/{len(relevant_times)}")
                    t = relevant_times[t_i]
                    # tm1 = relevant_times[t_i-1]
                    
                    if not obtained_B22:
                        if simplices_at_time(t)[q+1] == simplices_at_time(s)[q+1]:
                            projection_matrices[q][s_i][t_i] = torch.zeros((0,0), device = device)
                        elif simplices_at_time(t)[q] == simplices_at_time(s)[q]:
                            projection_matrices[q][s_i][t_i] = torch.zeros((simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1]), device = device)
                        else:
                            A_km1 = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
                            A_km1_pinv = torch.linalg.pinv(A_km1)
                            projection_matrices[q][s_i][t_i] = A_km1_pinv@A_km1
                            t_A_km1 = t
                            obtained_B22 = True
                    else:
                        changed_B22 = False
                        if simplices_at_time(t)[q] > simplices_at_time(t_A_km1)[q]:
                            # NOTE: Not necessary, can also just use boundary matrices to get it. But this makes it more clear.
                            A_km1_new = torch.zeros((simplices_at_time(t)[q]-simplices_at_time(s)[q], simplices_at_time(t_A_km1)[q+1]-simplices_at_time(s)[q+1]), device = device)
                            A_km1_new[:A_km1.shape[0], :A_km1.shape[1]] = A_km1
                            A_km1 = A_km1_new

                            A_km1_pinv_new = torch.zeros((simplices_at_time(t_A_km1)[q+1]-simplices_at_time(s)[q+1], simplices_at_time(t)[q]-simplices_at_time(s)[q]), device = device)
                            A_km1_pinv_new[:A_km1_pinv.shape[0], :A_km1_pinv.shape[1]] = A_km1_pinv
                            A_km1_pinv = A_km1_pinv_new

                            changed_B22 = True
                        
                        if simplices_at_time(t)[q+1] > simplices_at_time(t_A_km1)[q+1]:
                            # Use the update rule from Greville.
                            

                            for col in range(simplices_at_time(t_A_km1)[q+1], simplices_at_time(t)[q+1]):
                                a_k = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], [col]]

                                d_k = A_km1_pinv@a_k
                                c_k = a_k - A_km1@d_k

                                if torch.linalg.norm(c_k) > 1e-8:
                                    b_k = torch.linalg.pinv(c_k)
                                else:
                                    b_k = (1 + d_k.T@d_k)**(-1)*d_k.T@A_km1_pinv
                                    # print("c_k=0 norm c_k:", torch.linalg.norm(c_k))
                                
                                A_km1_pinv = torch.vstack((A_km1_pinv-d_k@b_k, b_k))
                                A_km1 = torch.hstack((A_km1, a_k))
                            changed_B22 = True
                            
                        
                        if changed_B22:
                            t_A_km1 = t
                            projection_matrices[q][s_i][t_i] = A_km1_pinv@A_km1
                        else:
                            projection_matrices[q][s_i][t_i] = projection_matrices[q][s_i][t_i-1]

    print("Calculating Laplacians and eigenvalues...")
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(1, len(relevant_times)), leave=False)
        for t_i in t_i_bar:
            for s_i in range(t_i):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t, tm1 = relevant_times[s_i], relevant_times[t_i], relevant_times[t_i-1]

                if s_i > 0:
                    sm1 = relevant_times[s_i-1]

                    A_matrix = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]

                    eye = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1], device = device)
                    B22_sm1t = projection_matrices[q][s_i-1][t_i]

                    B22_st = torch.zeros_like(eye, device = device)
                    B22_st[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = projection_matrices[q][s_i][t_i]

                    B22_stm1 = torch.zeros_like(eye, device = device)
                    B22_stm1_partial = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], device = device)
                    B22_stm1_partial[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = projection_matrices[q][s_i][t_i-1]
                    B22_stm1[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = B22_stm1_partial

                    B22_sm1tm1 = deepcopy(eye)
                    B22_sm1tm1[:(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1])] = projection_matrices[q][s_i-1][t_i-1]

                    cross_Lap = A_matrix@Laplacian_fun(B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye)@A_matrix.T
                    # cross_Lap = Laplacian_fun(B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye)

                    eigenvalues[q][s][t] = torch.linalg.eigvals(cross_Lap).cpu().numpy().real
                
                else:
                    B12_st = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]

                    eye = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], device = device)

                    B22_st = projection_matrices[q][s_i][t_i]

                    B22_stm1 = deepcopy(eye)
                    B22_stm1[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = projection_matrices[q][s_i][t_i-1]

                    cross_Lap = B12_st@(B22_stm1 - B22_st)@B12_st.T

                    eigenvalues[q][s][t] = torch.linalg.eigvals(cross_Lap).cpu().numpy().real

    return eigenvalues, relevant_times

def eig_plot_helper(x, fun, use_pos = True, eps = 1e-10):
    if len(x) > 0:
        if not use_pos:
            return fun(x)
        else:
            pos_x = x[np.abs(x)>eps]
            if len(pos_x) > 0:
                return fun(pos_x)
            else:
                return 0
    else:
        return np.nan


def plot_eigenvalues(eigenvalues, relevant_times, plot_types = "all", filtration = None, integer_time_steps = False,
                     plot_args_mesh = {}, 
                     plot_args_diag = {},
                     plot_args_line = {},
                     plot_type_to_fun = {},
                     plot_value_on_diagonal = False):
    # Can choose these plot types
    plot_type_to_fun = {
        "Min eigenvalue": (np.min, True),
        "Max eigenvalue": (np.max, False),
        "Sum of eigenvalues": (np.sum, False),
        "Mean of eigenvalues": (np.mean, True),
        "Product of eigenvalues": (np.prod, True),
        "Gmean of eigenvalues": (ss.gmean, True)
    } | plot_type_to_fun

    plot_args_mesh = {"alpha": 1, "cmap": "jet"} | plot_args_mesh
    plot_args_diag = {"c": "black", "marker": ".", "alpha": 0.75} | plot_args_diag
    plot_args_line = {"c": "r", "alpha": 0.5} | plot_args_line

    max_dim = max(eigenvalues.keys())

    if plot_types == "all":
        plot_types = list(plot_type_to_fun.keys())

    fig, ax = plt.subplots(max_dim+1, len(plot_types))
    fig.set_size_inches(4*len(plot_types), 4*(max_dim+1))
    if len(plot_types) == 1:
        cur_ax = lambda x,y: ax[x]
    else:
        cur_ax = lambda x,y: ax[x, y]
        

    fig.gca()

    # For pcolormesh, we need a grid that is one bigger as every rectangle needs starting and ending points.
    # Now added a point the same distance away as the last two real points.
    relevant_times = relevant_times.tolist()
    extended_relevant_times = relevant_times+[2*relevant_times[-1]-relevant_times[-2]]
    x, y = np.meshgrid(extended_relevant_times, extended_relevant_times)

    if filtration is not None:
        p = dio.cohomology_persistence(filtration, 47, True)
        dgms = dio.init_diagrams(p, filtration)
        barcodes_births, barcodes_deaths = [], []
        for q in range(len(dgms)):
            barcodes_births.append([])
            barcodes_deaths.append([])
            for p in dgms[q]:
                barcodes_births[q].append(p.birth)
                if p.death != np.inf:
                    barcodes_deaths[q].append(p.death)
                else:
                    barcodes_deaths[q].append(extended_relevant_times[-1])

    for q in range(max_dim+1):
        df_evals = pd.DataFrame(eigenvalues[q])
        for ax_i, plot_type in enumerate(plot_types):
            if plot_type == "Product of eigenvalues":
                plot_args_mesh["norm"] = "log"
            else:
                plot_args_mesh["norm"] = "linear"
            plot_fun, use_pos = plot_type_to_fun[plot_type]
            im = cur_ax(q, ax_i).pcolormesh(x, y, df_evals.apply(lambda x: x.apply(lambda y: eig_plot_helper(y, plot_fun, use_pos=use_pos))).values, **plot_args_mesh)
            plt.colorbar(im, ax=cur_ax(q, ax_i), pad=0.15)
            if q == 0:
                cur_ax(q, ax_i).set_title(f"{plot_type}")
            if q == max_dim:
                cur_ax(q, ax_i).set_xlabel("s")
            if filtration is not None:
                cur_ax(q, ax_i).scatter(barcodes_births[q], barcodes_deaths[q], **plot_args_diag)

            # Plot diagonal
            if plot_value_on_diagonal:
                ax_n = cur_ax(q, ax_i).twinx()
                ax_n.plot([relevant_times[0]] + [val for val in relevant_times[1:] for _ in (0, 1)], [eig_plot_helper(eigenvalues[q][t][t], plot_fun, use_pos=use_pos) for t in relevant_times for _ in (0,1)][:-1], **plot_args_line)
                if plot_type == "Product of eigenvalues":
                    ax_n.set_yscale("log")

            if integer_time_steps:
                cur_ax(q, ax_i).yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        cur_ax(q, 0).set_ylabel(f"q={q}\n\nt")
    fig.tight_layout()
    return fig, ax


# def eig_plot_helper(x, fun, eps = 1e-8):
#     if len(x) > 0:
#         pos_x = x[np.abs(x)>eps]
#         if len(pos_x) > 0:
#             return fun(pos_x)
#         else:
#             return 0
#     else:
#         return np.nan

# def plot_eigenvalues(eigenvalues, relevant_times, plot_types = "all", filtration = None, integer_time_steps = False,
#                      plot_args_mesh = {}, 
#                      plot_args_diag = {},
#                      plot_args_line = {},
#                      plot_type_to_fun = {}):
#     # NOTE: eigenvalues should be a numpy array, not a torch tensor.
#     # Can choose these plot types
#     plot_type_to_fun = {
#         "Min": np.min,
#         "Max": np.max,
#         "Sum": np.sum,
#         "Mean": np.mean,
#         "Prod": np.prod,
#         "Gmean": ss.gmean
#     } | plot_type_to_fun

#     plot_args_mesh = {"alpha": 1, "cmap": "jet"} | plot_args_mesh
#     plot_args_diag = {"c": "black", "marker": ".", "alpha": 0.75} | plot_args_diag
#     plot_args_line = {"c": "r", "alpha": 0.5} | plot_args_line

#     max_dim = max(eigenvalues.keys())

#     if plot_types == "all":
#         plot_types = list(plot_type_to_fun.keys())

#     fig, ax = plt.subplots(max_dim+1, len(plot_types))
#     fig.set_size_inches(4*len(plot_types), 4*(max_dim+1))
#     if len(plot_types) == 1:
#         cur_ax = lambda x,y: ax[x]
#     else:
#         cur_ax = lambda x,y: ax[x, y]
        

#     fig.gca()

#     # For pcolormesh, we need a grid that is one bigger as every rectangle needs starting and ending points.
#     # Now added a point the same distance away as the last two real points.
#     relevant_times = relevant_times.tolist()
#     extended_relevant_times = relevant_times+[2*relevant_times[-1]-relevant_times[-2]]
#     x, y = np.meshgrid(extended_relevant_times, extended_relevant_times)

#     if filtration is not None:
#         p = dio.cohomology_persistence(filtration, 47, True)
#         dgms = dio.init_diagrams(p, filtration)
#         barcodes_births, barcodes_deaths = [], []
#         for q in range(len(dgms)):
#             barcodes_births.append([])
#             barcodes_deaths.append([])
#             for p in dgms[q]:
#                 barcodes_births[q].append(p.birth)
#                 if p.death != np.inf:
#                     barcodes_deaths[q].append(p.death)
#                 else:
#                     barcodes_deaths[q].append(extended_relevant_times[-1])

#     for q in range(max_dim+1):
#         df_evals = pd.DataFrame(eigenvalues[q])
#         for ax_i, plot_type in enumerate(plot_types):
#             if plot_type == "Prod":
#                 plot_args_mesh["norm"] = "log"
#             else:
#                 plot_args_mesh["norm"] = "linear"
#             im = cur_ax(q, ax_i).pcolormesh(x, y, df_evals.apply(lambda x: x.apply(lambda y: eig_plot_helper(y, plot_type_to_fun[plot_type]))).values, **plot_args_mesh)
#             plt.colorbar(im, ax=cur_ax(q, ax_i), pad=0.15)
#             if q == 0:
#                 cur_ax(q, ax_i).set_title(f"{plot_type} eigenvalue")
#             if filtration is not None:
#                 cur_ax(q, ax_i).scatter(barcodes_births[q], barcodes_deaths[q], **plot_args_diag)

#             ax_n = cur_ax(q, ax_i).twinx()
#             ax_n.plot([relevant_times[0]] + [val for val in relevant_times[1:] for _ in (0, 1)], [eig_plot_helper(eigenvalues[q][t][t], plot_type_to_fun[plot_type]) for t in relevant_times for _ in (0,1)][:-1], **plot_args_line)
#             if plot_type == "Prod":
#                 ax_n.set_yscale("log")

#             if integer_time_steps:
#                 cur_ax(q, ax_i).yaxis.set_major_locator(plt.MaxNLocator(integer=True))
#         cur_ax(q, 0).set_ylabel(f"q={q}")
#     fig.tight_layout()
#     return fig, ax


def plot_trace_diagram(f, eigenvalues, show=False, min_dim=0, max_dim=1,
                 line_style=None, pt_style_normal=None, lap_pt_style=None,
                 limits = None, title = "Persistence diagram"):

    p = dio.cohomology_persistence(f, 47, True)
    dgms = dio.init_diagrams(p, f)
    fig, ax = plt.subplots(1, max_dim+1-min_dim, figsize=(6*(max_dim+1-min_dim),5))

    s_lists = [[]]*(max_dim + 1)
    t_lists = [[]]*(max_dim + 1)
    cmaps = [[]]*(max_dim + 1)

    for q in range(min_dim, max_dim+1):
        dgm = dgms[q]

        if max_dim+1-min_dim > 1:
            cur_ax = ax[q-min_dim]
        else:
            cur_ax = ax

        line_kwargs = {}
        pt_kwargs = {"c": "black", "marker": "x", "alpha": 0.5, "s": 100}
        if pt_style_normal is not None:
            pt_kwargs.update(pt_style_normal)
        if line_style is not None:
            line_kwargs.update(line_style)
        lap_pt_kwargs = {}
        if lap_pt_style is not None:
            lap_pt_kwargs.update(lap_pt_style)


        inf = float('inf')

        s_list = []
        t_list = []
        cmap = []
        for s in eigenvalues[q].keys():
            for t in eigenvalues[q][s].keys():
                eval_sum = np.sum(eigenvalues[q][s][t])
                if eval_sum > 1e-10:
                    # if eval_sum < 0.1:
                    #     print(s, t, eval_sum)
                    s_list.append(s)
                    t_list.append(t)
                    cmap.append(eval_sum)

        s_lists[q] = (s_list)
        t_lists[q] = (t_list)
        cmaps[q] = (cmap)

        if len(s_list) > 0:
            # min_diag = np.min(s_list)*0.9
            min_diag = 0
            max_diag = np.max(t_list)*1.1
        else:
            min_diag = -1
            max_diag = 2

        
        # min_birth,max_birth,min_death,max_death = inf,-1,inf,-1
        # if not limits:
        #     min_birth = min(min_birth, min((p.birth for p in dgm))*0.9)
        #     max_birth = max(max_birth, max((p.birth for p in dgm)))

        #     for p in dgm:
        #         if p.death != inf:
        #             max_death = max(max_death, max((p.death for p in dgm if p.death != inf))*1.1)
        #             break
        #     if max_death == -1:
        #         max_death = max_birth*1.1
        # else:
        #     min_birth, max_birth, min_death, max_death = limits


        cur_ax.set_aspect('equal', 'datalim')

        cur_ax.scatter([p.birth for p in dgm], [p.death if p.death != inf else max_diag for p in dgm], **pt_kwargs)

       

        scatter_im = cur_ax.scatter(s_list, t_list, c=cmap, **lap_pt_kwargs)
        plt.colorbar(scatter_im, ax=cur_ax, pad=0.15)

        # Line y=x
        cur_ax.plot([min_diag, max_diag], [min_diag, max_diag], **line_kwargs)
        # Line for infinity
        cur_ax.plot([min_diag, max_diag], [max_diag, max_diag], c="black")

        cur_ax.set_xlabel('birth')
        cur_ax.set_ylabel('death')
        cur_ax.set_title(f"{title} (dim {q})")

        ## clip the view
        #plt.axes().set_xlim([min_birth, max_birth])
        #plt.axes().set_ylim([min_death, max_death])

    if show:
        fig.show()
    return s_lists, t_lists, cmaps

def plot_trace_diagram_no_f(dgms, eigenvalues, show=False, min_dim=0, max_dim=1,
                 line_style=None, pt_style_normal=None, lap_pt_style=None,
                 limits = None, title="Persistence diagram"):

    fig, ax = plt.subplots(1, max_dim+1-min_dim, figsize=(6*(max_dim+1-min_dim),5))

    s_lists = [[]]*(max_dim + 1)
    t_lists = [[]]*(max_dim + 1)
    cmaps = [[]]*(max_dim + 1)

    for q in range(min_dim, max_dim+1):
        dgm = dgms[q]

        if max_dim+1-min_dim > 1:
            cur_ax = ax[q-min_dim]
        else:
            cur_ax = ax

        line_kwargs = {}
        pt_kwargs = {"c": "black", "marker": "x", "alpha": 0.5, "s": 100}
        if pt_style_normal is not None:
            pt_kwargs.update(pt_style_normal)
        if line_style is not None:
            line_kwargs.update(line_style)
        lap_pt_kwargs = {}  
        if lap_pt_style is not None:
            lap_pt_kwargs.update(lap_pt_style)


        inf = float('inf')

        s_list = []
        t_list = []
        cmap = []
        for s in eigenvalues[q].keys():
            for t in eigenvalues[q][s].keys():
                eval_sum = np.sum(eigenvalues[q][s][t])
                if eval_sum > 1e-10:
                    if eval_sum < 0.1:
                        print(s, t, eval_sum)
                    s_list.append(s)
                    t_list.append(t)
                    cmap.append(eval_sum)

        s_lists[q] = (s_list)
        t_lists[q] = (t_list)
        cmaps[q] = (cmap)

        if len(s_list) > 0:
            # min_diag = np.min(s_list)*0.9
            min_diag = 0
            max_diag = np.max(t_list)*1.1
        else:
            min_diag = -1
            max_diag = 2

        
        # min_birth,max_birth,min_death,max_death = inf,-1,inf,-1
        # if not limits:
        #     min_birth = min(min_birth, min((p.birth for p in dgm))*0.9)
        #     max_birth = max(max_birth, max((p.birth for p in dgm)))

        #     for p in dgm:
        #         if p.death != inf:
        #             max_death = max(max_death, max((p.death for p in dgm if p.death != inf))*1.1)
        #             break
        #     if max_death == -1:
        #         max_death = max_birth*1.1
        # else:
        #     min_birth, max_birth, min_death, max_death = limits


        cur_ax.set_aspect('equal', 'datalim')

        cur_ax.scatter([p.birth for p in dgm], [p.death if p.death != inf else max_diag for p in dgm], **pt_kwargs)

       

        scatter_im = cur_ax.scatter(s_list, t_list, c=cmap, **lap_pt_kwargs)
        plt.colorbar(scatter_im, ax=cur_ax, pad=0.15)

        # Line y=x
        cur_ax.plot([min_diag, max_diag], [min_diag, max_diag], **line_kwargs)
        # Line for infinity
        cur_ax.plot([min_diag, max_diag], [max_diag, max_diag], c="black")

        cur_ax.set_xlabel('birth')
        cur_ax.set_ylabel('death')
        cur_ax.set_title(f"{title} (dim {q})")

        ## clip the view
        #plt.axes().set_xlim([min_birth, max_birth])
        #plt.axes().set_ylim([min_death, max_death])

    if show:
        fig.show()
    return s_lists, t_lists, cmaps

def plot_trace_histogram(eigenvalues, show=False, max_dim=2,
                 hist_style=None):
    if show:
        fig, ax = plt.subplots(1, max_dim+1, figsize=(5*(max_dim+1),5))

    # s_lists = []
    # t_lists = []
    cmaps = []

    for q in range(max_dim+1):
        hist_kwargs = {"bins": 100}
        if hist_style is not None:
            hist_kwargs.update(hist_style)

        # s_list = []
        # t_list = []
        cmap = []
        for s in eigenvalues[q].keys():
            for t in eigenvalues[q][s].keys():
                if np.sum(eigenvalues[q][s][t]) > 1e-10:
                    # s_list.append(s)
                    # t_list.append(t)
                    cmap.append(np.sum(eigenvalues[q][s][t]))

        # s_lists.append(s_list)
        # t_lists.append(t_list)
        cmaps.append(cmap)

        if show:
            # ax[q].scatter(s_list, t_list, c=cmap)
            ax[q].hist(cmap, **hist_kwargs)

            # ax[q].set_xlabel('birth')
            # ax[q].set_ylabel('death')
            ax[q].set_title(f"Histogram of trace (dim {q})")

        ## clip the view
        #plt.axes().set_xlim([min_birth, max_birth])
        #plt.axes().set_ylim([min_death, max_death])

    if show:
        fig.show()
    return cmaps #s_lists, t_lists, 


def plot_Laplacian_new_eigenvalues(f: dio.Filtration, weight_fun, min_dim = 0, max_dim = 1, plot_types = "all", method = "slow", Laplacian_fun = None, integer_time_steps = False, device = "cpu",
                     plot_args_mesh = {}, 
                     plot_args_diag = {},
                     plot_args_line = {},
                     plot_type_to_fun = {},
                     lap_pt_style = {},
                     plot_diagram = False,
                     show_plot = True,
                     compute_only_trace = False,
                     plot_value_on_diagonal = False):
    """
    lapalcian_type: "persistent" for normal laplacian, or "cross" for cross laplacian, "cross_cor10" for cross Laplacian in two directions.
    """
    if method == "slow":
        eigenvalues, relevant_times = cross_Laplacian_new_eigenvalues(f, weight_fun=weight_fun, max_dim=max_dim, Laplacian_fun = Laplacian_fun, device=device)
    else:
        if method == "greville":
            eigenvalues, relevant_times = cross_Laplaican_eigenvalues_fast(f, weight_fun=weight_fun, max_dim=max_dim, Laplacian_fun = Laplacian_fun, device=device, use_greville=True)
        elif method == "less_memory":
            eigenvalues, relevant_times = cross_Laplaican_eigenvalues_less_memory(f, weight_fun=weight_fun, min_dim=min_dim, max_dim=max_dim, Laplacian_fun = Laplacian_fun, device=device, compute_only_trace=compute_only_trace)
        else:
            eigenvalues, relevant_times = cross_Laplaican_eigenvalues_fast(f, weight_fun=weight_fun, max_dim=max_dim, Laplacian_fun = Laplacian_fun, device=device)

    if plot_diagram:
        s_lists, t_lists, cmaps = plot_trace_diagram(f, eigenvalues, show=show_plot, min_dim=min_dim, max_dim=max_dim,
                     line_style=plot_args_line, pt_style_normal=plot_args_diag, lap_pt_style=lap_pt_style,
                     limits = (relevant_times[0], relevant_times[-1]*1.2, relevant_times[0], relevant_times[-1]*1.2), title="Laplacian diagram")
        return eigenvalues, relevant_times, s_lists, t_lists, cmaps

    # elif plot_histogram:
    #     cmaps = plot_trace_histogram(eigenvalues, show=show_plot, max_dim=max_dim,)
    
    else:
        if show_plot:
            fig, ax = plot_eigenvalues(eigenvalues, relevant_times, plot_types=plot_types, filtration=f, integer_time_steps=integer_time_steps,
                                plot_args_mesh = plot_args_mesh, plot_args_diag=plot_args_diag,
                                plot_args_line=plot_args_line, plot_type_to_fun=plot_type_to_fun,
                                plot_value_on_diagonal = plot_value_on_diagonal)
            return eigenvalues, relevant_times, fig, ax
        else:
            return eigenvalues, relevant_times