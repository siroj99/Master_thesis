from matplotlib.pylab import LinAlgError
import numpy as np
from copy import deepcopy
import dionysus as d
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.stats as ss
import pandas as pd
from sympy import use
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gudhi as gd

def pinv(A, sparse_type = "csr"):
    if sparse_type == "csr":
        return sp.csr_matrix(np.round(np.linalg.pinv(A.toarray()), 15))
    elif sparse_type == "csc":
        return sp.csc_matrix(np.round(np.linalg.pinv(A.toarray()), 15))
    elif sparse_type == "lil":
        return sp.lil_matrix(np.round(np.linalg.pinv(A.toarray()), 15))
    else:
        print(f"Sparse type {sparse_type} has not been implemented yet. Please use csr, csc or lil.")
        return None

def compute_boundary_matrices(f: d.Filtration, weight_fun, sparse_type = "csr"):
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
        i = 0
        if t == np.inf:
            return n_simplicies_seen_per_time[-1]
        while relevant_times[i] < t:
            i += 1
            if i >= len(relevant_times) - 1: # Changed this to >= from ==.
                return n_simplicies_seen_per_time[-1]
        return n_simplicies_seen_per_time[i]

    simplices_at_end = simplices_at_time(np.inf)

    # boundary_matrices = [0] + [torch.zeros((simplices_at_end[q-1], simplices_at_end[q])) for q in range(1, maxq)]
    if sparse_type == "csr":
        boundary_matrices = [sp.csr_matrix((simplices_at_end[max(0,q-1)], simplices_at_end[q])) for q in range(maxq)]
    elif sparse_type == "csc":
        boundary_matrices = [sp.csc_matrix((simplices_at_end[max(0,q-1)], simplices_at_end[q])) for q in range(maxq)]
    elif sparse_type == "lil":
        boundary_matrices = [sp.lil_matrix((simplices_at_end[max(0,q-1)], simplices_at_end[q])) for q in range(maxq)]
    else:
        print(f"Sparse type {sparse_type} has not been implemented yet. Please use csr, csc or lil.")
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

    print("Computing boundary matrices done.")

    return boundary_matrices, name_to_idx, simplices_at_time, relevant_times

def cross_Laplacian(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=False, Laplacian_fun = None, sparse_type = "csr"):
    if sparse_type == "csr":
        zero_matrix = sp.csr_matrix
    elif sparse_type == "csc":
        zero_matrix = sp.csc_matrix
    elif sparse_type == "lil":
        zero_matrix = sp.lil_matrix
    else:
        print(f"Sparse type {sparse_type} has not been implemented yet. Please use csr, csc or lil.")


    t, s = relevant_times[t_i], relevant_times[s_i]
    tm1 = relevant_times[t_i-1]

    if verb:
        print(f"Bqplus1:\n{boundary_matrices[q+1]}")
        print(f"n_q_t:{simplices_at_time(t)}, n_q_s: {simplices_at_time(s)}")
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
        eye_sm1t = sp.eye(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1], format = sparse_type)
        B22_sm1t_full = pinv(B22_sm1t)@B22_sm1t

        B22_st_full = zero_matrix(eye_sm1t.shape)
        B22_st_full[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = pinv(B22_st)@B22_st

        # B22_st_full = zero_matrix(eye_sm1t.shape)
        # B22_st_full[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = B22_st@B22_st

        B22_stm1_full = zero_matrix(eye_sm1t.shape)
        B22_stm1_partial_full = sp.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], format = sparse_type)
        B22_stm1_partial_full[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = pinv(B22_stm1)@B22_stm1
        B22_stm1_full[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = B22_stm1_partial_full

        B22_sm1tm1_full = sp.eye(eye_sm1t.shape[0], format = sparse_type)
        B22_sm1tm1_full[:(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1])] = pinv(B22_sm1tm1)@B22_sm1tm1

        if verb:
            print(f"A_matrix:\n{A_matrix.toarray()}")
            print(f"B22_st_full:\n{np.round(B22_st_full,decimals=5)}")
            print(f"B22_stm1_full:\n{np.round(B22_stm1_full.toarray(),decimals=5)}")
            print(f"B22_sm1t_full:\n{np.round(B22_sm1t_full.toarray(),decimals=5)}")
            print(f"B22_sm1tm1_full:\n{np.round(B22_sm1tm1_full.toarray(),decimals=5)}")
            print(f"B22_stm1_full@B22_sm1t_full:\n{np.round((B22_stm1_full@B22_sm1t_full).toarray(),decimals=5)}")

        # if verb:
            # VS = scipy.linalg.null_space(B22_st_full)
            # print(f"VS ({VS.shape}):\n{VS}")
            # G = VS.T@QR_matrix.T@QR_matrix@VS
            # print(f"G:\n{torch.round(G, decimals=5)}")
            # P1 = VS.T@B22_stm1_full@VS
            # P2 = VS.T@B22_sm1t_full@VS

            # Test_m1 = G@P1@P2
            # Test_m1 = torch.round(B22_stm1_full@(eye_sm1t-B22_st_full), decimals=6)
            # print(f"Test Matrix 1:\n{torch.round(Test_m1, decimals=4)}")
            # evals, evecs = torch.linalg.eigh(torch.round(Test_m1, decimals=4))
            # ker_B22 = []
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

    eye_st = sp.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], format= sparse_type)

    B22_st_full = pinv(B22_st)@B22_st

    B22_stm1_full = sp.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], format= sparse_type)
    B22_stm1_full[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = pinv(B22_stm1)@B22_stm1

    return B12_st@(eye_st-B22_st_full)@B22_stm1_full@B12_st.T

def calc_cross(f: d.Filtration, q, s, t, weight_fun = lambda x: 1, verb=False, Laplacian_fun = None, sparse_type = "csr"):
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun, sparse_type=sparse_type)
    relevant_times = np.array(relevant_times)
    t_i = np.argmin(np.abs(relevant_times - t))
    s_i = np.argmin(np.abs(relevant_times - s))
    return cross_Laplacian(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=verb, Laplacian_fun= Laplacian_fun, sparse_type=sparse_type)

def cross_Laplaican_eigenvalues_less_memory(f: d.Filtration, weight_fun = lambda x: 1, max_dim = 1, Laplacian_fun = None, device = "cuda"):
    f.sort()
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun, device=device)
    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    # projection_matrices = {q: {s: {t: torch.tensor([], dtype=torch.float64) for t in range(len(relevant_times))} for s in  range(len(relevant_times))} for q in range(max_dim+1)}
    
    if Laplacian_fun is None:
        Laplacian_fun = lambda B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye: B22_sm1t@B22_stm1@(eye-B22_st)@B22_stm1@B22_sm1t
    
    for q in range(max_dim+1):
        projection_matrices = {"sm1": {t: None for t in range(len(relevant_times))}, "s": {t: None for t in range(len(relevant_times))}}
        s_i_bar = tqdm(range(len(relevant_times)-1), leave=False)
        for s_i in s_i_bar:
            s = relevant_times[s_i]

            # NOTE: if n_q^K==n_q^L, then the persistent up-laplacian is just the combinatorial up-laplacian in L. Therefore, we can let B22 be 0.
            # NOTE: if n_q+1^K==n_q+1^L, then the persistent up-laplacian is just the combinatorial up-laplacian in K. Therefore, we can let B22 be the identity.
            obtained_B22 = False
            for t_i in range(s_i, len(relevant_times)):
                s_i_bar.set_description(f"t_i: {t_i}/{len(relevant_times)}")
                t = relevant_times[t_i]
                # tm1 = relevant_times[t_i-1]

                # Get B22^(-1)@B22
                if not obtained_B22:
                    if simplices_at_time(t)[q+1] == simplices_at_time(s)[q+1]:
                        cur_B22 = torch.zeros((0,0), device = device)
                    elif simplices_at_time(t)[q] == simplices_at_time(s)[q]:
                        cur_B22 = torch.zeros((simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1]), device = device)

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
                        cur_B22 = A_km1_pinv@A_km1
                    # Else stays the same
                
                # Calculate Laplacian and eigenvalues
                if s_i != t_i and t_i > 0:
                    tm1 = relevant_times[t_i-1]
                    if s_i > 0:
                        sm1 = relevant_times[s_i-1]

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
                        # cross_Lap = Laplacian_fun(B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye)

                        eigenvalues[q][s][t] = torch.linalg.eigvals(cross_Lap).cpu().numpy().real
                    
                    else:
                        B12_st = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]

                        eye = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], device = device)

                        B22_st = cur_B22

                        B22_stm1 = torch.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], device = device)
                        B22_stm1[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = projection_matrices["sm1"][t_i-1]

                        cross_Lap = B12_st@(B22_stm1 - B22_st)@B12_st.T

                        eigenvalues[q][s][t] = torch.linalg.eigvals(cross_Lap).cpu().numpy().real
                    

                # Saving and removing the right matrices
                if s_i > 0:
                    projection_matrices["s"][t_i] = cur_B22
                    projection_matrices["sm1"][t_i-1] = projection_matrices["s"][t_i-1]
                    projection_matrices["s"][t_i-1] = None
                else:
                    projection_matrices["sm1"][t_i] = cur_B22

                if t_i == len(relevant_times)-1:
                    projection_matrices["sm1"][t_i] = cur_B22

    return eigenvalues, relevant_times