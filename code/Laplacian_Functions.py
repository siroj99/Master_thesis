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

def SchurComp(M, ind):
    ind = [i if i >=0 else M.shape[0] + i for i in ind]
    nind = [i for i in range(M.shape[0]) if i not in ind]
    A = M[nind, :][:, nind]
    B = M[nind, :][:, ind]
    C = M[ind, :][:, nind]
    D = M[ind, :][:, ind]
    return A - B @ np.linalg.pinv(D) @ C

def combinatorial_Laplacian(Bqplus1, Bq):
    upLaplacian = Bqplus1@Bqplus1.T
    if Bq == 0:
        return upLaplacian
    else:
        downLaplacian = Bq.T@Bq
        return upLaplacian + downLaplacian

def persistent_Laplacian(Bqplus1: np.array, Bq: np.array, verb = False) -> np.array:
    """
    Inclusion of K in L. Assume the boundaries are ordered the same and first the boundaries in K appear in L.

    Bqplus1: (q+1)-boundary matrix of L.
    Bq: q-boundary matrix of K. (nqK if q=0)
    """
    if type(Bq) == int:
        nqK = Bq
        downL = 0
    else:
        nqK = Bq.shape[1]
        downL = Bq.T@Bq

    if type(Bqplus1) != int:
        # print("Bqplus1:", Bqplus1)
        upLaplacian = Bqplus1@Bqplus1.T
        # print("Delta^L:", upLaplacian)
        nqL = Bqplus1.shape[0]
        IKL = list(range(nqK, nqL))
        upL = SchurComp(upLaplacian, IKL)
    else:
        upL = 0
    if verb:
        print("upL:\n", upL)
        if type(upL) != int:
            eigval, eigvec = np.linalg.eigh(upL)
            eigval_product = 1
            for i, val in enumerate(eigval):
                print(f"eval: {np.round(val,2)}, evec: {np.round(eigvec[:,i]/np.min(np.abs(eigvec[:,i][np.abs(eigvec[:,i]) > 1e-8])),2)}")
                if val > 1e-8:
                    eigval_product *= val
            print("Product of eigenvalues:", np.round(eigval_product, 3))
        print()
        print("downL:\n", downL)
        if type(downL) != int:
            eigval, eigvec = np.linalg.eigh(downL)
            eigval_product = 1
            for i, val in enumerate(eigval):
                print(f"eval: {np.round(val,2)}, evec: {np.round(eigvec[:,i]/np.min(np.abs(eigvec[:,i][np.abs(eigvec[:,i]) > 1e-8])),2)}")
                if val > 1e-8:
                    eigval_product *= val
            print("Product of eigenvalues:", np.round(eigval_product, 3))
        print()
    return upL + downL

def compute_boundary_matrices(f: dio.Filtration, weight_fun):
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

    # boundary_matrices = [0] + [np.zeros((simplices_at_end[q-1], simplices_at_end[q])) for q in range(1, maxq)]
    boundary_matrices = [np.zeros((simplices_at_end[max(0,q-1)], simplices_at_end[q])) for q in range(maxq)]
    name_to_idx = [{} for _ in range(maxq)]
    for s in f:
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

        

    return boundary_matrices, name_to_idx, simplices_at_time, relevant_times

def persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time, verb=False, up_only = False):
    if up_only:
        Bq = simplices_at_time(s)[q]
    elif q > 0:
        Bq = boundary_matrices[q][:simplices_at_time(s)[q-1], :simplices_at_time(s)[q]]
    else:
        Bq = simplices_at_time(s)[q]
    
    if q+1 < len(simplices_at_time(t)) and q+1 < len(boundary_matrices):
        Bqplus1 = boundary_matrices[q+1][:simplices_at_time(t)[q], :simplices_at_time(t)[q+1]]
    else:
        Bqplus1 = 0

    if verb:
        print("Bqplus1:", Bqplus1)
    return persistent_Laplacian(Bqplus1, Bq, verb=verb)

# def persistent_Laplacian_new(Bqplus1: np.array, Bq: np.array, verb = False) -> np.array:
#     """
#     Inclusion of K in L. Assume the boundaries are ordered the same and first the boundaries in K appear in L.

#     Bqplus1: (q+1)-boundary matrix of L.
#     Bq: q-boundary matrix of K. (nqK if q=0)
#     """
#     if type(Bq) == int:
#         nqK = Bq
#         downL = 0
#     else:
#         nqK = Bq.shape[1]
#         downL = Bq.T@Bq

#     if type(Bqplus1) != int:
#         upLaplacian = Bqplus1@Bqplus1.T
#         nqL = Bqplus1.shape[0]
#         IKL = list(range(nqK, nqL))
#         upL = SchurComp(upLaplacian, IKL)
#     else:
#         upL = 0
#     if verb:
#         print("upL:\n", upL)
#         if type(upL) != int:
#             eigval, eigvec = np.linalg.eigh(upL)
#             for i, val in enumerate(eigval):
#                 print(np.round(val,2), np.round(eigvec[:,i]/np.min(np.abs(eigvec[:,i][np.abs(eigvec[:,i]) > 1e-8])),2))
#         print()
#         print("downL:\n", downL)
#         if type(downL) != int:
#             eigval, eigvec = np.linalg.eigh(downL)
#             for i, val in enumerate(eigval):
#                 print(np.round(val,2), np.round(eigvec[:,i]/np.min(np.abs(eigvec[:,i][np.abs(eigvec[:,i]) > 1e-8])),2))
#         print()

#     eigvals, eigvecs = np.linalg.eigh(upL + downL)
#     non_zero_evals = eigvals[eigvals > 1e-8]
#     if len(non_zero_evals) > 0:
#         product = np.prod(non_zero_evals)
#     else:
#         product = 1
#     betti = np.sum(eigvals<1e-8)

#     return betti, product

# def persistent_Laplacian_filtration_new(q, boundary_matrices, s, t, simplices_at_time, verb=False):
#     if q > 0:
#         Bq = boundary_matrices[q][:simplices_at_time(s)[q-1], :simplices_at_time(s)[q]]
#     else:
#         Bq = simplices_at_time(s)[q]
    
#     if q+1 <= len(simplices_at_time(t)):
#         Bqplus1 = boundary_matrices[q+1][:simplices_at_time(t)[q], :simplices_at_time(t)[q+1]]
#     else:
#         Bqplus1 = 0
#     return persistent_Laplacian_new(Bqplus1, Bq, verb=verb)

def complete_analysis(f: dio.Filtration, weight_fun, max_dim = 1):
    f.sort()
    max_time = int(f[len(f)-1].data)
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    barcodes = {"q": [], "birth": [], "death": [], "multiplicity": [], 
                "eigenvalues_ijm1": [], "eigenvalues_ij": [], "eigenvalues_im1jm1": [], "eigenvalues_im1j": [],
                "p_ijm1": [], "p_ij": [], "p_im1jm1": [], "p_im1j": []}
    
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(1,len(relevant_times)))
        for t_i in t_i_bar:
            for s_i in range(t_i):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t, tm1 = relevant_times[s_i], relevant_times[t_i], relevant_times[t_i-1] 

                if s_i != 0:
                    sm1 = relevant_times[s_i-1]
                
                Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, tm1, simplices_at_time)
                evals_ijm1 = np.linalg.eigh(Lap)[0]
                bijm1 = np.sum(evals_ijm1 < 1e-8)
                non_zero_evals = evals_ijm1[evals_ijm1 > 1e-8]
                if len(non_zero_evals) > 0:
                    pijm1 = np.prod(non_zero_evals)
                else:
                    pijm1 = 1

                Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
                evals_ij = np.linalg.eigh(Lap)[0]
                bij = np.sum(evals_ij < 1e-8)
                non_zero_evals = evals_ij[evals_ij > 1e-8]
                if len(non_zero_evals) > 0:
                    pij = np.prod(non_zero_evals)
                else:
                    pij = 1

                if s_i > 0:
                    Lap = persistent_Laplacian_filtration(q, boundary_matrices, sm1, tm1, simplices_at_time)
                    evals_im1jm1 = np.linalg.eigh(Lap)[0]
                    bim1jm1 = np.sum(evals_im1jm1 < 1e-8)
                    non_zero_evals = evals_im1jm1[evals_im1jm1 > 1e-8]
                    if len(non_zero_evals) > 0:
                        pim1jm1 = np.prod(non_zero_evals)
                    else:
                        pim1jm1 = 1

                    Lap = persistent_Laplacian_filtration(q, boundary_matrices, sm1, t, simplices_at_time)
                    evals_im1j = np.linalg.eigh(Lap)[0]
                    bim1j = np.sum(evals_im1j < 1e-8)
                    non_zero_evals = evals_im1j[evals_im1j > 1e-8]
                    if len(non_zero_evals) > 0:
                        pim1j = np.prod(non_zero_evals)
                    else:
                        pim1j = 1
                else:
                    evals_im1jm1, bim1jm1, pim1jm1, evals_im1j, bim1j, pim1j = [], 0, 1, [], 0, 1

                
                if bijm1-bij-(bim1jm1-bim1j) > 0:
                    barcodes["q"].append(q)
                    barcodes["birth"].append(s)
                    barcodes["death"].append(t)
                    barcodes["multiplicity"].append(bijm1-bij-(bim1jm1-bim1j))
                    barcodes["eigenvalues_ijm1"].append(evals_ijm1)
                    barcodes["eigenvalues_ij"].append(evals_ij)
                    barcodes["eigenvalues_im1jm1"].append(evals_im1jm1)
                    barcodes["eigenvalues_im1j"].append(evals_im1j)
                    barcodes["p_ijm1"].append(pijm1)
                    barcodes["p_ij"].append(pij)
                    barcodes["p_im1jm1"].append(pim1jm1)
                    barcodes["p_im1j"].append(pim1j)

                if t == max_time and bij-bim1j > 0:
                    barcodes["q"].append(q)
                    barcodes["birth"].append(s)
                    barcodes["death"].append(np.inf)
                    barcodes["multiplicity"].append(bij-bim1j)
                    barcodes["eigenvalues_ijm1"].append(evals_ijm1)
                    barcodes["eigenvalues_ij"].append(evals_ij)
                    barcodes["eigenvalues_im1jm1"].append(evals_im1jm1)
                    barcodes["eigenvalues_im1j"].append(evals_im1j)
                    barcodes["p_ijm1"].append(pijm1)
                    barcodes["p_ij"].append(pij)
                    barcodes["p_im1jm1"].append(pim1jm1)
                    barcodes["p_im1j"].append(pim1j)
    
    return pd.DataFrame(barcodes)

def compute_Laplacian(f: dio.Filtration, q, s, t, weight_fun, verb=False):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    return persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time, verb=verb)

def compute_cross_Laplacian(f: dio.Filtration, q, s, t, weight_fun, verb=False, use_4 = False):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    if verb:
        print("Laplacian s to t")
    Lap_ij = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time, verb=verb)
    Lap_ij = np.linalg.pinv(Lap_ij)@Lap_ij

    if verb:
        print("Laplacian s to t-1")
    Lap_ijm1 = persistent_Laplacian_filtration(q, boundary_matrices, s, t-1, simplices_at_time, verb=verb)
    Lap_ijm1 = np.linalg.pinv(Lap_ijm1)@Lap_ijm1

    if s >= 1:
        if verb:
            print("Laplacian s-1 to t-1")
        Lap_im1jm1 = np.eye(simplices_at_time(s)[q])
        Lap_temp = persistent_Laplacian_filtration(q, boundary_matrices, s-1, t-1, simplices_at_time, verb=verb)
        Lap_im1jm1[:simplices_at_time(s-1)[q], :simplices_at_time(s-1)[q]] = np.linalg.pinv(Lap_temp)@Lap_temp

        if verb:
            print("Laplacian s-1 to t")
        Lap_im1j = np.eye(simplices_at_time(s)[q])
        Lap_temp = persistent_Laplacian_filtration(q, boundary_matrices, s-1, t, simplices_at_time, verb=verb) 
        Lap_im1j[:simplices_at_time(s-1)[q], :simplices_at_time(s-1)[q]] =np.linalg.pinv(Lap_temp)@Lap_temp

        return  Lap_ijm1 - Lap_ij - (Lap_im1jm1 - Lap_im1j)
    
    else:
        return  (Lap_ijm1 - Lap_ij)
    

def compute_vertical_Laplacian(f: dio.Filtration, q, s, t, weight_fun, verb=False, use_restriction = False):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    relevant_times = np.array(relevant_times)
    s_i, t_i = np.argmin(np.abs(relevant_times-s)), np.argmin(np.abs(relevant_times-t))
    sm1 = relevant_times[s_i-1]
    if verb:
        print("Laplacian s to t")
    Lap_ij = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time, verb=verb)

    if verb:
        print("Laplacian s-1 to t")
    Lap_im1j = np.zeros((simplices_at_time(s)[q],simplices_at_time(s)[q]))
    Lap_im1j[:simplices_at_time(sm1)[q], :simplices_at_time(sm1)[q]] = persistent_Laplacian_filtration(q, boundary_matrices, sm1, t, simplices_at_time, verb=verb)
    
    if use_restriction:
        return (Lap_ij-Lap_im1j)[:simplices_at_time(sm1)[q], :simplices_at_time(sm1)[q]]
    return  Lap_ij-Lap_im1j

def complete_analysis_fast(f: dio.Filtration, weight_fun, max_dim = 1):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}

    barcodes = {"q": [], "birth": [], "death": [], "multiplicity": [], 
                "eigenvalues_ijm1": [], "eigenvalues_ij": [], "eigenvalues_im1jm1": [], "eigenvalues_im1j": [],
                "p_ijm1": [], "p_ij": [], "p_im1jm1": [], "p_im1j": []}
    
    for q in range(max_dim+1):
        # t_i_bar = tqdm(range(len(relevant_times)), leave=False)         # CHANGED THIS FOR LOADING BAR!!!!!
        t_i_bar = range(len(relevant_times))
        for t_i in t_i_bar:
            for s_i in range(t_i+1):
                # t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
                
                eigenvalues[q][s][t] = np.linalg.eigh(Lap)[0]

    for q in range(max_dim+1):
        for t_i in range(len(relevant_times)):
            for s_i in range(t_i+1):
                s, t, tm1 = relevant_times[s_i], relevant_times[t_i], relevant_times[t_i-1] 

                if s_i != 0:
                    sm1 = relevant_times[s_i-1]
                
                evals_ijm1 = eigenvalues[q][s][tm1]
                bijm1 = np.sum(evals_ijm1 < 1e-8)
                non_zero_evals = evals_ijm1[evals_ijm1 > 1e-8]
                if len(non_zero_evals) > 0:
                    pijm1 = np.prod(non_zero_evals)
                else:
                    pijm1 = 1

                evals_ij =  eigenvalues[q][s][t]
                bij = np.sum(evals_ij < 1e-8)
                non_zero_evals = evals_ij[evals_ij > 1e-8]
                if len(non_zero_evals) > 0:
                    pij = np.prod(non_zero_evals)
                else:
                    pij = 1

                if s_i > 0:
                    evals_im1jm1 =  eigenvalues[q][sm1][tm1]
                    bim1jm1 = np.sum(evals_im1jm1 < 1e-8)
                    non_zero_evals = evals_im1jm1[evals_im1jm1 > 1e-8]
                    if len(non_zero_evals) > 0:
                        pim1jm1 = np.prod(non_zero_evals)
                    else:
                        pim1jm1 = 1

                    evals_im1j =  eigenvalues[q][sm1][t]
                    bim1j = np.sum(evals_im1j < 1e-8)
                    non_zero_evals = evals_im1j[evals_im1j > 1e-8]
                    if len(non_zero_evals) > 0:
                        pim1j = np.prod(non_zero_evals)
                    else:
                        pim1j = 1
                else:
                    evals_im1jm1, bim1jm1, pim1jm1, evals_im1j, bim1j, pim1j = [], 0, 1, [], 0, 1

                
                # if bijm1-bij-(bim1jm1-bim1j) > 0:
                #     barcodes["q"].append(q)
                #     barcodes["birth"].append(s)
                #     barcodes["death"].append(t)
                #     barcodes["multiplicity"].append(bijm1-bij-(bim1jm1-bim1j))
                #     barcodes["eigenvalues_ijm1"].append(evals_ijm1)
                #     barcodes["eigenvalues_ij"].append(evals_ij)
                #     barcodes["eigenvalues_im1jm1"].append(evals_im1jm1)
                #     barcodes["eigenvalues_im1j"].append(evals_im1j)
                #     barcodes["p_ijm1"].append(pijm1)
                #     barcodes["p_ij"].append(pij)
                #     barcodes["p_im1jm1"].append(pim1jm1)
                #     barcodes["p_im1j"].append(pim1j)

                if bijm1-bij > 0:
                    barcodes["q"].append(q)
                    barcodes["birth"].append(s)
                    barcodes["death"].append(t)
                    barcodes["multiplicity"].append(bijm1-bij)
                    barcodes["eigenvalues_ijm1"].append(evals_ijm1)
                    barcodes["eigenvalues_ij"].append(evals_ij)
                    barcodes["eigenvalues_im1jm1"].append(evals_im1jm1)
                    barcodes["eigenvalues_im1j"].append(evals_im1j)
                    barcodes["p_ijm1"].append(pijm1)
                    barcodes["p_ij"].append(pij)
                    barcodes["p_im1jm1"].append(pim1jm1)
                    barcodes["p_im1j"].append(pim1j)

                if t == max_time and bij-bim1j > 0:
                    barcodes["q"].append(q)
                    barcodes["birth"].append(s)
                    barcodes["death"].append(np.inf)
                    barcodes["multiplicity"].append(bij-bim1j)
                    barcodes["eigenvalues_ijm1"].append(evals_ijm1)
                    barcodes["eigenvalues_ij"].append(evals_ij)
                    barcodes["eigenvalues_im1jm1"].append(evals_im1jm1)
                    barcodes["eigenvalues_im1j"].append(evals_im1j)
                    barcodes["p_ijm1"].append(pijm1)
                    barcodes["p_ij"].append(pij)
                    barcodes["p_im1jm1"].append(pim1jm1)
                    barcodes["p_im1j"].append(pim1j)
    
    return pd.DataFrame(barcodes)

def complete_analysis_fastest(f: dio.Filtration, weight_fun, max_dim = 1):
    """
    Requires one simplex added per time.
    """

    f.sort()
    max_time = int(f[len(f)-1].data)
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    barcodes = {"q": [], "birth": [], "death": [], "multiplicity": [], 
                "eigenvalues_ijm1": [], "eigenvalues_ij": [], "eigenvalues_im1jm1": [], "eigenvalues_im1j": [],
                "p_ijm1": [], "p_ij": [], "p_im1jm1": [], "p_im1j": []}
    
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(1,len(relevant_times)))
        for t_i in t_i_bar:
            s_i = t_i-1

            for s_i in range(t_i):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t, tm1 = relevant_times[s_i], relevant_times[t_i], relevant_times[t_i-1] 

                if s_i != 0:
                    sm1 = relevant_times[s_i-1]
                
                Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, tm1, simplices_at_time)
                evals_ijm1 = np.linalg.eigh(Lap)[0]
                bijm1 = np.sum(evals_ijm1 < 1e-8)
                non_zero_evals = evals_ijm1[evals_ijm1 > 1e-8]
                if len(non_zero_evals) > 0:
                    pijm1 = np.prod(non_zero_evals)
                else:
                    pijm1 = 1

                Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
                evals_ij = np.linalg.eigh(Lap)[0]
                bij = np.sum(evals_ij < 1e-8)
                non_zero_evals = evals_ij[evals_ij > 1e-8]
                if len(non_zero_evals) > 0:
                    pij = np.prod(non_zero_evals)
                else:
                    pij = 1

                if s_i > 0:
                    Lap = persistent_Laplacian_filtration(q, boundary_matrices, sm1, tm1, simplices_at_time)
                    evals_im1jm1 = np.linalg.eigh(Lap)[0]
                    bim1jm1 = np.sum(evals_im1jm1 < 1e-8)
                    non_zero_evals = evals_im1jm1[evals_im1jm1 > 1e-8]
                    if len(non_zero_evals) > 0:
                        pim1jm1 = np.prod(non_zero_evals)
                    else:
                        pim1jm1 = 1

                    Lap = persistent_Laplacian_filtration(q, boundary_matrices, sm1, t, simplices_at_time)
                    evals_im1j = np.linalg.eigh(Lap)[0]
                    bim1j = np.sum(evals_im1j < 1e-8)
                    non_zero_evals = evals_im1j[evals_im1j > 1e-8]
                    if len(non_zero_evals) > 0:
                        pim1j = np.prod(non_zero_evals)
                    else:
                        pim1j = 1
                else:
                    evals_im1jm1, bim1jm1, pim1jm1, evals_im1j, bim1j, pim1j = [], 0, 1, [], 0, 1

                
                if bijm1-bij-(bim1jm1-bim1j) > 0:
                    barcodes["q"].append(q)
                    barcodes["birth"].append(s)
                    barcodes["death"].append(t)
                    barcodes["multiplicity"].append(bijm1-bij-(bim1jm1-bim1j))
                    barcodes["eigenvalues_ijm1"].append(evals_ijm1)
                    barcodes["eigenvalues_ij"].append(evals_ij)
                    barcodes["eigenvalues_im1jm1"].append(evals_im1jm1)
                    barcodes["eigenvalues_im1j"].append(evals_im1j)
                    barcodes["p_ijm1"].append(pijm1)
                    barcodes["p_ij"].append(pij)
                    barcodes["p_im1jm1"].append(pim1jm1)
                    barcodes["p_im1j"].append(pim1j)

                if t == max_time and bij-bim1j > 0:
                    barcodes["q"].append(q)
                    barcodes["birth"].append(s)
                    barcodes["death"].append(np.inf)
                    barcodes["multiplicity"].append(bij-bim1j)
                    barcodes["eigenvalues_ijm1"].append(evals_ijm1)
                    barcodes["eigenvalues_ij"].append(evals_ij)
                    barcodes["eigenvalues_im1jm1"].append(evals_im1jm1)
                    barcodes["eigenvalues_im1j"].append(evals_im1j)
                    barcodes["p_ijm1"].append(pijm1)
                    barcodes["p_ij"].append(pij)
                    barcodes["p_im1jm1"].append(pim1jm1)
                    barcodes["p_im1j"].append(pim1j)
    
    return pd.DataFrame(barcodes)

def persistent_Laplacian_eigenvalues(f: dio.Filtration, weight_fun, max_dim = 1):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)), leave=False)
        for t_i in t_i_bar:
            for s_i in range(t_i+1):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
                
                eigenvalues[q][s][t] = np.linalg.eigvalsh(Lap)
    return eigenvalues, relevant_times

def cross_Laplacian_eigenvalues(f, weight_fun, max_dim = 1, use_4 = False):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    laplacians = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    
    print("Computing laplacians...")
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)), leave=False)
        for t_i in t_i_bar:
            for s_i in range(t_i):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]
                # Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
                try:
                    Lap = compute_cross_Laplacian(f, q, s, t, weight_fun=weight_fun)
                except:
                    print(s, t)
                    raise ValueError
                # laplacians[q][s][t] = Lap
                eigenvalues[q][s][t] = np.linalg.eigvalsh(Lap)

    # eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    # print("Computing eigenvalues...")
    # for q in range(max_dim+1):
    #     t_i_bar = tqdm(range(len(relevant_times)), leave=False)
    #     for t_i in t_i_bar:
    #         for s_i in range(t_i):
    #             t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
    #             s, t, tm1 = relevant_times[s_i], relevant_times[t_i], relevant_times[t_i-1] 
    #             if s_i != 0 and use_4:
    #                 sm1 = relevant_times[s_i-1]
    #                 Lap_im1jm1 = laplacians[q][sm1][tm1]
    #                 Lap_im1j = laplacians[q][sm1][t]
    #                 part_im1 = Lap_im1jm1 - Lap_im1j
    #             else:
    #                 part_im1 = np.zeros((1,1))

    #             Lap_ij = laplacians[q][s][t]
    #             Lap_ijm1 = laplacians[q][s][tm1]

    #             if use_4:
    #                 b = np.zeros_like(Lap_ij)
    #                 b[:part_im1.shape[0], :part_im1.shape[1]] = part_im1
    #                 cross_Lap = -1*((Lap_ijm1 - Lap_ij) - (b))
    #             else:
    #                 cross_Lap = Lap_ij-Lap_ijm1
                
    #             try:
    #                 eigenvalues[q][s][t] = np.linalg.eigvalsh(cross_Lap)
    #             except:
    #                 print(f"s: {s}, t: {t}, q: {q}")
    #                 print(f"Lap_ij\n{Lap_ij}")
    #                 print(f"Lap_ijm1\n{Lap_ijm1}")
    #                 if s_i != 0:
    #                     print(f"b:\n{b}")
    #                 raise ValueError

    return eigenvalues, relevant_times

def vertical_Laplacian_eigenvalues(f, weight_fun, max_dim = 1, use_restriction = True):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    laplacians = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    
    print("Computing laplacians...")
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)), leave=False)
        for t_i in t_i_bar:
            for s_i in range(t_i+1):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]
                Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
                laplacians[q][s][t] = Lap

    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    print("Computing eigenvalues...")
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)), leave=False)
        for t_i in t_i_bar:
            for s_i in range(t_i):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                Lap_ij = laplacians[q][s][t]
                
                if s_i != 0:
                    sm1 = relevant_times[s_i-1]
                    Lap_im1j = np.zeros((simplices_at_time(s)[q], simplices_at_time(s)[q]))
                    Lap_im1j[:simplices_at_time(sm1)[q], :simplices_at_time(sm1)[q]] = laplacians[q][sm1][t]
                    cross_Lap = Lap_ij - Lap_im1j
                    if use_restriction:
                        cross_Lap = cross_Lap[:simplices_at_time(sm1)[q], :simplices_at_time(sm1)[q]]
                else:
                    cross_Lap = Lap_ij
                
                try:
                    eigenvalues[q][s][t] = np.linalg.eigvalsh(cross_Lap)
                except:
                    print(f"s: {s}, t: {t}, q: {q}")
                    print(f"Lap_ij\n{Lap_ij}")
                    if s_i != 0:
                        print(f"Lap_im1j\n{Lap_im1j}")
                    raise ValueError

    return eigenvalues, relevant_times

def eig_plot_helper(x, fun, eps = 1e-8):
    if len(x) > 0:
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
                     plot_type_to_fun = {}):
    # Can choose these plot types
    plot_type_to_fun = {
        "Min": np.min,
        "Max": np.max,
        "Sum": np.sum,
        "Mean": np.mean,
        "Prod": np.prod,
        "Gmean": ss.gmean
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
            if plot_type == "Prod":
                plot_args_mesh["norm"] = "log"
            else:
                plot_args_mesh["norm"] = "linear"
            im = cur_ax(q, ax_i).pcolormesh(x, y, df_evals.apply(lambda x: x.apply(lambda y: eig_plot_helper(y, plot_type_to_fun[plot_type]))).values, **plot_args_mesh)
            plt.colorbar(im, ax=cur_ax(q, ax_i), pad=0.15)
            if q == 0:
                cur_ax(q, ax_i).set_title(f"{plot_type} eigenvalue")
            if filtration is not None:
                cur_ax(q, ax_i).scatter(barcodes_births[q], barcodes_deaths[q], **plot_args_diag)

            ax_n = cur_ax(q, ax_i).twinx()
            ax_n.plot([relevant_times[0]] + [val for val in relevant_times[1:] for _ in (0, 1)], [eig_plot_helper(eigenvalues[q][t][t], plot_type_to_fun[plot_type]) for t in relevant_times for _ in (0,1)][:-1], **plot_args_line)
            if plot_type == "Prod":
                ax_n.set_yscale("log")

            if integer_time_steps:
                cur_ax(q, ax_i).yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        cur_ax(q, 0).set_ylabel(f"q={q}")
    fig.tight_layout()
    return fig, ax

def plot_Laplacian_eigenvalues(f: dio.Filtration, weight_fun, max_dim = 1, plot_types = "all", 
                     plot_args_mesh = {}, 
                     plot_args_diag = {},
                     plot_args_line = {},
                     plot_type_to_fun = {},
                     laplacian_type = "persistent",
                     use_restriction = True):
    """
    lapalcian_type: "persistent" for normal laplacian, or "cross" for cross laplacian.
    """
    if laplacian_type == "persistent":
        eigenvalues, relevant_times = persistent_Laplacian_eigenvalues(f, weight_fun, max_dim=max_dim)
    elif laplacian_type == "vertical":
        eigenvalues, relevant_times = vertical_Laplacian_eigenvalues(f, weight_fun, max_dim=max_dim, use_restriction=use_restriction)
    else:
        eigenvalues, relevant_times = cross_Laplacian_eigenvalues(f, weight_fun, max_dim=max_dim)

    fig, ax = plot_eigenvalues(eigenvalues, relevant_times, plot_types=plot_types, filtration=f,
                               plot_args_mesh = plot_args_mesh, plot_args_diag=plot_args_diag,
                               plot_args_line=plot_args_line, plot_type_to_fun=plot_type_to_fun)
    return eigenvalues, relevant_times, fig, ax

def plot_non_persistent_eigenvalues(eigenvalues, relevant_times, plot_type="all"):
    max_dim = max(eigenvalues.keys())
    if plot_type == "all":
        fig, ax = plt.subplots(max_dim+1, 4)
        fig.set_size_inches(4*4, 4*(max_dim+1))
        cur_ax = lambda x,y: ax[x, y]
    else:
        fig, ax = plt.subplots(max_dim+1, 1)
        fig.set_size_inches(4, 4*(max_dim+1))
        cur_ax = lambda x,y: ax[x]

    fig.gca()
    for q in range(max_dim+1):
        ax_i = 0

        if plot_type in ["min", "all"]:     
            im = cur_ax(q, ax_i).plot(relevant_times, [np.min(eigenvalues[q][t][t]) for t in relevant_times])
            if q == 0:
                cur_ax(q, ax_i).set_title("Minimum eigenvalue")
            ax_i += 1

        if plot_type in ["max", "all"]:
            im = cur_ax(q, ax_i).plot(relevant_times, [np.max(eigenvalues[q][t][t]) for t in relevant_times])
            if q == 0:
                cur_ax(q, ax_i).set_title("Maximum eigenvalue")
            ax_i += 1

        if plot_type in ["prod", "all"]:
            im = cur_ax(q, ax_i).plot(relevant_times, [np.prod(eigenvalues[q][t][t]) for t in relevant_times])
            if q == 0:
                cur_ax(q, ax_i).set_title("Product of eigenvalues")
            ax_i += 1
        
        if plot_type in ["sum", "all"]:
            im = cur_ax(q, ax_i).plot(relevant_times, [np.sum(eigenvalues[q][t][t]) for t in relevant_times])
            if q == 0:
                cur_ax(q, ax_i).set_title("Sum of eigenvalues")
            ax_i += 1

        cur_ax(q, 0).set_ylabel(f"q={q}")
    fig.tight_layout()
    fig.show()
    return fig, ax

# def complete_analysis_new(f: dio.Filtration, weight_fun, max_dim = 1):
#     f.sort()
#     max_time = int(f[len(f)-1].data)
#     boundary_matrices, name_to_idx, simplices_at_time = compute_boundary_matrices(f, weight_fun)

#     barcodes = {"q": [], "birth": [], "death": [], "multiplicity": [], "p_ijm1": [], "p_ij": [], "p_im1jm1": [], "p_im1j": []}
#     for q in range(max_dim+1):
#         for t in range(1,max_time+1):
#             for s in range(t):

#                 bijm1, pijm1 = persistent_Laplacian_filtration_new(q, boundary_matrices, s, t-1, simplices_at_time, verb=False)
#                 bij, pij = persistent_Laplacian_filtration_new(q, boundary_matrices, s, t, simplices_at_time)
#                 bim1jm1, pim1jm1 = persistent_Laplacian_filtration_new(q, boundary_matrices, s-1, t-1, simplices_at_time)
#                 bim1j, pim1j = persistent_Laplacian_filtration_new(q, boundary_matrices, s-1, t, simplices_at_time)

#                 if bijm1-bij-(bim1jm1-bim1j) > 0:
#                     barcodes["q"].append(q)
#                     barcodes["birth"].append(s)
#                     barcodes["death"].append(t)
#                     barcodes["multiplicity"].append(bijm1-bij-(bim1jm1-bim1j))
#                     barcodes["p_ijm1"].append(pijm1)
#                     barcodes["p_ij"].append(pij)
#                     barcodes["p_im1jm1"].append(pim1jm1)
#                     barcodes["p_im1j"].append(pim1j)

#                 if t == max_time and bij-bim1j > 0:
#                     barcodes["q"].append(q)
#                     barcodes["birth"].append(s)
#                     barcodes["death"].append(np.inf)
#                     barcodes["multiplicity"].append(bij-bim1j)
#                     barcodes["p_ijm1"].append(pijm1)
#                     barcodes["p_ij"].append(pij)
#                     barcodes["p_im1jm1"].append(pim1jm1)
#                     barcodes["p_im1j"].append(pim1j)

    
#     return pd.DataFrame(barcodes)