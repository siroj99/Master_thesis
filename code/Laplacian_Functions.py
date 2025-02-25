import numpy as np
from copy import deepcopy
import dionysus as d
import scipy
import pandas as pd
from tqdm.auto import tqdm
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
        upLaplacian = Bqplus1@Bqplus1.T
        nqL = Bqplus1.shape[0]
        IKL = list(range(nqK, nqL))
        upL = SchurComp(upLaplacian, IKL)
    else:
        upL = 0
    if verb:
        print("upL:\n", upL)
        if type(upL) != int:
            eigval, eigvec = np.linalg.eig(upL)
            eigval_product = 1
            for i, val in enumerate(eigval):
                print(f"eval: {np.round(val,2)}, evec: {np.round(eigvec[:,i]/np.min(np.abs(eigvec[:,i][np.abs(eigvec[:,i]) > 1e-8])),2)}")
                if val > 1e-8:
                    eigval_product *= val
            print("Product of eigenvalues:", np.round(eigval_product, 3))
        print()
        print("downL:\n", downL)
        if type(downL) != int:
            eigval, eigvec = np.linalg.eig(downL)
            eigval_product = 1
            for i, val in enumerate(eigval):
                print(f"eval: {np.round(val,2)}, evec: {np.round(eigvec[:,i]/np.min(np.abs(eigvec[:,i][np.abs(eigvec[:,i]) > 1e-8])),2)}")
                if val > 1e-8:
                    eigval_product *= val
            print("Product of eigenvalues:", np.round(eigval_product, 3))
        print()
    return upL + downL

def compute_boundary_matrices(f: d.Filtration, weight_fun):
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
        if q >= len(n_simplicies_seen_total):
            n_simplicies_seen_total.append(0)
        n_simplicies_seen_total[q] += 1

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
            if i == len(relevant_times) - 1:
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

def persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time, verb=False):
    if q > 0:
        Bq = boundary_matrices[q][:simplices_at_time(s)[q-1], :simplices_at_time(s)[q]]
    else:
        Bq = simplices_at_time(s)[q]
    
    if q+1 < len(simplices_at_time(t)) and q+1 < len(boundary_matrices):
        Bqplus1 = boundary_matrices[q+1][:simplices_at_time(t)[q], :simplices_at_time(t)[q+1]]
    else:
        Bqplus1 = 0
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
#             eigval, eigvec = np.linalg.eig(upL)
#             for i, val in enumerate(eigval):
#                 print(np.round(val,2), np.round(eigvec[:,i]/np.min(np.abs(eigvec[:,i][np.abs(eigvec[:,i]) > 1e-8])),2))
#         print()
#         print("downL:\n", downL)
#         if type(downL) != int:
#             eigval, eigvec = np.linalg.eig(downL)
#             for i, val in enumerate(eigval):
#                 print(np.round(val,2), np.round(eigvec[:,i]/np.min(np.abs(eigvec[:,i][np.abs(eigvec[:,i]) > 1e-8])),2))
#         print()

#     eigvals, eigvecs = np.linalg.eig(upL + downL)
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

def complete_analysis(f: d.Filtration, weight_fun, max_dim = 1):
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
                evals_ijm1 = np.linalg.eig(Lap)[0]
                bijm1 = np.sum(evals_ijm1 < 1e-8)
                non_zero_evals = evals_ijm1[evals_ijm1 > 1e-8]
                if len(non_zero_evals) > 0:
                    pijm1 = np.prod(non_zero_evals)
                else:
                    pijm1 = 1

                Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
                evals_ij = np.linalg.eig(Lap)[0]
                bij = np.sum(evals_ij < 1e-8)
                non_zero_evals = evals_ij[evals_ij > 1e-8]
                if len(non_zero_evals) > 0:
                    pij = np.prod(non_zero_evals)
                else:
                    pij = 1

                if s_i > 0:
                    Lap = persistent_Laplacian_filtration(q, boundary_matrices, sm1, tm1, simplices_at_time)
                    evals_im1jm1 = np.linalg.eig(Lap)[0]
                    bim1jm1 = np.sum(evals_im1jm1 < 1e-8)
                    non_zero_evals = evals_im1jm1[evals_im1jm1 > 1e-8]
                    if len(non_zero_evals) > 0:
                        pim1jm1 = np.prod(non_zero_evals)
                    else:
                        pim1jm1 = 1

                    Lap = persistent_Laplacian_filtration(q, boundary_matrices, sm1, t, simplices_at_time)
                    evals_im1j = np.linalg.eig(Lap)[0]
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

def compute_Laplacian(f: d.Filtration, q, s, t, weight_fun, verb=False):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    return persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time, verb=verb)


def complete_analysis_fast(f: d.Filtration, weight_fun, max_dim = 1):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}

    barcodes = {"q": [], "birth": [], "death": [], "multiplicity": [], 
                "eigenvalues_ijm1": [], "eigenvalues_ij": [], "eigenvalues_im1jm1": [], "eigenvalues_im1j": [],
                "p_ijm1": [], "p_ij": [], "p_im1jm1": [], "p_im1j": []}
    
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)))
        for t_i in t_i_bar:
            for s_i in range(t_i+1):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
                
                eigenvalues[q][s][t] = np.linalg.eig(Lap)[0]

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

def complete_analysis_fastest(f: d.Filtration, weight_fun, max_dim = 1):
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
                evals_ijm1 = np.linalg.eig(Lap)[0]
                bijm1 = np.sum(evals_ijm1 < 1e-8)
                non_zero_evals = evals_ijm1[evals_ijm1 > 1e-8]
                if len(non_zero_evals) > 0:
                    pijm1 = np.prod(non_zero_evals)
                else:
                    pijm1 = 1

                Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
                evals_ij = np.linalg.eig(Lap)[0]
                bij = np.sum(evals_ij < 1e-8)
                non_zero_evals = evals_ij[evals_ij > 1e-8]
                if len(non_zero_evals) > 0:
                    pij = np.prod(non_zero_evals)
                else:
                    pij = 1

                if s_i > 0:
                    Lap = persistent_Laplacian_filtration(q, boundary_matrices, sm1, tm1, simplices_at_time)
                    evals_im1jm1 = np.linalg.eig(Lap)[0]
                    bim1jm1 = np.sum(evals_im1jm1 < 1e-8)
                    non_zero_evals = evals_im1jm1[evals_im1jm1 > 1e-8]
                    if len(non_zero_evals) > 0:
                        pim1jm1 = np.prod(non_zero_evals)
                    else:
                        pim1jm1 = 1

                    Lap = persistent_Laplacian_filtration(q, boundary_matrices, sm1, t, simplices_at_time)
                    evals_im1j = np.linalg.eig(Lap)[0]
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

# def complete_analysis_new(f: d.Filtration, weight_fun, max_dim = 1):
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