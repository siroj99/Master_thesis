import numpy as np
from copy import deepcopy
import dionysus as d
import scipy
import pandas as pd
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import smith_normal_form
from collections import Counter

from Laplacian_Functions import *

def list_minus(x, y):
    counter_x = Counter(x)
    counter_y = Counter(y)

    c_xmy = counter_x - counter_y
    return list(c_xmy.elements())

def reduce_matrix(M, compute_kernel = False, compute_image = False, verb=False):
    M = deepcopy(M)
    low_i_dict = {}
    non_pivot_cols = []
    basis_kernel = []
    operations_applied = {}
    for col in range(M.shape[1]):
        low_i = M.shape[0]-1
        if verb:
            print(col, low_i_dict)

        if compute_kernel:
            basis = np.zeros(M.shape[1])
            basis[col] = 1
        while True:
            if verb:
                print(col, M)
            while M[low_i, col] == 0 and low_i >= 0:
                low_i -= 1
            if low_i < 0:
                non_pivot_cols.append(col)

                if compute_kernel:
                    if verb:
                        print("here:", basis)
                    basis_kernel.append(basis)
                break
            if low_i not in low_i_dict.keys():
                low_i_dict[low_i] = col
                if compute_kernel:
                    operations_applied[col] = basis
                break
            else:
                other_col = low_i_dict[low_i]
                lcm = np.lcm(int(M[low_i, col]), int(M[low_i, other_col]))
                
                if compute_kernel:
                    basis *= lcm//M[low_i, col]
                    basis -= lcm//M[low_i, other_col]*operations_applied[other_col]
                M[:, col] *= lcm//M[low_i, col]
                M[:, col] -= (lcm//M[low_i, other_col])*M[:, other_col]
    
    if compute_image:
        basis_image = []
        for low_i in low_i_dict.keys():
            basis_image.append(M[:, low_i_dict[low_i]])
            
    if compute_image and not compute_kernel:
        return basis_image
    if compute_kernel and not compute_image:
        return basis_kernel
    return basis_image, basis_kernel


def method_Ren(q, s, t, boundary_matrices, simplices_at_time):
    
    # Step (1)
    Z = reduce_matrix(boundary_matrices[q][:simplices_at_time(s)[max(0,q-1)], :simplices_at_time(s)[q]], compute_kernel=True)
    rank_Z = len(Z)

    # Step (2)
    B = reduce_matrix(boundary_matrices[q+1][:simplices_at_time(t)[q], :simplices_at_time(t)[q+1]], compute_image=True)
    rank_B = len(B)
    
    # THINK THIS DOES NOT MAKE SENSE, BUT MIGHT BE A STRONG ASSUMPTION!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if rank_Z == 0:
        return 0, []
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Z = np.vstack(Z).T

    if rank_B > 0:
        B = np.vstack(B).T
        Z = np.pad(Z, [(0,B.shape[0]-Z.shape[0]), (0,0)], constant_values=0)
    else:
        B = np.array([[]])

    # Step (3)
    if rank_B > 0:
        A = np.hstack([Z, B])
    else:
        A = deepcopy(Z)
    
    a_q = reduce_matrix(A, compute_kernel=True)

    # Step (4)
    B_s_t = []
    for a in a_q:
        # Does not matter which one you choose. (beta works always, while alpha may not exist!)
        beta_q = a[:rank_Z]
        B_s_t.append((Z@beta_q)[:,np.newaxis])
        # alpha_q = a[-rank_B:]

        # B_s_t.append((B@alpha_q)[:,np.newaxis])
    if len(B_s_t) > 0:
        M_kp1 = np.hstack(B_s_t)
    else:
        M_kp1 = np.array([[]])

    # Step (5)
    # im_M_kp1 = reduce_matrix(M_kp1, compute_image=True)
    M_kp1 = Matrix(M_kp1)
    M_reduced = np.array(smith_normal_form(M_kp1, domain=ZZ)).astype(np.float64)
    torsion_coefs = np.abs(np.diag(M_reduced))   


    rank_B_s_t = len(B_s_t)
    torsion_coefs = [i for i in torsion_coefs if i not in  [0,1]]

    # Step (6)
    betti_number = rank_Z - rank_B_s_t

    return betti_number, torsion_coefs

def complete_analysis_Ren_fast(f, weight_fun):
    f.sort()
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    p = d.cohomology_persistence(f, 47, True)
    dgms = d.init_diagrams(p, f)
    barcodes = {"q": [], "birth": [], "death": [], "multiplicity": [], 
                "torsion_ijm1": [], "torsion_ij": [], "torsion_im1jm1": [], "torsion_im1j": []}
    for q in range(len(dgms)):
        for point in tqdm(dgms[q]):

            birth_i = relevant_times.index(point.birth)
            if point.death == np.inf:
                death_i = -1
            else:
                death_i = relevant_times.index(point.death)
            try:
                if point.death != np.inf:
                    bijm1, tijm1 = method_Ren(q, relevant_times[birth_i], relevant_times[death_i-1]
                                        , boundary_matrices, simplices_at_time)
                
                bij, tij = method_Ren(q, relevant_times[birth_i], relevant_times[death_i]
                                        , boundary_matrices, simplices_at_time)

                if birth_i > 0:
                    if point.death != np.inf:
                        bim1jm1, tim1jm1 = method_Ren(q, relevant_times[birth_i-1], relevant_times[death_i-1]
                                            , boundary_matrices, simplices_at_time)
                    
                    bim1j, tim1j = method_Ren(q, relevant_times[birth_i-1], relevant_times[death_i]
                                            , boundary_matrices, simplices_at_time)
                    
                else:
                    tim1jm1, bim1jm1, tim1j, bim1j = [], 0, [], 0

            except:
                print(q, point.birth, point.death, birth_i, death_i)
                raise ValueError()
            barcodes["q"]
            barcodes["q"].append(q)
            barcodes["birth"].append(point.birth)
            barcodes["death"].append(point.death)
            if point.death == np.inf:
                barcodes["multiplicity"].append(bij-bim1j)
            else: 
                barcodes["multiplicity"].append(bijm1-bij-(bim1jm1-bim1j))
            
            if point.death != np.inf:
                barcodes["torsion_ijm1"].append(tijm1)
            else:
                barcodes["torsion_ijm1"].append(None)
            barcodes["torsion_ij"].append(tij)

            if point.death != np.inf:
                barcodes["torsion_im1jm1"].append(tim1jm1)
            else:
                barcodes["torsion_im1jm1"].append(None)
            barcodes["torsion_im1j"].append(tim1j)
    
    out_df = pd.DataFrame(barcodes)
    out_df["Weight"] = out_df.apply(lambda x: list_minus(list_minus(x["torsion_ij"], x["torsion_ijm1"]), list_minus(x["torsion_im1j"], x["torsion_im1jm1"])), axis=1)
    return out_df

def complete_analysis_Ren(f: d.Filtration, weight_fun, max_dim = 1):
    f.sort()
    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    output = {q: {s: {t: {} for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}

    barcodes = {"q": [], "birth": [], "death": [], "multiplicity": [], 
                "torsion_ijm1": [], "torsion_ij": [], "torsion_im1jm1": [], "torsion_im1j": []}
    
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)))
        for t_i in t_i_bar:
            for s_i in range(t_i+1):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                betti, torsion = method_Ren(q, s, t, boundary_matrices, simplices_at_time)
                # Lap = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
                
                output[q][s][t]["betti"] = betti
                output[q][s][t]["torsion"] = torsion
    
    for q in range(max_dim+1):
        for t_i in range(1,len(relevant_times)):
            for s_i in range(t_i):
                s, t, tm1 = relevant_times[s_i], relevant_times[t_i], relevant_times[t_i-1] 

                if s_i != 0:
                    sm1 = relevant_times[s_i-1]
                
                bijm1 = output[q][s][tm1]["betti"]
                tijm1 = output[q][s][tm1]["torsion"]

                bij = output[q][s][t]["betti"]
                tij = output[q][s][t]["torsion"]

                if s_i > 0:
                    bim1jm1 = output[q][sm1][tm1]["betti"]
                    tim1jm1 = output[q][sm1][tm1]["torsion"]

                    bim1j = output[q][sm1][t]["betti"]
                    tim1j = output[q][sm1][t]["torsion"]
                else:
                    tim1jm1, bim1jm1, tim1j, bim1j = [], 0, [], 0

                
                if bijm1-bij-(bim1jm1-bim1j) > 0:
                    barcodes["q"].append(q)
                    barcodes["birth"].append(s)
                    barcodes["death"].append(t)
                    barcodes["multiplicity"].append(bijm1-bij-(bim1jm1-bim1j))
                    barcodes["torsion_ijm1"].append(tijm1)
                    barcodes["torsion_ij"].append(tij)
                    barcodes["torsion_im1jm1"].append(tim1jm1)
                    barcodes["torsion_im1j"].append(tim1j)

                if t == max_time and bij-bim1j > 0:
                    barcodes["q"].append(q)
                    barcodes["birth"].append(s)
                    barcodes["death"].append(np.inf)
                    barcodes["multiplicity"].append(bij-bim1j)
                    barcodes["torsion_ijm1"].append(tijm1)
                    barcodes["torsion_ij"].append(tij)
                    barcodes["torsion_im1jm1"].append(tim1jm1)
                    barcodes["torsion_im1j"].append(tim1j)
    
    out_df = pd.DataFrame(barcodes)
    out_df["Weight"] = out_df.apply(lambda x: list_minus(list_minus(x["torsion_ij"], x["torsion_ijm1"]), list_minus(x["torsion_im1j"], x["torsion_im1jm1"])), axis=1)
    return out_df