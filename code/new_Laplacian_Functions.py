from multiprocessing import Value
from re import A, I
from matplotlib.pylab import LinAlgError
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

def vertical_Laplacian_projection(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=False):
    t, s = relevant_times[t_i], relevant_times[s_i]
    tm1 = relevant_times[t_i-1]

    Lap_s = persistent_Laplacian_filtration(q, boundary_matrices, s, t, simplices_at_time)
    if s_i > 0:
        sm1 = relevant_times[s_i-1]
        Lap_sm1 = np.eye(simplices_at_time(s)[q])
        Lap_sm1[:simplices_at_time(sm1)[q], :simplices_at_time(sm1)[q]] = persistent_Laplacian_filtration(q, boundary_matrices, sm1, t, simplices_at_time)
        return (np.eye(simplices_at_time(s)[q]) - np.linalg.pinv(Lap_s)@Lap_s)@(np.linalg.pinv(Lap_sm1)@Lap_sm1)
    return Lap_s
        

def vertical_Laplacian_extended(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=False):
    t, s = relevant_times[t_i], relevant_times[s_i]
    tm1 = relevant_times[t_i-1]
    # if t_i < relevant_times[-1]:
    #     return np.zeros((0,0))

    if simplices_at_time(s)[q] == 0:
        return np.zeros((0,0))
    # If no new q+1 simplices, then it is just 0
    # if simplices_at_time(t)[q+1] == simplices_at_time(tm1)[q+1]:
    #     return np.zeros((simplices_at_time(s)[q], simplices_at_time(s)[q]))

    if verb:
        print(f"Bqplus1:\n{boundary_matrices[q+1]}")
        print(f"n_q_t:{simplices_at_time(t)}, n_q_s: {simplices_at_time(s)}")
    if s_i > 0:
        sm1 = relevant_times[s_i-1]
        B12_sm1t = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
        # if B12_sm1t.shape[1] == 0 or B12_sm1t.shape[0] == 0:
        #     B12_sm1t = np.zeros((max(simplices_at_time(t)[q]-simplices_at_time(sm1)[q],1), 1))

        B22_sm1t = boundary_matrices[q+1][simplices_at_time(sm1)[q]:simplices_at_time(t)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]
        # if B22_sm1t.shape[0] == 0:
        #     # NOTE: max cannot be required as something happens between t and t-1!
        #     B22_sm1t = np.zeros((1,max(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1],1)))

        B22_sm1tm1 = boundary_matrices[q+1][simplices_at_time(sm1)[q]:simplices_at_time(tm1)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(tm1)[q+1]]

        # if B22_sm1tm1.shape[0] == 0:
        #     B22_sm1tm1 = np.zeros((1,simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1]))
            # ker_B22_sm1tm1 = scipy.linalg.null_space(B22_sm1tm1)
        
        A_matrix = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]
    else:
        Bs_qp1 = boundary_matrices[q+1][:simplices_at_time(s)[q], :simplices_at_time(s)[q+1]]
    
    B12_st = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
    # if B12_st.shape[1] == 0 or B12_st.shape[0] == 0:
    #     B12_st = np.zeros((max(simplices_at_time(t)[q]-simplices_at_time(s)[q],1), 1))

    B22_st = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
    # if B22_st.shape[0] == 0:
    #     B22_st = np.zeros((1,simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1]))

    B22_stm1 = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(tm1)[q], simplices_at_time(s)[q+1]:simplices_at_time(tm1)[q+1]]
    # if B22_stm1.shape[0] == 0:
    #     B22_stm1 = np.zeros((1,simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]))
    if q > 0:
        Bs_q = boundary_matrices[q][:simplices_at_time(s)[q-1], :simplices_at_time(s)[q]]
        Down_Lap = Bs_q.T@Bs_q

        if s_i > 0:
            # This corresponds to subtracting the two down Laplacians.
            # Down_Lap[:simplices_at_time(sm1)[q], :simplices_at_time(sm1)[q]] = 0

            # This corresponds to older cycles go to 0, while new ones go to 1
            # Bsm1_q = boundary_matrices[q][:simplices_at_time(sm1)[q-1], :simplices_at_time(sm1)[q]]
            # down_sm1_part = np.eye(simplices_at_time(s)[q])
            # down_sm1_part[:simplices_at_time(sm1)[q], :simplices_at_time(sm1)[q]] = np.linalg.pinv(Bsm1_q)@Bsm1_q
            # Down_Lap -= down_sm1_part

            Bsm1_q = boundary_matrices[q][:simplices_at_time(sm1)[q-1], :simplices_at_time(sm1)[q]]
            down_sm1_part = np.eye(simplices_at_time(s)[q])
            down_sm1_part[:simplices_at_time(sm1)[q], :simplices_at_time(sm1)[q]] = np.linalg.pinv(Bsm1_q)@Bsm1_q

            down_s_part = np.linalg.pinv(Bs_q)@Bs_q

            Down_Lap = down_sm1_part@(np.eye(simplices_at_time(s)[q])-down_s_part)@down_sm1_part
            # Down_Lap = down_sm1_part-down_s_part

        if verb:
            print(f"Down Lap:\n{Down_Lap}")
            evals, evecs = np.linalg.eigh(Down_Lap)
            for i, eval in enumerate(evals):
                evec = evecs[:,i]
                print(f"eval: {np.round(eval, 4)}, evec: {np.round(evec/np.min(np.abs(evec[np.abs(evec) > 1e-8])),5)}")

        
    else:
        Down_Lap = np.zeros((simplices_at_time(s)[q],simplices_at_time(s)[q]))
        Bs_q = np.zeros((simplices_at_time(s)[q-1], simplices_at_time(s)[q]))


    if verb:
        print(s_i, t_i)
        print(f"B12_st:\n{B12_st}")
        print(f"B22_st:\n{B22_st}")
        print(f"B22_stm1:\n{B22_stm1}")
        if s_i > 0:
            print(f"B22_sm1t:\n{B22_sm1t}")
            print(f"B22_sm1tm1:\n{B22_sm1tm1}")
    if s_i > 0:
        B22_ext = np.zeros((simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1], simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1]))
        B22_ext[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):, (simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = np.linalg.pinv(B22_st)@B22_st
        B22_sm1_ext = np.linalg.pinv(B22_sm1t)@B22_sm1t
        up_Lap = A_matrix@B22_sm1_ext@(np.eye(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1])-B22_ext)@B22_sm1_ext@A_matrix.T
        if verb:
            print(f"up_Lap:\n{up_Lap}")
            evals, evecs = np.linalg.eigh(up_Lap)
            for i, eval in enumerate(evals):
                evec = evecs[:,i]
                print(f"eval: {np.round(eval, 4)}, evec: {np.round(evec/np.min(np.abs(evec[np.abs(evec) > 1e-8])),5)}")
        # return up_Lap + Down_Lap
        up_proj = np.linalg.pinv(up_Lap)@up_Lap
        if type(Down_Lap) != int:
            return (np.eye(simplices_at_time(s)[q])-up_proj)@Down_Lap@(np.eye(simplices_at_time(s)[q])-up_proj)
        return up_Lap
    
    if verb:
        print(f"B22 inv:\n{np.linalg.pinv(B22_st)}")
        print(f"B22 part:\n{np.linalg.pinv(B22_st)@B22_st}")
        print(f"Eye only:\n{np.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1])}")
        print(f"Eye part:\n{np.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1])-np.linalg.pinv(B22_st)@B22_st}")
        print(f"Full part:\n{B12_st@(np.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1])-np.linalg.pinv(B22_st)@B22_st)@B12_st.T}")
        print("-"*20)
    return Bs_qp1@Bs_qp1.T + B12_st@(np.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1])-np.linalg.pinv(B22_st)@B22_st)@B12_st.T + Down_Lap

def cross_Laplacian_extended(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=False, Laplacian_fun = None):
    t, s = relevant_times[t_i], relevant_times[s_i]
    tm1 = relevant_times[t_i-1]

    # If no new q+1 simplices, then it is just 0
    # if simplices_at_time(t)[q+1] == simplices_at_time(tm1)[q+1]:
    #     return np.zeros((simplices_at_time(s)[q], simplices_at_time(s)[q]))

    if verb:
        print(f"Bqplus1:\n{boundary_matrices[q+1]}")
        print(f"n_q_t:{simplices_at_time(t)}, n_q_s: {simplices_at_time(s)}")
    if s_i > 0:
        sm1 = relevant_times[s_i-1]
        B12_sm1t = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
        # if B12_sm1t.shape[1] == 0 or B12_sm1t.shape[0] == 0:
        #     B12_sm1t = np.zeros((max(simplices_at_time(t)[q]-simplices_at_time(sm1)[q],1), 1))

        B22_sm1t = boundary_matrices[q+1][simplices_at_time(sm1)[q]:simplices_at_time(t)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]
        # if B22_sm1t.shape[0] == 0:
        #     # NOTE: max cannot be required as something happens between t and t-1!
        #     B22_sm1t = np.zeros((1,max(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1],1)))

        B22_sm1tm1 = boundary_matrices[q+1][simplices_at_time(sm1)[q]:simplices_at_time(tm1)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(tm1)[q+1]]
        # if B22_sm1tm1.shape[0] == 0:
        #     B22_sm1tm1 = np.zeros((1,simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1]))
            # ker_B22_sm1tm1 = scipy.linalg.null_space(B22_sm1tm1)
        
        QR_matrix = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(sm1)[q+1]:simplices_at_time(t)[q+1]]
    
    B12_st = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
    # if B12_st.shape[1] == 0 or B12_st.shape[0] == 0:
    #     B12_st = np.zeros((max(simplices_at_time(t)[q]-simplices_at_time(s)[q],1), 1))

    B22_st = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
    # if B22_st.shape[0] == 0:
    #     B22_st = np.zeros((1,simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1]))

    B22_stm1 = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(tm1)[q], simplices_at_time(s)[q+1]:simplices_at_time(tm1)[q+1]]
    # if B22_stm1.shape[0] == 0:
    #     B22_stm1 = np.zeros((1,simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]))
    
    if verb:
        print(f"B12_st:\n{B12_st}")
        print(f"B22_st:\n{B22_st}")
        print(f"B22_stm1:\n{B22_stm1}")
        if s_i > 0:
            print(f"B22_sm1t:\n{B22_sm1t}")
            print(f"B22_sm1tm1:\n{B22_sm1tm1}")
    
    if s_i > 0:
        eye_sm1t = np.eye(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1])
        B22_sm1t_full = np.linalg.pinv(B22_sm1t)@B22_sm1t

        B22_st_full = np.zeros_like(eye_sm1t)
        B22_st_full[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = np.linalg.pinv(B22_st)@B22_st

        B22_stm1_full = np.zeros_like(eye_sm1t)
        B22_stm1_partial_full = np.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1])
        B22_stm1_partial_full[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = np.linalg.pinv(B22_stm1)@B22_stm1
        B22_stm1_full[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = B22_stm1_partial_full

        B22_sm1tm1_full = deepcopy(eye_sm1t)
        B22_sm1tm1_full[:(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1])] = np.linalg.pinv(B22_sm1tm1)@B22_sm1tm1

        # First vertical then horizontal
        # return -1*QR_matrix@((eye_sm1t-B22_sm1t_full)@B22_sm1tm1_full-(eye_sm1t-B22_st_full)@B22_stm1_full)@QR_matrix.T

        # First horizontal then vertical
        # return QR_matrix@((eye_sm1t-B22_st_full)@B22_sm1t_full-(eye_sm1t-B22_stm1_full)@B22_sm1tm1_full)@QR_matrix.T

        # Less multiplications
        # return QR_matrix@(B22_sm1tm1_full - B22_sm1t_full - B22_stm1_full + B22_st_full)@QR_matrix.T

        # s_part = (eye_sm1t-B22_st_full)@B22_stm1_full
        # sm1_part = (eye_sm1t-B22_sm1t_full)@B22_sm1tm1_full

        s_part = (B22_stm1_full-B22_st_full)
        sm1_part = (B22_sm1tm1_full-B22_sm1t_full)
        B22_mult_tm1sm1 = B22_stm1_full@B22_sm1t_full
        B22_mult_sm1tm1 = B22_sm1t_full@B22_stm1_full
        if verb:
            print(f"A_matrix:\n{QR_matrix}")
            print(f"B22_st_full:\n{np.round(B22_st_full,5)}")
            print(f"B22_stm1_full:\n{np.round(B22_stm1_full,5)}")
            print(f"B22_sm1t_full:\n{np.round(B22_sm1t_full,5)}")
            print(f"B22_sm1tm1_full:\n{np.round(B22_sm1tm1_full,5)}")
            print(f"B22_stm1_full@B22_sm1t_full:\n{np.round(B22_stm1_full@B22_sm1t_full,5)}")
            print(f"A_matrix@B22_stm1_full@B22_sm1t_full@A_matrix^T:\n{np.round(QR_matrix@(B22_stm1_full@B22_sm1t_full)@QR_matrix.T, 5)}")
            print(f"A_matrix@B22_sm1t_full@B22_stm1_full@A_matrix^T:\n{np.round(QR_matrix@(B22_sm1t_full@B22_stm1_full)@QR_matrix.T, 5)}")
        
        # first_projection = B22_stm1_full@(eye_sm1t-B22_st_full)@np.linalg.pinv(B22_stm1_full@(eye_sm1t-B22_st_full))
        # return QR_matrix@(B22_sm1t_full@first_projection@np.linalg.pinv(B22_sm1t_full@first_projection))@QR_matrix.T
    
        first_projection = eye_sm1t - (2*eye_sm1t - B22_sm1t_full - B22_stm1_full)@np.linalg.pinv(2*eye_sm1t - B22_sm1t_full - B22_stm1_full)
        Pa = (eye_sm1t-B22_st_full)
        Pb = first_projection
        # return QR_matrix@(Pa-Pa@np.linalg.pinv(Pa@(eye_sm1t-Pb)))@QR_matrix.T

        if verb:
            # VS = scipy.linalg.null_space(B22_st_full)
            # print(f"VS ({VS.shape}):\n{VS}")
            # G = VS.T@QR_matrix.T@QR_matrix@VS
            # print(f"G:\n{np.round(G, 5)}")
            # P1 = VS.T@B22_stm1_full@VS
            # P2 = VS.T@B22_sm1t_full@VS

            # Test_m1 = G@P1@P2
            Test_m1 = np.round(B22_stm1_full@(eye_sm1t-B22_st_full), 6)
            print(f"Test Matrix 1:\n{np.round(Test_m1, 4)}")
            evals, evecs = np.linalg.eigh(np.round(Test_m1, 4))
            ker_B22 = []
            for i, eval in enumerate(evals.real):
                if eval > 1e-8:
                    ker_B22.append(evecs[:,i].real)
                # if np.abs(eval) > 1e-8:
                print("eval:", eval)
                evec = evecs[:,i].real
                print("normalized evec", np.round(evec, 4))
                print("denormalized evec",np.round(evec/np.min(np.abs(evec[np.abs(evec) > 1e-8])),4))
                print()
            
            for v in ker_B22:
                print("v:", np.round(v,4))
                print("Av:", np.round(QR_matrix@v,4))
                print("AP^s-1,tv:", np.round(QR_matrix@B22_sm1t_full@v,4))
                print()
                # print("ori calc:", np.linalg.norm(QR_matrix@B22_sm1tm1_full@v)**2 - np.linalg.norm(QR_matrix@B22_sm1t_full@v)**2 - np.linalg.norm(QR_matrix@B22_stm1_full@v)**2)
                # print("new calc:", (QR_matrix@B22_stm1_full@v).T@(QR_matrix@B22_sm1t_full@v))
                # print("newest calc:",0.5*(np.linalg.norm(QR_matrix@(B22_sm1t_full + B22_stm1_full)@v)**2 - np.linalg.norm(QR_matrix@B22_sm1t_full@v)**2 - np.linalg.norm(QR_matrix@B22_stm1_full@v)**2))
                # print("final calc:",0.5*(-1*np.linalg.norm(QR_matrix@(B22_sm1t_full - B22_stm1_full)@v)**2 + np.linalg.norm(QR_matrix@B22_sm1t_full@v)**2 + np.linalg.norm(QR_matrix@B22_stm1_full@v)**2))

                # print("left side:", np.round(np.linalg.norm(QR_matrix@B22_sm1tm1_full@v)**2 - 0.5*np.linalg.norm(QR_matrix@(B22_sm1t_full - B22_stm1_full)@v)**2, 5))
                # print("right side:", np.round(0.5*(np.linalg.norm(QR_matrix@B22_sm1t_full@v)**2 + np.linalg.norm(QR_matrix@B22_stm1_full@v)**2), 5))
                # print("sm1tm1:", np.round(np.linalg.norm(QR_matrix@B22_sm1tm1_full@v)**2, 5))
                # print("sm1t:", np.round(np.linalg.norm(QR_matrix@B22_sm1t_full@v)**2, 5))
                # print("stm1:", np.round(np.linalg.norm(QR_matrix@B22_stm1_full@v)**2, 5))
                # print("sm1 - tm1:", np.round(0.5*np.linalg.norm(QR_matrix@(B22_sm1t_full - B22_stm1_full)@v)**2, 5))
                # print("sm1 + tm1:", np.round(0.5*np.linalg.norm(QR_matrix@(B22_sm1t_full + B22_stm1_full)@v)**2, 5))
            # Test_m2 = B22_sm1t_full@(eye_sm1t-B22_st_full)@B22_sm1t_full@B22_stm1_full
            # print(f"Test Matrix 2:\n{np.round(Test_m2, 5)}")
            # evals, evecs = np.linalg.eig(np.round(Test_m2, 5))
            # for i, eval in enumerate(evals):
            #     # if np.abs(eval) > 1e-8:
            #     print("eval:", eval)
            #     evec = evecs[:,i]
            #     print("normalized evec", np.round(evec, 5))
            #     print("denormalized evec",np.round(evec/np.min(np.abs(evec[np.abs(evec) > 1e-8])),5))
            #     print()

            # Test_m3 = B22_stm1_full@B22_sm1t_full@(eye_sm1t-B22_st_full)@B22_sm1t_full
            # print(f"Test Matrix:\n{np.round(Test_m3, 5)}")
            # evals, evecs = np.linalg.eig(np.round(Test_m3, 5))
            # for i, eval in enumerate(evals):
            #     # if np.abs(eval) > 1e-8:
            #     print("eval:", eval)
            #     evec = evecs[:,i]
            #     print("normalized evec", np.round(evec, 5))
            #     print("denormalized evec",np.round(evec/np.min(np.abs(evec[np.abs(evec) > 1e-8])),5))
            #     print()



        if Laplacian_fun is not None:
            return QR_matrix@Laplacian_fun(B22_st_full, B22_stm1_full, B22_sm1t_full, B22_sm1tm1_full, eye_sm1t)@QR_matrix.T

        # Seems to work best
        return QR_matrix@(B22_sm1t_full@B22_stm1_full@(eye_sm1t-B22_st_full)@B22_stm1_full)@QR_matrix.T

        # return QR_matrix@s_part@(eye_sm1t - sm1_part)@s_part@QR_matrix.T
        # return QR_matrix@(s_part - s_part@np.linalg.pinv(s_part)@sm1_part)@QR_matrix.T
        # return QR_matrix@(s_part + (B22_stm1_full@B22_sm1t_full-B22_stm1_full-B22_st_full+B22_st_full@B22_sm1tm1_full)@s_part)@QR_matrix.T
        # return QR_matrix@(s_part + (B22_stm1_full@B22_sm1t_full-B22_stm1_full-B22_st_full+B22_st_full@B22_sm1tm1_full)@s_part)@QR_matrix.T
        # return QR_matrix@(B22_sm1t_full@B22_stm1_full@B22_sm1t_full-B22_st_full)@QR_matrix.T
        # return QR_matrix@(B22_stm1_full@B22_sm1t_full@(eye_sm1t-B22_st_full)@B22_sm1t_full@B22_stm1_full)@QR_matrix.T
        
        # return QR_matrix@(B22_sm1t_full@B22_stm1_full@(eye_sm1t-B22_st_full)@B22_sm1t_full@B22_stm1_full)@QR_matrix.T
        # return QR_matrix@(B22_sm1t_full@B22_stm1_full@(eye_sm1t-B22_st_full)@B22_stm1_full@B22_sm1t_full+\
        #                   B22_stm1_full@B22_sm1t_full@(eye_sm1t-B22_st_full)@B22_stm1_full@B22_sm1t_full+\
        #                   B22_sm1t_full@B22_stm1_full@(eye_sm1t-B22_st_full)@B22_sm1t_full@B22_stm1_full+\
        #                   B22_stm1_full@B22_sm1t_full@(eye_sm1t-B22_st_full)@B22_sm1t_full@B22_stm1_full)@QR_matrix.T

        # Q = B22_stm1_full-B22_st_full
        # P = B22_sm1tm1_full@(eye_sm1t-B22_sm1t_full)
        Q = B22_sm1t_full - B22_st_full
        # P = B22_stm1_full
        P = B22_sm1tm1_full@(eye_sm1t-B22_stm1_full)
        # P = B22_sm1t_full

        if verb:
            print(f"Q\n{np.round(Q, 5)}")
            evals, evecs = np.linalg.eig(np.round(Q, 5))
            for i, eval in enumerate(evals):
                # if np.abs(eval) > 1e-8:
                print("eval:", eval)
                evec = evecs[:,i]
                print("normalized evec", np.round(evec, 5))
                print("denormalized evec",np.round(evec/np.min(np.abs(evec[np.abs(evec) > 1e-8])),5))
                print()


            print(f"P\n{np.round(P, 5)}")
            evals, evecs = np.linalg.eig(np.round(P, 5))
            for i, eval in enumerate(evals):
                # if np.abs(eval) > 1e-8:
                print("eval:", eval)
                evec = evecs[:,i]
                print("normalized evec", np.round(evec, 5))
                print("denormalized evec",np.round(evec/np.min(np.abs(evec[np.abs(evec) > 1e-8])),5))
                print()
        

        # P_intersect = 2*Q@np.linalg.pinv(Q+P)@P
        # P_intersect = Q@P@np.linalg.pinv(Q@P)
        # P_intersect = B22_sm1tm1_full-B22_st_full
        P_intersect = eye_sm1t - B22_st_full - (eye_sm1t - B22_st_full)@(eye_sm1t - B22_stm1_full) - (eye_sm1t - B22_st_full)@(eye_sm1t - B22_sm1t_full) + (eye_sm1t - B22_st_full)@(eye_sm1t - B22_sm1tm1_full)
        return QR_matrix@P_intersect@QR_matrix.T
        

        s_part_QR = QR_matrix@s_part@QR_matrix.T
        sm1_part_QR = QR_matrix@sm1_part@QR_matrix.T
        s_QR_basis = scipy.linalg.orth(np.round(s_part_QR))
        sm1_QR_basis = scipy.linalg.orth(np.round(sm1_part_QR, 15))
        if verb:
            print(f"s_QR_basis\n{s_QR_basis}")
            print(f"sm1_QR_basis\n{sm1_QR_basis}")
        # null_part = s_part@scipy.linalg.null_space(eye_sm1t-sm1_part)
        # null_part = null_part@null_part.T
        # print(f"null_space:\n{null_part}")

        sm1_projected = s_part@(eye_sm1t - sm1_part)@s_part
        # sm1_projected = s_part - null_part
        # sm1_projected = np.linalg.pinv(sm1_projected)@sm1_projected
        if verb:
            print(f"s_part\n{np.round(s_part, 5)}")
            print(f"sm1_part\n{np.round(sm1_part, 5)}")
            print(f"sm1_proj\n{np.round(sm1_projected, 5)}")
            print(f"QR s_part\n{np.round(s_part_QR, 5)}")
            print(f"QR sm1_part\n{np.round(sm1_part_QR, 5)}")
            print(f"QR sm1_proj\n{np.round(QR_matrix@sm1_projected, 5)}")
            print(f"new_part:\n{np.linalg.pinv(sm1_part_QR)@sm1_part_QR@np.linalg.pinv(s_part_QR)@s_part_QR}")

        return s_part_QR - np.linalg.pinv(sm1_part_QR)@sm1_part_QR@np.linalg.pinv(s_part_QR)@s_part_QR
        return QR_matrix@(sm1_projected)@QR_matrix.T
    
    eye_st = np.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1])

    B22_st_full = np.linalg.pinv(B22_st)@B22_st

    B22_stm1_full = deepcopy(eye_st)
    B22_stm1_full[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = np.linalg.pinv(B22_stm1)@B22_stm1

    return B12_st@(eye_st-B22_st_full)@B22_stm1_full@B12_st.T
    



def cross_Laplacian_new_eigenvalues(f: d.Filtration, weight_fun, max_dim = 1, method = "cor10", Laplacian_fun = None):
    f.sort()

    if method == "cor10":
        cross_Laplacian_f = cross_Laplacian_cor10
    elif method == "vertical_extended":
        cross_Laplacian_f = vertical_Laplacian_extended
    elif method == "cross_extended":
        cross_Laplacian_f = cross_Laplacian_extended
    elif method == "vertical_projection":
        cross_Laplacian_f = vertical_Laplacian_projection

    max_time = f[len(f)-1].data
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)), leave=False)
        for t_i in t_i_bar:
            if method == "vertical_extended":
                s_i_bar = range(t_i+1)
            else:
                s_i_bar = range(t_i)
            for s_i in s_i_bar:
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                try:
                    if method == "cross_extended":
                        Lap = cross_Laplacian_f(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times=relevant_times, Laplacian_fun=Laplacian_fun)
                    else:
                        Lap = cross_Laplacian_f(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times=relevant_times)
                except:
                    print(s,t)
                    if method == "cross_extended":
                        Lap = cross_Laplacian_f(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times=relevant_times, Laplacian_fun=Laplacian_fun, verb=True)
                    else:
                        Lap = cross_Laplacian_f(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times=relevant_times, verb=True)
                    raise ValueError
                
                eigenvalues[q][s][t] = np.linalg.eigvals(Lap).real
                if np.any(eigenvalues[q][s][t].real != eigenvalues[q][s][t]):
                    print(f"q: {q}, s: {s}, t: {t}, evals: {eigenvalues[q][s][t]}, Lap:\n{Lap}")
    return eigenvalues, relevant_times
    
def cross_Laplaican_eigenvalues_fast(f: d.Filtration, weight_fun = lambda x: 1, max_dim = 1, Laplacian_fun = None, use_greville = False):
    f.sort()
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    projection_matrices = {q: {s: {t: np.array([]) for t in range(len(relevant_times))} for s in  range(len(relevant_times))} for q in range(max_dim+1)}

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
                        projection_matrices[q][s_i][t_i] = np.linalg.pinv(B22)@B22
                    except LinAlgError:
                        print("RAN INTO ERROR !!!")
                        print(f"s: {s}, t: {t}")

                        projection_matrices[q][s_i][t_i] = np.eye(simplices_at_time(t)[q+1] - simplices_at_time(s)[q+1])
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
                            projection_matrices[q][s_i][t_i] = np.zeros((0,0))
                        elif simplices_at_time(t)[q] == simplices_at_time(s)[q]:
                            projection_matrices[q][s_i][t_i] = np.zeros((simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1], simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1]))
                        else:
                            A_km1 = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
                            A_km1_pinv = np.linalg.pinv(A_km1)
                            projection_matrices[q][s_i][t_i] = A_km1_pinv@A_km1
                            t_A_km1 = t
                            obtained_B22 = True
                    else:
                        changed_B22 = False
                        if simplices_at_time(t)[q] > simplices_at_time(t_A_km1)[q]:
                            # NOTE: Not necessary, can also just use boundary matrices to get it. But this makes it more clear.
                            A_km1_new = np.zeros((simplices_at_time(t)[q]-simplices_at_time(s)[q], simplices_at_time(t_A_km1)[q+1]-simplices_at_time(s)[q+1]))
                            A_km1_new[:A_km1.shape[0], :A_km1.shape[1]] = A_km1
                            A_km1 = A_km1_new

                            A_km1_pinv_new = np.zeros((simplices_at_time(t_A_km1)[q+1]-simplices_at_time(s)[q+1], simplices_at_time(t)[q]-simplices_at_time(s)[q]))
                            A_km1_pinv_new[:A_km1_pinv.shape[0], :A_km1_pinv.shape[1]] = A_km1_pinv
                            A_km1_pinv = A_km1_pinv_new

                            changed_B22 = True
                        
                        if simplices_at_time(t)[q+1] > simplices_at_time(t_A_km1)[q+1]:
                            # Use the update rule from Greville.
                            for col in range(simplices_at_time(t_A_km1)[q+1], simplices_at_time(t)[q+1]):
                                a_k = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], [col]]

                                d_k = A_km1_pinv@a_k
                                c_k = a_k - A_km1@d_k

                                norm_c_k_sq = np.linalg.norm(c_k)**(2)
                                if norm_c_k_sq > 1e-10:
                                    # b_k = np.linalg.pinv(c_k)
                                    b_k = c_k.T/norm_c_k_sq
                                else:
                                    b_k = (1 + d_k.T@d_k)**(-1)*d_k.T@A_km1_pinv
                                A_km1_pinv = np.vstack((A_km1_pinv-d_k@b_k, b_k))
                                A_km1 = np.hstack((A_km1, a_k))
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

                    eye = np.eye(simplices_at_time(t)[q+1]-simplices_at_time(sm1)[q+1])
                    B22_sm1t = projection_matrices[q][s_i-1][t_i]

                    B22_st = np.zeros_like(eye)
                    B22_st[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = projection_matrices[q][s_i][t_i]

                    B22_stm1 = np.zeros_like(eye)
                    B22_stm1_partial = np.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1])
                    B22_stm1_partial[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = projection_matrices[q][s_i][t_i-1]
                    B22_stm1[(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):,(simplices_at_time(s)[q+1]-simplices_at_time(sm1)[q+1]):] = B22_stm1_partial

                    B22_sm1tm1 = deepcopy(eye)
                    B22_sm1tm1[:(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(sm1)[q+1])] = projection_matrices[q][s_i-1][t_i-1]

                    cross_Lap = A_matrix@Laplacian_fun(B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye)@A_matrix.T
                    # cross_Lap = Laplacian_fun(B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye)

                    eigenvalues[q][s][t] = np.linalg.eigvals(cross_Lap).real
                
                else:
                    B12_st = boundary_matrices[q+1][:simplices_at_time(s)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]

                    eye = np.eye(simplices_at_time(t)[q+1]-simplices_at_time(s)[q+1])

                    B22_st = projection_matrices[q][s_i][t_i]

                    B22_stm1 = deepcopy(eye)
                    B22_stm1[:(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1]), :(simplices_at_time(tm1)[q+1]-simplices_at_time(s)[q+1])] = projection_matrices[q][s_i][t_i-1]

                    cross_Lap = B12_st@(B22_stm1 - B22_st)@B12_st.T

                    eigenvalues[q][s][t] = np.linalg.eigvals(cross_Lap).real

    return eigenvalues, relevant_times





def calc_cor10(f: d.Filtration, q, s, t, weight_fun = lambda x: 1, verb=False):
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    relevant_times = np.array(relevant_times)
    t_i = np.argmin(np.abs(relevant_times - t))
    s_i = np.argmin(np.abs(relevant_times - s))
    return cross_Laplacian_cor10(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=verb)

def calc_cross_extended(f: d.Filtration, q, s, t, weight_fun = lambda x: 1, verb=False, Laplacian_fun = None):
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    relevant_times = np.array(relevant_times)
    t_i = np.argmin(np.abs(relevant_times - t))
    s_i = np.argmin(np.abs(relevant_times - s))
    return cross_Laplacian_extended(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=verb, Laplacian_fun= Laplacian_fun)

def calc_vertical_extended(f: d.Filtration, q, s, t, weight_fun = lambda x: 1, verb=False):
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    relevant_times = np.array(relevant_times)
    t_i = np.argmin(np.abs(relevant_times - t))
    s_i = np.argmin(np.abs(relevant_times - s))
    return vertical_Laplacian_extended(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=verb)

def plot_Laplacian_new_eigenvalues(f: d.Filtration, weight_fun, max_dim = 1, plot_types = "all", method = "cor10", Laplacian_fun = None, integer_time_steps = False,
                     plot_args_mesh = {}, 
                     plot_args_diag = {},
                     plot_args_line = {},
                     plot_type_to_fun = {}):
    """
    lapalcian_type: "persistent" for normal laplacian, or "cross" for cross laplacian, "cross_cor10" for cross Laplacian in two directions.
    """
    if method not in ["fast", "greville"]:
        eigenvalues, relevant_times = cross_Laplacian_new_eigenvalues(f, weight_fun, max_dim=max_dim, method = method)
    else:
        if method == "fast":
            eigenvalues, relevant_times = cross_Laplaican_eigenvalues_fast(f, weight_fun=weight_fun, max_dim=max_dim, Laplacian_fun=Laplacian_fun)
        else:
            eigenvalues, relevant_times = cross_Laplaican_eigenvalues_fast(f, weight_fun=weight_fun, max_dim=max_dim, Laplacian_fun=Laplacian_fun, use_greville=True)
        

    fig, ax = plot_eigenvalues(eigenvalues, relevant_times, plot_types=plot_types, filtration=f, integer_time_steps=integer_time_steps,
                               plot_args_mesh = plot_args_mesh, plot_args_diag=plot_args_diag,
                               plot_args_line=plot_args_line, plot_type_to_fun=plot_type_to_fun)
    return eigenvalues, relevant_times, fig, ax

