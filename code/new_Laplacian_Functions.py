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

def cross_Laplacian_extended(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=False):
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
    
    print("check:", boundary_matrices[q+1])
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
            print(f"B22_mult_tm1sm1:\n{np.round(B22_mult_tm1sm1,5)}")
            print(f"B22_mult_tm1sm1^2:\n{np.round(B22_mult_tm1sm1@B22_mult_tm1sm1,5)}")
            print(f"B22_mult_sm1tm1:\n{np.round(B22_mult_sm1tm1,5)}")
            print(f"B22_mult_sm1tm1^2:\n{np.round(B22_mult_sm1tm1@B22_mult_sm1tm1,5)}")
        
        # Seems to work best
        return QR_matrix@(B22_sm1t_full@B22_stm1_full@(eye_sm1t-B22_st_full)@B22_stm1_full@B22_sm1t_full)@QR_matrix.T

        # return QR_matrix@s_part@(eye_sm1t - sm1_part)@s_part@QR_matrix.T
        # return QR_matrix@(s_part + (B22_stm1_full@B22_sm1t_full-B22_stm1_full-B22_st_full+B22_st_full@B22_sm1tm1_full)@s_part)@QR_matrix.T
        # return QR_matrix@(s_part + (B22_stm1_full@B22_sm1t_full-B22_stm1_full-B22_st_full+B22_st_full@B22_sm1tm1_full)@s_part)@QR_matrix.T
        # return QR_matrix@(B22_sm1t_full@B22_stm1_full@B22_sm1t_full-B22_st_full)@QR_matrix.T
        # return QR_matrix@(B22_stm1_full@B22_sm1t_full@(eye_sm1t-B22_st_full)@B22_sm1t_full@B22_stm1_full)@QR_matrix.T
        
        # return QR_matrix@(B22_sm1t_full@B22_stm1_full@(eye_sm1t-B22_st_full)@B22_sm1t_full@B22_stm1_full)@QR_matrix.T

    
        

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
    



def cross_Laplacian_new_eigenvalues(f: d.Filtration, weight_fun, max_dim = 1, method = "cor10"):
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
                    Lap = cross_Laplacian_f(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times=relevant_times)
                except:
                    print(s,t)
                    Lap = cross_Laplacian_f(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times=relevant_times, verb=True)
                    raise ValueError
                
                eigenvalues[q][s][t] = np.linalg.eigvalsh(Lap)
    return eigenvalues, relevant_times
    

def calc_cor10(f: d.Filtration, q, s, t, weight_fun = lambda x: 1, verb=False):
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    relevant_times = np.array(relevant_times)
    t_i = np.argmin(np.abs(relevant_times - t))
    s_i = np.argmin(np.abs(relevant_times - s))
    return cross_Laplacian_cor10(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=verb)

def calc_cross_extended(f: d.Filtration, q, s, t, weight_fun = lambda x: 1, verb=False):
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    relevant_times = np.array(relevant_times)
    t_i = np.argmin(np.abs(relevant_times - t))
    s_i = np.argmin(np.abs(relevant_times - s))
    return cross_Laplacian_extended(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=verb)

def calc_vertical_extended(f: d.Filtration, q, s, t, weight_fun = lambda x: 1, verb=False):
    boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)
    relevant_times = np.array(relevant_times)
    t_i = np.argmin(np.abs(relevant_times - t))
    s_i = np.argmin(np.abs(relevant_times - s))
    return vertical_Laplacian_extended(q, boundary_matrices, s_i, t_i, simplices_at_time, relevant_times, verb=verb)

def plot_Laplacian_new_eigenvalues(f: d.Filtration, weight_fun, max_dim = 1, plot_types = "all", method = "cor10", 
                     plot_args_mesh = {}, 
                     plot_args_diag = {},
                     plot_args_line = {},
                     plot_type_to_fun = {}):
    """
    lapalcian_type: "persistent" for normal laplacian, or "cross" for cross laplacian, "cross_cor10" for cross Laplacian in two directions.
    """
    eigenvalues, relevant_times = cross_Laplacian_new_eigenvalues(f, weight_fun, max_dim=max_dim, method = method)

    fig, ax = plot_eigenvalues(eigenvalues, relevant_times, plot_types=plot_types, filtration=f,
                               plot_args_mesh = plot_args_mesh, plot_args_diag=plot_args_diag,
                               plot_args_line=plot_args_line, plot_type_to_fun=plot_type_to_fun)
    return eigenvalues, relevant_times, fig, ax

