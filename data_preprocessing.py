
import copy
import numpy as np
import math
import random

def InSm(sm1, sm2, w):
    simm = w * sm1 + (1-w) * sm2
    return simm
def GIP_sm(A):
    # find the weights Y_d - rows
    w1 = np.linalg.norm(A,axis=1)
    widthSum = np.sum(np.square(w1))
    Y_sm = A.shape[0]/widthSum
    #constructing the GIP kernel similarity matrix
    G = np.zeros((A.shape[0],A.shape[0]))
    for i in range(G.shape[0]):
        for j in range(i, G.shape[1]):
            G[i, j] = math.exp((-Y_sm)*np.square(np.linalg.norm(A[i] - A[j])))
            G[j, i] = G[i, j]
    return G
def GIP_m(A):
    # find the weights Y_m - rows
    w1 = np.linalg.norm(A,axis=0)
    widthSum = np.sum(np.square(w1))
    Y_m = A.shape[1]/widthSum
    #onstructing the GIP kernel similarity matrix
    G = np.zeros((A.shape[1],A.shape[1]))
    for i in range(G.shape[0]):
        for j in range(i, G.shape[1]):
            G[i, j] = math.exp((-Y_m)*np.square(np.linalg.norm(A[:,i] - A[:,j])))
            G[j, i] = G[i, j]
    return G
def  get_syn_sim1 (A,w):
    disease_sim1 = np.loadtxt("dataset/d-d2matrix.txt")
    miRNA_sim1 = np.loadtxt("dataset/m-mmatrix.txt")
    GIP_m_sim = GIP_sm(A)
    GIP_d_sim = GIP_m(A)
    Pm_final = InSm(copy.deepcopy(miRNA_sim1),copy.deepcopy(GIP_m_sim),w)
    Pd_final = InSm(copy.deepcopy(disease_sim1),copy.deepcopy(GIP_d_sim),w)

    return Pm_final, Pd_final


def get_all_the_samples(A):
    m, n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i, j] != 0:  # If the element is not 0, it is treated as a positive case
                pos.append([i, j, 1])
            else:
                neg.append([i, j, 0])
    samplesz = np.array(pos)
    samplesf = np.array(neg)
    selected_indices = np.random.choice(len(samplesf), len(samplesz), replace=False)
    selected_samplesf = samplesf[selected_indices]

    # Merge the selected samples into an array with samplesz
    all_samples = np.concatenate([samplesz, selected_samplesf], axis=0)

    return all_samples

def update_Adjacency_matrix (A, test_samples):
    print(type(A))
    A = np.array(A)
    print(A.shape)
    m = test_samples.shape[0]
    A_tep = A.copy()
    for i in range(m):
        if test_samples[i,2] ==1:
            A_tep [test_samples[i,0], test_samples[i,1]] = 0
    return A_tep
def set_digo_zero(sim, z):
    sim_new = sim.copy()
    n = sim.shape[0]
    for i in range(n):
        sim_new[i][i] = z
    return sim_new



















