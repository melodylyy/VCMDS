import  numpy as np



def updating_U ( A, U, V, Sm, lam1, lam2):
    m, n = U.shape
    fenzi = (A.dot(V.T)-lam1*Sm.dot(U))

    fenmu = (V).dot(V.T)+lam2*(np.ones([n, n]))+lam1*((U.T).dot(U))

    fenmu1 = np.linalg.inv(fenmu)

    U_new = (fenzi.dot(fenmu1))

    return U_new


def updating_V(A, U, V, Sd, lam2, lam3):
    m, n = V.shape
    fenzi = ((A.T).dot(U) - lam3 * Sd.dot(V.T))
    fenmu = (U.T).dot(U) + lam2 * (np.ones([m, m])) + lam3 * ((V).dot(V.T))
    fenmu1 = np.linalg.inv(fenmu)
    V_new = (fenzi.dot(fenmu1))
    VV = V_new.T
    return VV










def objective_function(W, A, U, V, lam):
    m, n = A.shape
    sum_obj = 0
    for i in range(m):
        for j in range(n):
            sum_obj = sum_obj + W[i,j]*(A[i,j] - U[i,:].dot(V[:,j]))+ lam*(np.linalg.norm(U[i, :], ord=1,keepdims= False) + np.linalg.norm(V[:, j], ord = 1, keepdims = False))
    return  sum_obj



def get_low_feature(k, lam, A, Sm, Sd, lam1, lam2, lam3):#k is the number elements in the features, lam is the parameter for adjusting, th is the threshold for coverage state
    m, n = A.shape
    arr1=np.random.randint(0,100,size=(m,k))
    U = arr1/100#miRNA
    arr2=np.random.randint(0,100,size=(k,n))
    V = arr2/100#disease
    obj_value = objective_function(A, A, U, V, lam)
    obj_value1 = obj_value + 1
    i = 0
    diff = abs(obj_value1 - obj_value)
    while i < 10000:
        i =i + 1
        U = updating_U ( A, U, V, Sm, lam1, lam2)
        V = updating_V (A, U, V, Sd, lam2, lam3)

    return U, V.transpose()

