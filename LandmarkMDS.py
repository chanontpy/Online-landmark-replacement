if __name__ == '__main__':

    def Adjacency(C):
        Net =[[ [], -1.0 ]]
        Net2 =[[ [], -1.0 ]]
        time_points = C[:,2]
        nodes0 = C[:,0]
        nodes1 = C[:,1]
        num1 = int(max(np.unique(nodes0)))
        num2 = int(max(np.unique(nodes1)))
        num = max([num1,num2])
        N = np.shape(C[:,0])[0]
        time_points = np.unique(C[:,2])
    
        for t in time_points:
            A = np.zeros((num+1,num+1))
            for k in range(N):
                if C[k][2] == t:
                    n = int(C[k][0])
                    m = int(C[k][1])
                    A[n][m] = 1
                    A[m][n] = 1
            A2 = A.tolist()
            Net.append([A2,t])
        return Net

    def ComputeBk(net_k, t_k, alpha, k ,B_k_1,t_k_1):
        if k == 1 :
            net1  = net_k
            t1 = t_k
            B1 = np.copy(net1)
            return B1, t1

        numrow = len(net_k)
        numcol = len(net_k[0])
        Bk = np.zeros((numrow, numcol))
        for i in range(0, numrow):
            for j in range(0,numcol):
                node_k   = net_k[i][j]
                if node_k != 0:
                    update = node_k + ( B_k_1[i][j]/math.exp( alpha*(t_k-t_k_1) ) )
                else:
                    update = 0 + ( B_k_1[i][j]/math.exp( alpha*(t_k-t_k_1) ) )

                Bk[i][j] = update
        return Bk,t_k

    def Tie_decay_matrices(alpha, C):
        Net = Adjacency(C)
        T = []
        B = [ [ [], -1.0 ] ]
        for i, n_i in enumerate(Net):
            if i==0 :
                continue
            Bi_1   = B[i-1]
            Bi,Ti  = ComputeBk(n_i[0],n_i[1] ,alpha, i, Bi_1[0],Bi_1[1])
            B.append([Bi,Ti])
        for j in range(1,len(B)):
            T.append(B[j][0])
        return T,B
    
    def Squared_Frobenius_Distance_matrix(T): #T is from tie_decay_matrices(alpha,C)
        D3 = np.zeros((len(T),len(T)))
        for i, ti in enumerate(T):
            for j, tj in enumerate(T):
                if i == j:
                    D3[i][j] = 0
                    D3[j][i] = 0
                elif i < j:
                    D3[i][j] = LA.norm(ti-tj)**2
                    D3[j][i] = D3[i][j]
                else:
                    continue
        return D3
    
    def Degree_matrix(A): #A is a tie-decay adjacency matrix at a time t.
        m = np.shape(A)[0]
        n = np.shape(A)[1]
        degree_matrix = np.zeros((m,n))
    
        for i in range(0,m):
            r = sum(A[i])
            degree_matrix[i][i] = degree_matrix[i][i] + r

        return degree_matrix
    
    def Laplacian_matrix(A): #A is also a tie-decay adjacency matrix at a time t.
        L = Degree_matrix(A) - A
        return L

import numpy as np
import numpy.linalg as LA

def Find_eigenpair(D):
    A = []
    eigenvalues = LA.eig(D)[0]
    eigenvectors_col = LA.eig(D)[1]
    
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors_col = eigenvectors_col[:,idx]
    
    eigenvectors_row = np.transpose(eigenvectors_col)
    for i in range(len(eigenvalues)):
        A.append([eigenvalues[i],eigenvectors_row[i]])
    return A

def Centered_Distance_matrix(D):
    n = np.shape(D)[0]
    m = np.shape(D)[1]
    I = np.identity(n)
    One_mat = np.ones((n,n))
    H = I - (1/n * One_mat)
    D_scaled = -0.5 * np.matmul(H,np.matmul(D,H))
    return D_scaled  

def Col_sum(D):#Averaged columns of D
    N = np.shape(D)[1]
    Col_D_sum = np.reshape(np.sum(D, axis = 1), (N,1))
    Delta_mu = 1/N * Col_D_sum
    
    return Delta_mu

def Classical_MDS(D,k): # D is the squared distance
    L = []
    B = Centered_Distance_matrix(D)
    Lambda = Find_eigenpair(B)
    sqrt_eigenvalue = []
    for i in range(0,len(Lambda)):
        if Lambda[i][0] > 0:
            sqrt_eigenvalue.append(np.sqrt(Lambda[i][0]))

    for i in range(0,k):
        L.append(sqrt_eigenvalue[i] * Lambda[i][1])

    return L, Lambda

def vector_sq_dist(a,L,d):#A vector of squared distances between a point and landmarks, d is a distance measure.
    vec = []
    for i in L:
        vec.append(d(a,i)**2)
    return np.array(vec)