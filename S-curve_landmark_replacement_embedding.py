#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Online_LandmarkReplacement_Functions import *
from LandmarkMDS import *

def d2(x,y):#Euclidean distance between two nodes, where pos[x],pos[y] are the coordinate of x and y, respectively.
    return np.linalg.norm(np.array(pos[x])-np.array(pos[y]))


# In[2]:


X0, y0 = sklearn.datasets.make_s_curve(n_samples=1000,noise=0, random_state=1)#To import S-curve from sklearn.
zipped = zip(X0,y0)
zipped = list(zipped)
res = sorted(zipped, key = lambda x: x[1])#Sort the data points by the univariate position.
X = [i for i,j in res]
X = np.array(X)
y = [j for i,j in res]
y = np.array(y)


# In[3]:


G = nx.Graph()
m = 100#the permitted number of landmarks
epsilon = 10**-20
G.add_nodes_from([(i,{"pos":(X[i,0],X[i,1],X[i,2])}) for i in range(0,m)])#Assign the coordinate for each node.
pos=nx.get_node_attributes(G,'pos')
for i in G.nodes:
    for j in G.nodes:
        if i != j and d2(i,j) < epsilon:
            G.add_edge(i,j)
#adj_list = adjacency_list(G)


# In[4]:


pairwise_dist = []
for i in G.nodes:#Compute the pairwise distance of any pair of existing nodes in G.
    for j in G.nodes:
        if i < j:
            pairwise_dist.append([(i,j),d2(i,j)])
pairwise_dist.sort(key = lambda x:x[1])#Sort the elements in pairwise_dist by the encoded distance values.


# In[5]:


L = set(range(0,m))#Initialized landmarks
epsilon_growth = [10**-20]#To keep track the epsilon-value once the new data have been admitted.


# In[6]:


get_ipython().run_cell_magic('time', '', 'adj_list = adjacency_list(G)\nindex0 = bisect.bisect_left([x[1] for x in pairwise_dist], epsilon)\ndist = pairwise_dist[index0:]\nfor ai in range(m,1000):\n    G.add_nodes_from([(i,{"pos":(X[i,0],X[i,1],X[i,2])}) for i in range(ai,ai+1)])\n    pos=nx.get_node_attributes(G,\'pos\')\n    adj_list[ai] = []\n    dist_new = []\n\n    for i in G.nodes:\n        if i != ai and d2(i,ai) < epsilon:\n            adj_list = update_graph_and_adjacency(G,adj_list,i,ai)\n        elif i != ai and d2(i,ai) >= epsilon:\n            dist_new.append([(i,ai),d2(i,ai)])\n        else:\n            continue\n    dist_new.sort(key = lambda x: x[1])\n    \n    if decide_case(ai,L,d2,G,epsilon) == 1:\n        epsilon_growth.append([epsilon,ai])\n        continue\n    \n    elif len(L) < m:\n        L.add(ai)\n        epsilon_growth.append([epsilon,ai])\n        continue\n    else:\n        Z = landmark_replacement(G,adj_list,L,ai,dist,dist_new,epsilon)\n        L = Z[0]\n        epsilon1 = Z[1]\n        epsilon_growth.append([epsilon1,ai])\n        epsilon = epsilon1\n        adj_list = Z[2]\n        dist_old = Z[3]\n        index = bisect.bisect_left([x[1] for x in dist_old], epsilon)\n        dist = dist_old[index:]')


# In[7]:


sq_distance_mat = np.zeros((len(L),len(L)))#the distance matrix
for i, ti in enumerate(L):
    for j, tj in enumerate(L):
        if i == j:
            sq_distance_mat[i][j] = 0
            sq_distance_mat[j][i] = 0
        elif i < j:
            sq_distance_mat[i][j] = d2(ti,tj)**2
            sq_distance_mat[j][i] = sq_distance_mat[i][j]
        else:
            continue
Landmark_eigenpair = Classical_MDS(sq_distance_mat,2)[1]#Eigenpairs already dependend on the choice of distance.
delta_mu = Col_sum(sq_distance_mat)

N = np.shape(sq_distance_mat)[1]
inv_sqrt_eigenvalues = []
L_prime_before = []
k = 2#Embedding dimension
for i in range(0,k):
    if Landmark_eigenpair[i][0] > 0:
        inv_sqrt_eigenvalues.append(1 / np.sqrt(Landmark_eigenpair[i][0]))
    
for i in range(0,k):
    vi = inv_sqrt_eigenvalues[i] * Landmark_eigenpair[i][1]
    L_prime_before = np.concatenate((L_prime_before,vi.tolist()), axis = 0)
        
L_prime = np.reshape(L_prime_before, (k,N))

embed_coord = []
for i in G.nodes():
    delta = vector_sq_dist(i,L,d2)
    delta = delta.reshape((len(L),1))
    Del = delta - delta_mu
    embed_coord.append(-0.5 * np.matmul(L_prime,Del))

X_embed = [embed_coord[i][0][0].real for i in G.nodes()]#x-axis embedding coordinate
Y_embed = [embed_coord[i][1][0].real for i in G.nodes()]#y-axis embedding coordinate

L_embed_x = [embed_coord[i][0][0].real for i in L]#x-axis landmark embedding coordinate
L_embed_y = [embed_coord[i][1][0].real for i in L]#y-axis landmark embedding coordinate


# In[8]:


w = 10
h = 10
d = 60
plt.figure(figsize=(w, h), dpi=d)
plt.scatter(X_embed, Y_embed, s=100, ec="w",c=y,cmap=plt.cm.cool)
plt.scatter(L_embed_x, L_embed_y,s=100,color = 'black',zorder = 1, marker = "D")
plt.grid()
plt.show()

