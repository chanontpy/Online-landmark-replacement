#------Import files--------------------------------------------------
from Online_LandmarkReplacement_Functions import *
from LandmarkMDS import *

#------Import S-curve from sklearn.-------------------------------------------
X0, y0 = sklearn.datasets.make_s_curve(n_samples=1000,noise=0, random_state=1)
zipped = zip(X0,y0)
zipped = list(zipped)
res = sorted(zipped, key = lambda x: x[1])#Sort the data points by the univariate position.
X = [i for i,j in res]
X = np.array(X)
y = [j for i,j in res]
y = np.array(y)

#-----Construct a graph G------
G = nx.Graph()
m = 100#the permitted number of landmarks
epsilon = 10**-20
G.add_nodes_from([(i,{"pos":(X[i,0],X[i,1],X[i,2])}) for i in range(0,m)])#Assign the coordinate for each node.
pos=nx.get_node_attributes(G,'pos')

#------Euclidean distance between two nodes, where pos[x],pos[y] are the coordinate of x and y, respectively.------------
def d2(x,y):
    return np.linalg.norm(np.array(pos[x])-np.array(pos[y]))

#-----Construct the geometric graph G------
for i in G.nodes:
    for j in G.nodes:
        if i != j and d2(i,j) < epsilon:
            G.add_edge(i,j)

#-----Compute the pairwise distance of any pair of existing nodes in G.------------------
pairwise_dist = []
for i in G.nodes:
    for j in G.nodes:
        if i < j:
            pairwise_dist.append([(i,j),d2(i,j)])
pairwise_dist.sort(key = lambda x:x[1])#Sort the elements in pairwise_dist by the encoded distance values.

L = set(range(0,m))#Initialized landmarks
epsilon_growth = [10**-20]#To keep track the epsilon-value once the new data have been admitted.

#-----Landmark replacement algorithm---------------------------------
adj_list = adjacency_list(G)
index0 = bisect.bisect_left([x[1] for x in pairwise_dist], epsilon)
dist = pairwise_dist[index0:]
for ai in range(100,1000):
    G.add_nodes_from([(i,{"pos":(X[i,0],X[i,1],X[i,2])}) for i in range(ai,ai+1)])
    pos=nx.get_node_attributes(G,'pos')
    adj_list[ai] = []
    dist_new = []

    for i in G.nodes:
        if i != ai and d2(i,ai) < epsilon:
            adj_list = update_graph_and_adjacency(G,adj_list,i,ai)
        elif i != ai and d2(i,ai) >= epsilon:
            dist_new.append([(i,ai),d2(i,ai)])
        else:
            continue
    dist_new.sort(key = lambda x: x[1])
    
    if decide_case(ai,L,d2,G,epsilon) == 1:
        epsilon_growth.append([epsilon,ai])
        continue
    
    elif len(set(L)) < m:
        L.append(ai)
        epsilon_growth.append([epsilon,ai])
        continue
    else:
        Z = landmark_replacement(G,adj_list,L,ai,dist,dist_new,epsilon)
        L = Z[0]
        epsilon1 = Z[1]
        epsilon_growth.append([epsilon1,ai])
        epsilon = epsilon1
        adj_list = Z[2]
        dist_old = Z[3]
        index = bisect.bisect_left([x[1] for x in dist_old], epsilon)
        dist = dist_old[index:]
#-----Landmark Multidimensional scaling begins here.------------------
sq_distance_mat = np.zeros((len(L),len(L)))
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

#-----Plot 2D embedding of S-surve-------------------
plt.figure(figsize=(10, 10), dpi=60)
plt.scatter(X_embed, Y_embed, s=100, ec="w",c=y,cmap=plt.cm.cool)
plt.scatter(L_embed_x, L_embed_y,s=100,color = 'black',zorder = 1, marker = "D")
plt.grid()
plt.show()
