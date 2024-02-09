if __name__ == '__main__':
    pass

import pandas as pd
import numpy as np
import numpy.linalg as LA
import math
import matplotlib.pyplot as plt
import networkx as nx
import sklearn
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import bisect

def decide_case(z,L,d,G,epsilon):
    if G.degree[z] == 0:
        return 3
    else:
        for l in L:
            if d(l,z) < epsilon:
                return 1
                break
        return 2
        
def adjacency_list(G):
    nodes_graph = list(G.nodes())
    edges_graph = list(G.edges)
    adj_list = {x:[x] for x in nodes_graph}
    
    for key in list(adj_list.keys()):
        for node1,node2 in edges_graph:
            if key == node1:
                adj_list[key].append(node2)
            if key == node2:
                adj_list[key].append(node1)
                
    return adj_list

def update_graph_and_adjacency(G,Adj,x,y):#G will be updated
    if x not in list(G.nodes()):
        G.add_node(x)
        
    if y not in list(G.nodes()):
        G.add_node(y)
        
    G.add_edge(x,y)
    
    if x not in list(Adj.keys()) and y not in list(Adj.keys()):
        Adj[x] = [y]
        Adj[y] = [x]
        
    elif x not in list(Adj.keys()) and y in list(Adj.keys()):
        Adj[x] = [y]
        Adj[y].append(x)
        
    elif x in list(Adj.keys()) and y not in list(Adj.keys()):
        Adj[x].append(y)
        Adj[y] = [x]
        
    else:
        Adj[x].append(y)
        Adj[y].append(x)
        
    return Adj

def OnlyBy(Adj,z,L,n):#Adj=adjacency list,z=the landmark in which we compute,L=list of landmarks,n=number of nodes
    if z not in L:
        return("Not a landmark")
    
    if set(Adj[z]).issubset(set(L)):
        return set()
    else:
        union_neighbor = []
        
        for l in L:
            if l!=z:
                union_neighbor = union_neighbor + Adj[l]
            else:
                continue
        
        compare_union = [0]*n
        compare_z = [0]*n
        
        for i in union_neighbor:
            compare_union[i] = 1
            
        for i in Adj[z]:
            compare_z[i] = 1
            
        result = set()
        for i,j in enumerate(compare_z):
            if j == 1:
                if compare_union[i] == 0:
                    result.add(i)
                else:
                    continue
            else:
                continue
        return result

def REMOVE(x,y,Adj,L,n):
    if x in L and y in L:
        if OnlyBy(Adj,x,L,n) == set():
            return {x}
        elif OnlyBy(Adj,y,L,n) == set():
            return {y}
        else:
            return set()
    elif x in L and y not in L:
        if OnlyBy(Adj,x,L,n) == set():
            return {x}
        else:
            return set()
    else:
        if OnlyBy(Adj,y,L,n) == set():
            return {y}
        else:
            return set()

def Merge_sort(A,B):
    A1 = [x[1] for x in A]
    B1 = [x[1] for x in B]
    i = 0
    j = 0
    C = []
    while i < len(A1) and j < len(B1):
        if A1[i] <= B1[j]:
            C.append(A[i])
            i = i + 1
        else:
            C.append(B[j])
            j = j + 1
    if i == len(A1):
        return C + B[j:]
    elif j == len(B1):
        return C + A[i:]
    else:
        print("Error")
        print("Error")

def landmark_replacement(G,Adj,L,x,dist,dist_new,epsilon):#G=graph, L=landmarks,x=new datum,d=distance between data points
    V = list(G.nodes())
    n = len(V)#do not forget to update n

    if G.degree[x] <= 2*np.sqrt(len(G.edges())):
        L.add(x)
    else:
        for i in G.neighbors(x):
            if G.degree[i] <= np.sqrt(len(G.edges())):
                L.add(i)
                break
            else:
                continue

    dist_all = Merge_sort(dist,dist_new)
    pw_dist = [x[1] for x in dist_all]
    pw_node = [x[0] for x in dist_all]
    ell, k = pw_node[0]
    epsilon = pw_dist[0]
    count = 0
    
    while (ell not in L) and (k not in L):
        count = count +1
        ell,k = pw_node[count]
        epsilon = pw_dist[count]
        Adj = update_graph_and_adjacency(G,Adj,ell,k)
        n = len(Adj.keys())
        
    pi = REMOVE(ell,k,Adj,L,n)
    
    while pi == set():
        count = count + 1
        epsilon = pw_dist[count]
        v1,v2 = pw_node[count]#To update the edges
        Adj = update_graph_and_adjacency(G,Adj,v1,v2)
        n = len(Adj.keys())
        pi = REMOVE(ell,k,Adj,L,n)

    L = L-pi
    return L, epsilon, Adj, dist_all
