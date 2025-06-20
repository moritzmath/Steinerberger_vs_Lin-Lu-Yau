import networkx as nx
import ot
from numpy.linalg import pinv
import numpy as np


# Implementations of the Steinerberger curvature, the alpha-Ollivier-Ricci curvature and the Lin-Lu-Yau curvature for networkx graphs

def steinerbergerCurvature(G):
    """Networkx implementation of Steinerberger Curvature"""
    n = G.number_of_nodes()
    d_dict = dict(nx.all_pairs_shortest_path_length(G))
    d_matrix = [[d_dict[u][v] for v in range(n)] for u in range(n)]
    v = np.ones(n) * n
    d_inverse = pinv(d_matrix)
    curvature = d_inverse@v
    return curvature


def orc_edge(G,alpha,x,y,dist):
    """Compute alpha-Ollivier-Ricci curvature of a given edge"""
    d_x = G.degree[x]
    d_y = G.degree[y]
    B_x = [x] + [l for l in sorted(G[x])]
    B_y = [y] + [l for l in sorted(G[y])]
    M = np.array([[float(dist[l][k]) for l in B_y] for k in B_x])
    p_x = [alpha] + [(1-alpha)/d_x for _ in range(d_x)]
    p_y = [alpha] + [(1-alpha)/d_y for _ in range(d_y)]
    W = ot.emd2(p_x, p_y, M)
    return 1-W

def ollivier_curvature(G, alpha, double_edges=True):
    """Networkx implementation of alpha-Ollivier-Ricci curvature"""
    dist = dict(nx.all_pairs_shortest_path_length(G))
    curvature = {}
    for x,y in G.edges:
        curvature[(x,y)] = round(orc_edge(G,alpha,x,y,dist),3)
        if double_edges:
            curvature[(x,y)] = curvature[(y,x)]
    
    return curvature

def lly_curvature(G, double_edges=True):
    """Networkx implementation of Lin-Lu-Yau curvature"""
    dist = dict(nx.all_pairs_shortest_path_length(G))
    lly_curvature = {}
    for x,y in G.edges:
        lly_curvature[(x,y)] = round(2*orc_edge(G,1/2,x,y,dist),3)
        if double_edges:
            lly_curvature[(x,y)] = lly_curvature[(y,x)]

    return lly_curvature