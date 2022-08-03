import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def edge2mat_b(link, num_bone):
    A = np.zeros((num_bone, num_bone+1))
    for i, j in link:
        A[i, j] = 1
    return A

def edge2mat_d(link, num_bone):
    A = np.zeros((num_bone+1, num_bone))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    # In = normalize_digraph(edge2mat(inward, num_node))
    # Out = normalize_digraph(edge2mat(outward, num_node))
    In = edge2mat(inward, num_node)
    Out = edge2mat(outward, num_node)
    A = np.stack((I, In, Out))
    return A

def get_spatial_graph_b(num_bone, inward, outward):
    I = np.zeros((num_bone, num_bone+1))
    In = normalize_digraph(edge2mat_b(inward, num_bone))
    Out = normalize_digraph(edge2mat_b(outward, num_bone))
    B = np.stack((I, In, Out))
    return B

def get_spatial_graph_c(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    C = np.stack((I, In, Out))
    return C


def get_spatial_graph_d(num_bone, inward, outward):
    I = np.zeros((num_bone+1, num_bone))
    In = normalize_digraph(edge2mat_d(inward, num_bone))
    Out = normalize_digraph(edge2mat_d(outward, num_bone))
    D = np.stack((I, In, Out))
    return D

def get_spatial_graph_pa(num_node, link):
    # A = normalize_digraph(edge2mat(link, num_node))
    A = edge2mat(link, num_node)
    return A