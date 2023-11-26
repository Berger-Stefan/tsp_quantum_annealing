import numpy as np
import pandas as pd
import cv2
import networkx as nx

import itertools
import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import dwave.inspector
import dimod
import matplotlib.pyplot as plt

## Parameters
rescale_factor = 0.05

# --- Graph formulation ---
img         = cv2.imread("cats.png") 
X_range, Y_range, _ = img.shape
new_size    = ((int)(Y_range*rescale_factor), (int)(X_range*rescale_factor))
img         = cv2.resize(img, new_size)
gray_img    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, bw_img = cv2.threshold(gray_img, 200, 255,0)
X_range, Y_range = bw_img.shape
bw_img      = cv2.resize(bw_img, new_size)

black_pixel_coordinates = [(x, y) for x in range(bw_img.shape[0]) for y in range(bw_img.shape[1]) if bw_img[x, y] == 0]


print(f"Shape of the domain {bw_img.shape}")
print(f"Number of pixels {len(black_pixel_coordinates)}")

G = nx.Graph()
G.add_nodes_from(np.arange(len(black_pixel_coordinates)))

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Create a distance matrix
num_nodes = len(G.nodes)
distance_matrix = np.zeros((num_nodes, num_nodes))

# Populate the distance matrix
for i in range(num_nodes):
    for j in range(num_nodes):
        distance_matrix[i, j] = euclidean_distance(black_pixel_coordinates[i], black_pixel_coordinates[j])

for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            G.add_weighted_edges_from([(i,j,distance_matrix[i,j])])

# --- Problem formulation ---
# Q = dnx.traveling_salesman_qubo(G)
N = num_nodes
distance = distance_matrix

max_dist = max(np.max(distance, axis=0))
alpha = ((N**3 - 2*N**2 + N)*max_dist + 1e-3)
beta = ((N**2 + 1)*alpha) + 1e-3
#Q_mat = np.zeros((N,N))
A_mat = np.zeros((N,N))

def concator(A,B,N,row):
    order_list = []
    for i in range(N):
        if i == row:
            order_list.append(A)
        else:
            mat = np.zeros((N,N))
            for i in range(N):
                for j in range(N):
                    mat[i,j] = i * B[i,j]
            order_list.append(mat)
    return order_list

for i in range(N):
    for j in range(N):
        if i == j:
            A_mat[i,j] = -alpha
        else:
            A_mat[i,j] = beta

block = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i == j:
            block[i,j] = 2*alpha
        else:
            block[i,j] = -1 / (distance[i,j]**1.7)

full_order_list = []

for j in range(N):
    row_array = np.concatenate(concator(A_mat, block, N, j), axis=0)
    full_order_list.append(row_array)

Q = np.concatenate(full_order_list, axis=1)
Q -= Q.mean(axis=0)
qmax = max(np.max(Q, axis=0))
for i in range(N**2):
    for j in range(N**2):
        Q[i,j] *= (1/qmax)
        
Q = dnx.traveling_salesperson(G, dimod.ExactSolver(), start=0) 

plt.imshow(Q)
plt.show()

chainstrength = 100
numruns = 1000
print("Starting sampler")
sampler = EmbeddingComposite(DWaveSampler(token="kAWe-bd81b032df883596474d0f5d590bd37b3467a591"))
print("Sampler finished, starting computation")
response = sampler.sample_qubo(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               label='TSP')

dwave.inspector.show(response)
df = response.to_pandas_dataframe().sort_values('energy').reset_index(drop=True)
df.to_csv('tsp_results.csv', index=False)


permutation = []
for i in range(N):
    for j in range(N):
        if df[0][j+i*N] == 1:
            permutation.append(j)

print(f"Permuation: {permutation}")

G_plot = nx.DiGraph()
G_plot.add_nodes_from(np.arange(len(black_pixel_coordinates)))

for i in range(len(permutation)-1):
    G_plot.add_edge(permutation[i],permutation[i+1])

nx.draw(G_plot,black_pixel_coordinates)
plt.show()
