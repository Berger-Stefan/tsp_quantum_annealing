import numpy as np
import pandas as pd
import cv2
import networkx as nx

import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import dwave.inspector

## Parameters
rescale_factor = 0.1

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
        G.add_weighted_edges_from([(i,j,distance_matrix[i,j])])


# --- Problem formulation ---
Q = dnx.traveling_salesman_qubo(G)

chainstrength = 8
numruns = 10
print("Starting sampler")
sampler = EmbeddingComposite(DWaveSampler(token="kAWe-bd81b032df883596474d0f5d590bd37b3467a591"))
print("Sampler finished, starting copmutation")
response = sampler.sample_qubo(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               label='TSP')

dwave.inspector.show(response)
df = response.to_pandas_dataframe().sort_values('energy').reset_index(drop=True)
df.to_csv('tsp_results.csv', index=False)

