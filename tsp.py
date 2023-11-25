import numpy as np
import pandas as pd
import cv2
import networkx as nx

import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import dwave.inspector

## Parameters
rescale_factor = 0.3

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

print(bw_img.shape)
print(len(black_pixel_coordinates))

G = nx.Graph()
G.add_nodes_from(black_pixel_coordinates)

# --- Problem formulation ---
Q = dnx.traveling_salesman_qubo(G)

chainstrength = 8
numruns = 10
sampler = EmbeddingComposite(DWaveSampler(token="kAWe-bd81b032df883596474d0f5d590bd37b3467a591"))
response = sampler.sample_qubo(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               label='TSP')

dwave.inspector.show(response)
df = response.to_pandas_dataframe().sort_values('energy').reset_index(drop=True)
df.to_csv('tsp_results.csv', index=False)

