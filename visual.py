import numpy as np
import pandas as pd
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
from python_tsp.heuristics import solve_tsp_local_search, simulated_annealing
from python_tsp.exact import solve_tsp_dynamic_programming

## Parameters
rescale_factor = 0.3

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

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Create a distance matrix
num_nodes = len(G.nodes)
distance_matrix = np.zeros((num_nodes, num_nodes))

# Populate the distance matrix
for i in range(num_nodes):
    for j in range(num_nodes):
        distance_matrix[i, j] = euclidean_distance(black_pixel_coordinates[i], black_pixel_coordinates[j])

# plt.imshow(bw_img, cmap='hot')
# plt.show()

permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

global index, data
data  = np.zeros((X_range, Y_range))
index = 0
# Create a figure and axis for plotting
fig, ax = plt.subplots()
im = ax.imshow(data, cmap='hot')

# Function to update the plot
def update_plot(frame):
    global index, data
    x,y = black_pixel_coordinates[permutation[index]]
    data[x,y] = -1
    im = ax.imshow(data, cmap='hot')
    #im.set_array(data)
    index = index + 1
    return im,

# Set up the animation
num_frames = np.linspace(0,len(black_pixel_coordinates))
ani = FuncAnimation(fig, update_plot, frames=num_frames, interval=10, blit=True)

# Display the animation
plt.show()