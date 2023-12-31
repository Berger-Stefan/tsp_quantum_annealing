import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
from python_tsp.heuristics import solve_tsp_local_search, simulated_annealing
from python_tsp.exact import solve_tsp_dynamic_programming
import matplotlib.animation as animation 

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

print(f"Shape of the domain {bw_img.shape}")
print(f"Number of pixels {len(black_pixel_coordinates)}")
#plt.imshow(bw_img)
#plt.show()


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


permutation, distance = solve_tsp_local_search(distance_matrix)

# print(permutation)
# print(black_pixel_coordinates)

# for i,(x,y) in enumerate(black_pixel_coordinates):
#     plt.scatter(x,y)

# for i in range(len(black_pixel_coordinates)):
#     plt.annotate(str(i),(black_pixel_coordinates[i][0], black_pixel_coordinates[i][1]))

# plt.show()

# G_plot = nx.DiGraph()
# G_plot.add_nodes_from(np.arange(len(black_pixel_coordinates)))

# for i in range(len(permutation)-1):
#     G_plot.add_edge(permutation[i],permutation[i+1])

# nx.draw(G_plot,black_pixel_coordinates)
# plt.show()

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
    index = index + 1

    return im,

# Set up the animation
num_frames = len(permutation)
ani = FuncAnimation(fig, update_plot, frames=num_frames, interval=10, blit=True)

ani.save('aninmation.mp4', writer="ffmpeg", fps=15)
