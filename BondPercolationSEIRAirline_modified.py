# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:02:14 2018

@author: Kossi Edoh
Modified by Azaldin Freidoon on June 23 2020
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas

from ContactNetwork import ContactNetwork

with open('Airline_Jan_2013.csv', newline='') as csvfile:
    airline_data = pandas.read_csv(csvfile)

# Formatting the Data
airline_data = airline_data.iloc[1::2]
airport_nodes = set(airline_data["ORIGIN"].unique()).union(airline_data["DEST"].unique())
num_nodes = len(airport_nodes)

# Creating a matrix with r_ij values
r = np.zeros([num_nodes, num_nodes])

airport_nodes_map = dict([(node,i) for i,node in enumerate(airport_nodes)])
origin = airline_data["ORIGIN"].map(airport_nodes_map).tolist()
destination = airline_data["DEST"].map(airport_nodes_map).tolist()
probabilities = airline_data["PASSENGERS"].to_numpy(dtype=float) / 80000#airline_data["PASSENGERS"].max()

for i in range(len(origin)):
    r[origin[i]][destination[i]] = probabilities[i]
    r[destination[i]][origin[i]] = probabilities[i]

for i in range(num_nodes):
    r[i][i] = 1.0

print("Average value of r_ij is ", r.sum()/len(origin))

# Calculate the degrees
degree = np.count_nonzero(r,axis=1)

# Calculate the degree distribution
dist = np.zeros(max(degree)+1)
for i in degree:
    dist[i] += 1

# convert adjacency matrix into edge weights
nodes_airport_map = dict([(i,node) for i,node in enumerate(airport_nodes)])
edges = []

idx = np.nonzero(r)
for i in range(np.shape(idx)[1]):
    a = idx[0][i]
    b = idx[1][i]
    if (a != b):
        edges += [(nodes_airport_map[a], nodes_airport_map[b], {'weight': r[a][b]})]

# creating a NetworkX graph with the set of edge weights
g = nx.Graph(edges)
CN = ContactNetwork(g)

infected_nodes = ['BKG', 'JAX', 'MLI', 'SWF', 'LCH']
stats = CN.collect_statistics(infected_nodes, 100)

# plotting the number of exposed and infected nodes over time
ax = plt.subplot()
ax.plot(range(101), stats[0],'orange', label='exposed')
ax.plot(range(101), stats[1], 'red', label='infected')
ax.set_xlabel('time')
ax.set_title('Total number of exposed and infected')
ax.legend()
plt.show()