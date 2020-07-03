#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:45:36 2020

@author: azaldinfreidoon
"""

import networkx as nx
from ContactNetwork import ContactNetwork

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

fig, ax = plt.subplots()
g = nx.random_lobster(10,.9,.9)

weights = dict([(e, np.random.sample(1)[0]) for e in g.edges()])
nx.set_edge_attributes(g, weights, 'weight')

legend_elements = [Line2D([0], [0], marker='o', color='w', lw=4, label='Susceptible', markersize=14, markerfacecolor='b'),
                   Line2D([0], [0], marker='o', color='w', label='Recovered', markersize=14, markerfacecolor='g'),
                   Line2D([0], [0], marker='o', color='w', label='Infected', markersize=14, markerfacecolor='r'),
                   Line2D([0], [0], marker='o', color='w', label='Exposed', markersize=14, markerfacecolor='orange')]

ax.set_title("Contact Network SEIR Model")
ax.legend(handles=legend_elements, loc='upper left')

pos = nx.spring_layout(g)
nodes = nx.draw_networkx_nodes(g,pos, ax=ax, edgecolors="#000000")
edges = nx.draw_networkx_edges(g,pos, ax=ax)
text = ax.text(1,1,"Day 0")
nx.draw_networkx_labels(g, pos, font_color="#ffffff")

cn = ContactNetwork(g)
func = cn.get_animation_func([0,1], nodes, edges, text)

"""
TODO: Setting blit=True for animation.FuncAnimation draws over labels after the first iteration.
"""
ani = animation.FuncAnimation(fig, func, np.arange(1, 200), interval=500)