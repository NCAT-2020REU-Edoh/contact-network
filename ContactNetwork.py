#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:50:05 2020

@author: azaldinfreidoon
"""

from enum import Enum
from Helper import Probability as Prob
from Helper import Distribution as Dist
import numpy as np

class ContactNetwork:
    """
    Class used to represent a contact network of nodes and edges used to model Epidemiological events
    based on the compartmental SEIR model.
    
    Note: Some of the methods are prefixed with one underscore to indicate that it is a private method.
    However, they can still be accessed if need be without errors. There is no use in prefixing them with
    a double under score since name mangling is not for privacy, but to avoid inheritance name collision
    with subclasses.
    
    Attributes
    ----------
    graph : classes.graph.Graph
        NetworkX graph that is used to model the contact network.
    """
    
    def __init__(self, graph):
        """
        Constructor that initializes the contact network with the specified NetworkX graph.

        Parameters
        ----------
        graph : classes.graph.Graph
            NetworkX graph that is used to model the contact network. The graph is deep copied and no changes
            made to the graph within this class implementation will affect the original graph. The graph must
            have the weight attribute defined for all its edges. However, the state of the nodes, the time
            the node changed state, and the time elapsed are all encoded as attributes to the nodes and graph,
            respectively, by the algorithm.
            
            The 'tau' graph attribute is used to encode the current time of the state of the entire graph.
            This allows us to look at every different iteration of the contact network modeling algorithm
            as working with a snapshot of the contact network graph. As such, this attribute is used by the
            _update() function to modify the states of the nodes.

        Returns
        -------
        None.

        """
        self.graph = graph.copy()
        self.graph.graph["tau"] = 0
        for node in self.graph.nodes():
            self._change_node_state(node, self.State.SUSCEPTIBLE)
    
    class State(Enum):
        """
        This is an ENUM nested class that encodes the different states of a node in an SEIR model.
        """
        SUSCEPTIBLE = 0
        INFECTED = 1
        RECOVERED = 2
        EXPOSED = 3
    
    def _change_node_state(self, node, state):
        """
        Changes the state of the node.
        
        The state of the node is encoded as node attributes in the networkx graph. Similarily, the time that
        has elapsed is encoded as an attribute to the graph itself. It is used to determine the time at which
        the node has changed states. When updating the state attribute of the node, we need to encode the time
        the node changed state because the rate at which it infects other nodes increases starting from the
        time the node changed states.

        Parameters
        ----------
        node : Any hashable python object except None.
            This is the object that exists in the NetworkX node dictionary.
        state : Enum
            This is the state of the node as defined by the State inner class.

        Returns
        -------
        None.

        """
        self.graph.nodes[node]["state"] = state
        self.graph.nodes[node]["time_changed_state"] = self.graph.graph["tau"]
    
    def _get_node_state(self, node):
        """
        Returns the current state of the node.

        Parameters
        ----------
        node : Any hashable python object except None.
            This is the object that exists in the NetworkX node dictionary.

        Returns
        -------
        Enum
            This is the state of the node as defined by the State inner class.

        """
        return self.graph.nodes[node]["state"]
    
    def model_contact_network(self, infected_nodes, num_iterations):
        """
        Begins modeling the graph associated with this instance of the class with the specified number of
        iterations and starting set of infected nodes.

        Parameters
        ----------
        infected_nodes : list of any hashable python objects except None.
            List of initially infected nodes.
        num_iterations : int
            The specified number of iterations for which the algorithm will run.

        Returns
        -------
        None.

        """
        for node in infected_nodes:
            self._change_node_state(node, self.State.INFECTED)
        
        print("Press enter to advance at the end of every iteration")
        for tau in range(num_iterations):
            print("Time:", tau)
            modified_node_states = self._update()
            for node, state in modified_node_states.items():
                print("node:", node, "->", state)
            input()
    
    def collect_statistics(self, infected_nodes, num_iterations):
        """
        Tracks the states of the nodes in the graph for every iteration.
        
        The returned matrix contains three rows corresponding to exposed, infected and recovered nodes.
        Each element in the row is the total number of nodes that belong to that state at that iteration.
        The susceptible row was omitted since it can easily be derived from the first three rows.

        Parameters
        ----------
        infected_nodes : ist of any hashable python objects except None.
            List of initially infected nodes.
        num_iterations : int
            The specified number of iterations for which the algorithm will run.

        Returns
        -------
        stats : Array of float64
            numpy ndarray with (3, num_iterations) shape and float64 type.

        """
        for node in infected_nodes:
            self._change_node_state(node, self.State.INFECTED)
        
        stats = np.zeros([3, num_iterations+1])
        state_map = dict({
            self.State.EXPOSED: 0,
            self.State.INFECTED: 1,
            self.State.RECOVERED: 2
        })
        
        stats[state_map[self.State.INFECTED]][0] = len(infected_nodes)
        
        for tau in range(1, num_iterations+1):
            modified_node_states = self._update()
            
            total_nodes_modified = [0,0,0]
            for node in modified_node_states:
                state = modified_node_states[node]
                total_nodes_modified[state_map[state]] += 1
            
            idx_e = state_map[self.State.EXPOSED]
            idx_i = state_map[self.State.INFECTED]
            idx_r = state_map[self.State.RECOVERED]
            stats[idx_e][tau] = stats[idx_e][tau-1] + total_nodes_modified[idx_e] - total_nodes_modified[idx_i]
            stats[idx_i][tau] = stats[idx_i][tau-1] + total_nodes_modified[idx_i] - total_nodes_modified[idx_r]
            stats[idx_r][tau] = stats[idx_r][tau-1] + total_nodes_modified[idx_r]
        
        return stats
    
    def get_animation_func(self, infected_nodes, *artists):
        """
        Returns the func used to animate networkx graphs using FuncAnimation from Matplotlib.animation.
        
        The local data structures and variables defined in this function will be stored in the returned
        function closure.

        Parameters
        ----------
        infected_nodes : list of any hashable python objects except None.
            List of initially infected nodes.
        *artists : collections.PathCollection and collections.LineCollection objects
            The artist objects that represent the nodes and the edges.

        Returns
        -------
        function
            The animation function that is used in Matplotlib.animation.FuncAnimation(...).

        """
        nodes = artists[0]
        text = artists[2]
        
        for node in infected_nodes:
            self._change_node_state(node, self.State.INFECTED)
        
        def _get_node_color(node):
            return {self.State.SUSCEPTIBLE: "#0000ff",
                    self.State.INFECTED: "#ff0000",
                    self.State.RECOVERED: "#00ff00",
                    self.State.EXPOSED: "#ffa500"}[self.graph.nodes[node]["state"]]
        
        idx = dict((node, idx) for (idx, node) in enumerate(self.graph.nodes()))
        colors = [_get_node_color(node) for node in self.graph.nodes()]
        def animate(frame):
            modified_node_states = self._update()
            for node in modified_node_states:
                color = _get_node_color(node)
                colors[idx[node]] = color
            nodes.set_facecolor(colors)
            text.set_text("Day " + str(frame))
            return artists
        return animate
    
    def _update(self):
        """
        Updates the graph based on the states of the nodes, weights of the edges, and the current time
        attribute of the graph.

        Returns
        -------
        modified_node_states : dict
            A dictionary that maps all the nodes that were modified to the their respective modified state.

        """
        
        """
        TODO: Optimize this so that we are not traversing all nodes. Can be
        done by only checking non_susceptible nodes and their neighbors.
        """
        tau = self.graph.graph["tau"]
        # modified node states is used to prevent having changed node states affect the current iteration
        modified_node_states = dict()
        for node in self.graph.nodes():
            state = self._get_node_state(node)
            if (state == self.State.INFECTED):
                if (tau - self.graph.nodes[node]["time_changed_state"] >= Dist.sampleRecoveryDistribution()):
                    modified_node_states[node] = self.State.RECOVERED
            elif (state == self.State.SUSCEPTIBLE):
                # Find all adjacent edges to the susceptible node connected to exposed or infected
                # nodes. Creates a tuple with the weight of the edge and the time the node changed
                # state of the connected infected or exposed node.
                edges = [(self.graph[node][n]['weight'], self.graph.nodes[n]['time_changed_state'])
                           for n in self.graph.neighbors(node) if
                                   ((self._get_node_state(n) == self.State.INFECTED) or
                                   (self._get_node_state(n) == self.State.EXPOSED))
                        ]
                T = lambda r, t: 1 - (1 - r)**(tau - t)
                z = Prob.unionProbability([T(r,t) for (r,t) in edges])
                if (z >= Dist.sampleExposureDistribution()):
                    modified_node_states[node] = self.State.EXPOSED
            elif (state == self.State.EXPOSED):
                if (tau - self.graph.nodes[node]["time_changed_state"] >= Dist.sampleInfectionDistribution()):
                    modified_node_states[node] = self.State.INFECTED
        # apply all the modifications to the networkx graph
        for node in modified_node_states:
            self._change_node_state(node, modified_node_states[node])
        self.graph.graph["tau"] += 1
        return modified_node_states