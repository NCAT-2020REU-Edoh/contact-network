#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:50:05 2020

@author: azaldinfreidoon
"""

from enum import Enum
from Helper import Probability as Prob
from Helper import Distribution as Dist

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

        Returns
        -------
        None.

        """
        self.graph = graph.copy()
    
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
    
    def model_contact_network(self, num_iterations, infected_nodes):
        """
        Begins modeling the graph associated with this instance of the class with the specified number of
        iterations and starting set of infected nodes.

        Parameters
        ----------
        num_iterations : int
            The specified number of iterations for which the algorithm will run.
        infected_nodes : list of str
            List of initially infected nodes.

        Returns
        -------
        None.

        """
        # This graph attribute is used to encode the current time of the state of the entire graph.
        # This allows us to look at every different iteration of the contact network modeling algorithm
        # as working with a snapshot of the updating graph.
        self.graph.graph["tau"] = 0
        
        for node in self.graph.nodes():
            self._change_node_state(node, self.State.SUSCEPTIBLE)
        
        for node in infected_nodes:
            self._change_node_state(node, self.State.INFECTED)
        
        for tau in range(1, num_iterations):
            self._update()
            input("Continue?")
    
    def _update(self):
        """
        Updates the graph based on the states of the nodes, weights of the edges, and the current time
        attribute of the graph.

        Returns
        -------
        None.

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
        # apply all the modifications to the networkx graph
        for node in modified_node_states:
            self._change_node_state(node, modified_node_states[node])
        self.graph.graph["tau"] += 1