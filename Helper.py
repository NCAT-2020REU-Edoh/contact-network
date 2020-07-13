#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 07:11:40 2020

@author: azaldinfreidoon
"""

import numpy as np
from itertools import combinations

class Probability:
    """
    This is a class for mathematical operations involving probabilities
    """
    
    @staticmethod
    def jointProbability(e):
        """
        Calculates the joint probability of a set of independent events.
        
        We assume that all the events are independent. The joint probability is calculated
        by taking the product of all the corresponding probabilities of the independent events.

        Parameters
        ----------
        e : array_like
            List of the probabilities of all the independent events.

        Returns
        -------
        product : depends on the types of the elements of e
            Product of all probabilities of all the independent events.

        """
        return np.prod(e)
    
    @staticmethod
    def unionProbability(e):
        """
        Calculates the union probability of a set of independent events.
        
        We assume all the events are independent. The union probability is calculated as follows:
            P(A + B) = P(A) + P(B) - jointProbability(A,B)
            P(A + B + C) = P(A) + P(B) + P(C) - jointProbability(A,B) - jointProbability(A,C)
                - jointProbability(B,C) + jointProbability(A,B,C)
            etc.
        
        However, this is equivalent to
            P(A+B) = 1 - (1-P(A))*(1-P(B))

        Parameters
        ----------
        e : array_like
            List of the probabilities of all the independent events.

        Returns
        -------
        union probability : depends on the types of the elements of e
            The union probability of the independent events.

        """
        return 1 - Probability.jointProbability([1-x for x in e])
    
    @staticmethod
    def millers_algorithm(e):
        """
        Implementation of Miller's algorithm of computing union probabilities of independent events
        
        Reference:
            Miller, G. D. (1968). Programming Techniques: An algorithm for the probability of the union
                of a large number of events. Communications of the ACM, 11(9), 630-631.
                doi:10.1145/364063.364084

        Parameters
        ----------
        e : array_like
            List of the probabilities of all the independent events.

        Returns
        -------
        union probability : depends on the types of the elements of e
            The union probability of the independent events.

        """
        n = len(e)
        prob = np.array([e]).T
        prev = prob
        res = prev.sum()
        for r in range(2, n+1):
            size = n-r+1
            u = np.triu(np.ones([size]*2),0)
            prev = np.matmul(u, prev[1:]) * prob[0:size]
            res += ((-1)**(r-1)) * prev.sum()
        return res


class Distribution:
    """
    This is a class that implements the different normal distributions used in a contact network.
    """
    
    @staticmethod
    def sampleRecoveryDistribution():
        """
        Samples the recovery normal distribution, which has a mean of 14 days and a standard
        deviation of 1 day.

        Returns
        -------
        sample: scalar
            Sample from the normal distribution.

        """
        return np.random.normal(14,1)
    
    @staticmethod
    def sampleExposureDistribution():
        """
        Samples the exposure normal distribution, which has a mean of 0.97 and a standard deviation
        of 0.1. The values are bounded by 0 and 1.

        Returns
        -------
        sample: scalar
            Sample from the normal distribution.

        """
        return max(0, min(1, np.random.normal(0.97, 0.1)))
    
    @staticmethod
    def sampleInfectionDistribution():
        """
        Samples the infectious rate normal distribution, which has a mean of 9 days and a standard deviation
        of 2 days.

        Returns
        -------
        sample: scalar
            Sample from the normal distribution.

        """
        return np.random.normal(9,2)