#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import copy
import networkx as nx
import hypergraphx as hx
from hypergraphx.representations.projections import clique_projection

class Hyperedge_percolation:
    """
    This is a class for detecting community structure in hypergraphs, with hyperedge percolation methods.
    
    ...
    
    Attributes
    ----------
    N_hyperedges: int
        number of hyperedges
        
    hyperedges: list
        a list of the hyperedges stored in sets, sorted in the order of hyperedge sizes
        
    hyperedge_sizes: list
        a list of the sizes of the hyperedges
        
    Methods
    -------
    hyperedges_greater_less_than_k(self, k: int, gle: str) -> list:
        Returns a list of hyperedges (IDs), that are < or <= or > or >= in size than k.

    hyperedge_percolation_largeEdgeFocus(self, k: int, I_abs: int = None) -> list:
        Returns the community structure detected in a hypergraph analogously to the original k-clique method,
        building up the communities from large hyperedges.
		
    hyperedge_percolation_smallEdgeFocus(self, K: int, I_rel: float, round_type: str = "floor") -> list:
        Returns the community structure detected in a hypergraph, building up the communities from small hyperedges instead of large ones.
        
    """
        
    def __init__(self, list_of_tuples: list, random_shuffle: bool = True) -> None:
        """
        Constructs all the necessary attributes for the hyperedge_percolation object.
        The order of the hyperedges with the same size can be randomized.
        
        Parameters
        ----------
        list_of_tuples: list
            a list of the hyperedges, that can have a type of lists, tuples or sets
            
        random_shuffle: bool
            True ensures that same-sized hyperedges are sorted randomly.
            This has no impact on the detected community structure.
            
        """
        
        if type(list_of_tuples) == list:
            try:
                if not type(list_of_tuples[0]) in [list, tuple, set]:
                    raise TypeError("Only list of tuples, sets or lists allowed!")
            except:
                raise IndexError("The list is empty!")
        else:
            raise TypeError("Only list of tuples, sets or lists allowed!")
            
        self.N_hyperedges = len(list_of_tuples)
        if random_shuffle:
            random.shuffle(list_of_tuples)
            
        self.hyperedges = [set(t) for t in sorted(list_of_tuples, key = len)] #hyperedges in an increasing order of their size
        
        self.hyperedge_sizes = list(map(len, self.hyperedges)) #hyperedge sizes in an increasing order

        
    def hyperedges_greater_less_than_k(self, k: int, gle: str) -> list:
        """
        Lists hyperedges that greater or less in size than k.
        
        Parameters
        ----------
        k: int
            The threshold.
        gle: str
            If "g", list hyperedges with size > k.
            If "l", list hyperedges with size < k.
            If "ge", list hyperedges with size >= k.
            If "le", list hyperedges with size <= k.
            
        Returns
        -------
            Returns a list of hyperedge IDs.
        """
        
        if gle == "g":
            return list(np.argwhere(np.array(self.hyperedge_sizes) > k).T[0])
        elif gle == "l":
            return list(np.argwhere(np.array(self.hyperedge_sizes) < k).T[0])
        elif gle == "ge":
            return list(np.argwhere(np.array(self.hyperedge_sizes) >= k).T[0])
        elif gle == "le":
            return list(np.argwhere(np.array(self.hyperedge_sizes) <= k).T[0])
        else:
            raise ValueError("Inappropriate value for argument gle!")


    def hyperedge_percolation_largeEdgeFocus(self, k: int, I_abs: int = None) -> list:
        """Determines community structure in a hypergraph analogously to the original k-clique method, building up the communities from large hyperedges.
        
        Parameters
        ----------
        k: int
            the minimum hyperedge size
            
        I_abs: int
            the minimum absolute intersection; default=k-1
            
        Returns
        -------
        comm_struct: list
            List of sets, where each set corresponds to a community.
            
        """
        
        if I_abs == None: #use the default setting
            I_abs = k-1
        
        comm_struct = []
        L = self.hyperedges_greater_less_than_k(k, gle = "ge")[::-1] #a list of the name (i.e., index) of those hyperedges that contain at least k nodes (these are the edges that form the communities), following a decreasing order of the hyperedge sizes
            #Note that nodes that are not part of any hyperedge of proper size might not join any of the communities

        while len(L) > 0:
            nameOfStartingHyperEdge = L[0] #the name of the largest hyperedge that still needs to be assigned to a community
            startingHyperEdge = copy.deepcopy(self.hyperedges[nameOfStartingHyperEdge])
            growing_comm = copy.deepcopy(startingHyperEdge) #start to build the current community with the first hyperedge
            hyperedges_in_current_comm = [startingHyperEdge] #the new hyperedges will join the community based on their overlap with each of the included hyperedges, calculated one by one
            L.remove(nameOfStartingHyperEdge)

            attempts = dict(zip(L, [0]*len(L))) #the number of attempts to add the given hyperedge to the current community is initially 0 for all the hyperedges named in the list L

            end_while = False
            while True: #build up the current community
                for i in L: #iterate over the names of the hyperedges that are still needed to be examined in the increasing order of the hyperedge sizes
                    if attempts[i] == 1: #the hyperedge named i has already tried to join the current community in the presence of the exactly same community members
                        end_while = True
                        break #do not check the remaining members of the list L
                    else:
                        currentHyperedge = copy.deepcopy(self.hyperedges[i])
                        joinedTheCurrentComm = False
                        #check whether the overlap between the hyperedge named i and any hyperedge of the current community is large enough
                        for com_member in hyperedges_in_current_comm: #going through the hyperedges already belonging to the community
                            if len(com_member & currentHyperedge) >= I_abs:
                                joinedTheCurrentComm = True
                                growing_comm |= currentHyperedge #add the nodes from the hyperedge named i to the current community
                                hyperedges_in_current_comm.append(currentHyperedge)
                                L.remove(i) #the hyperedge named i has joined a community, so it doesn't have to be examined anymore with regard to any community
                                attempts = dict(zip(L, [0]*len(L))) #there was a change in the current community, so all the remaining hyperedges have an other chance to join the community
                                break #there is no need to continue the comparison between the hyperedge named i and the hyperedges of the current community
                        if not joinedTheCurrentComm: #the hyperedge named i cannot be added to the current community now
                            attempts[i] += 1
                if len(L) == 0:
                    end_while = True
                if end_while: #the current community is ready              
                    break #break out from the while loop that built up the current community -> allow to move on to the next community
            comm_struct.append(growing_comm) #add the current community to the list of communities

        #all the communities are ready -> delete those that are just a subset of another community
        numOfFoundComms = len(comm_struct)
        to_delete = [] #list of the names (indexes) of those communities that has to be deleted (are just a subset of another community)
        for i in range(numOfFoundComms-1):
            for j in range(i+1,numOfFoundComms):
                if comm_struct[i].issubset(comm_struct[j]): #the community named i is the same as or is included by the community named j
                    to_delete.append(i)
                    break #the remaining communities don't have to be compared to the community named i: if there are communities that are just a subset of the community i, then these will be a subset of the community j too, so they will be listed in to_delete due to their comparison with community j
                elif comm_struct[j].issubset(comm_struct[i]): #the community named j is included by the community named j but not the same
                    to_delete.append(j)
        to_delete = np.unique(to_delete) #np.unique returns the sorted unique elements of an array
        for i in to_delete[::-1]:
            comm_struct.pop(i) #remove the ith community (i.e., set of nodes) from the list of communities

        return sorted(comm_struct, key = lambda x: (len(x), min(x), max(x), sum(x))) #communities in the increasing order of their sizes; if equal: in the increasing order of the smallest nodeID among their members; if also equal: in the increasing order of the largest nodeID among their members    [this ensures (in most of the cases...) that the order of the communities in the returned list does not change when the community detection is re-run on the same graph]


    def hyperedge_percolation_smallEdgeFocus(self, K: int, I_rel: float, round_type: str = "floor") -> list:
        """Determines community structure in a hypergraph, building up the communities from small hyperedges instead of large ones.
        
        Parameters
        ----------
        K: int
            the maximum hyperedge size
            
        I_rel: float
            the minimum relative intersection
        
        round_type: str
            It can take the values of 'floor', 'ceil' or 'round', and the integer value of the threshold is calculated accordingly.
            
        Returns
        -------
        comm_struct: list
            List of sets, where each set corresponds to a community. The elements are ordered according to increasing community sizes.
            
        """
        
        comm_struct = []
        L = self.hyperedges_greater_less_than_k(K, gle = "le") #a list of the name (i.e., index) of those hyperedges that contain at most K nodes (these are the edges that form the communities), following an increasing order of the hyperedge sizes
            #Note that nodes that are not part of any hyperedge of proper size might not join any of the communities
        L_starter = copy.deepcopy(L) #a list of the name of those hyperedges from which the growth of a new community must be started
        
        while len(L_starter) > 0:
            nameOfStartingHyperEdge = L_starter[0] #the name of the smallest hyperedge that still needs to be used as a community starter
            growing_comm = copy.deepcopy(self.hyperedges[nameOfStartingHyperEdge]) #start to build the current community with the first hyperedge
            L_starter.remove(nameOfStartingHyperEdge)

            L_to_add = copy.deepcopy(L) #a list of the name of those hyperedges that might join the current community, following an increasing order of the hyperedge sizes
            L_to_add.remove(nameOfStartingHyperEdge)

            attempts = dict(zip(L_to_add, [0]*len(L_to_add))) #the number of attempts to add the given hyperedge to the current community is initially 0 for all the hyperedges named in the list L_to_add

            end_while = False
            while True: #build up the current community
                for i in L_to_add: #iterate over the names of the hyperedges that are still needed to be examined in the increasing order of the hyperedge sizes
                    if attempts[i] == 1: #the hyperedge named i has already tried to join the current community in the presence of the exactly same community members
                        end_while = True
                        break #do not check the remaining members of the list L_to_add
                    else:
                        currentHyperedge = copy.deepcopy(self.hyperedges[i])
                        #calculate the required number of nodes being both in the hyperedge named i and the current community
                        if round_type == "floor":
                            m = np.floor(I_rel * self.hyperedge_sizes[i])
                        elif round_type == "ceil":
                            m = np.ceil(I_rel * self.hyperedge_sizes[i])
                        elif round_type == "round":
                            m = round(I_rel * self.hyperedge_sizes[i])
                        else:
                            raise ValueError("The value of round_type can be only 'floor', 'ceil' or 'round'.")
                        #check whether the overlap between the hyperedge named i and the current community is large enough
                        if len(currentHyperedge & growing_comm) >= m:
                            growing_comm |= currentHyperedge #add the nodes from the hyperedge named i to the current community
                            L_to_add.remove(i) #the hyperedge named i doesn't have to be examined anymore with regard to the current community
                            try:
                                L_starter.remove(i) #the hyperedge named i doesn't have to be examined anymore as a starter of any community
                            except ValueError: #the hyperedge name i has already been removed from the list L_starter
                                pass
                            attempts = dict(zip(L_to_add, [0]*len(L_to_add))) #there was a change in the current community, so all the remaining hyperedges have an other chance to join the community
                        else: #the hyperedge named i cannot be added to the current community now
                            attempts[i] += 1
                if len(L_to_add) == 0:
                    end_while = True
                if end_while: #the current community is ready
                    break #break out from the while loop that built up the current community -> allow to move on to the community of the next starting hyperedge

            comm_struct.append(growing_comm) #add the current community to the list of communities

        #all the communities are ready -> delete those that are just a subset of another community
        numOfFoundComms = len(comm_struct)
        to_delete = [] #list of the names (indexes) of those communities that has to be deleted (are just a subset of another community)
        for i in range(numOfFoundComms-1):
            for j in range(i+1,numOfFoundComms):
                if comm_struct[i].issubset(comm_struct[j]): #the community named i is the same as or is included by the community named j
                    to_delete.append(i)
                    break #the remaining communities don't have to be compared to the community named i: if there are communities that are just a subset of the community i, then these will be a subset of the community j too, so they will be listed in to_delete due to their comparison with community j
                elif comm_struct[j].issubset(comm_struct[i]): #the community named j is included by the community named j but not the same
                    to_delete.append(j)
        to_delete = np.unique(to_delete) #np.unique returns the sorted unique elements of an array
        for i in to_delete[::-1]:
            comm_struct.pop(i) #remove the ith community (i.e., set of nodes) from the list of communities
            
        return sorted(comm_struct, key=lambda x:(len(x),min(x),max(x),sum(x))) #communities in the increasing order of their sizes; if equal: in the increasing order of the smallest nodeID among their members; if also equal: in the increasing order of the largest nodeID among their members    [this ensures (in most of the cases...) that the order of the communities in the returned list does not change when the community detection is re-run on the same graph]


class Clique_percolation_projection:
    """This is a class for projecting a hypergraph to a simple graph, with hyperedges becoming fully connected subgraphs (cliques),
    and detecting the community structure of this pairwise projection with the original k-clique percolation method."""
    def __init__(self, hyper_graph: hx.Hypergraph) -> None:

        self.clique_proj = clique_projection(hyper_graph, keep_isolated = True) # nx.Graph object

    def clique_percolation_on_projection(self, k: int) -> list:
        return list(map(set, nx.community.k_clique_communities(self.clique_proj, k)))
