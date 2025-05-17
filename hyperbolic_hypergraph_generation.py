#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import uniform, binom, rv_discrete
import hypergraphx as hx
from hypergraphx.representations.projections import clique_projection
import networkx as nx

class Hypergraph_model:

    @staticmethod
    def inverse_distr(u: float, alpha: float, R: float) -> float:
        return np.arccosh(1 + u * (np.cosh(alpha * R) - 1)) / alpha
    

    @staticmethod
    def get_hyper_distance(theta1: float, theta2: float, r1: float, r2: float) -> float:
        dTheta_ij = np.pi - np.abs(np.pi - np.abs(theta1 - theta2))
        return np.arccosh(np.cosh(r1) * np.cosh(r2) - np.sinh(r1) * np.sinh(r2) * np.cos(dTheta_ij))
    

    def __init__(self, N_nodes: int, avg_hyp_deg: float,
                 alpha: float = 1.0, s_min: int = 10, s_max: int = 20, s_avg: float = 15., 
                 gamma_s: float = 3.0, edge_size_distr: str = "UNI") -> None:

        self.N_nodes = N_nodes
        self.avg_hyp_deg = avg_hyp_deg
        self.s_min = s_min #the allowed smallest hyperedge size
        self.s_max = s_max #the allowed largest hyperedge size
        self.s_avg = s_avg #the expected value of the hyperedge size distribution if it is binomial
        self.gamma_s = gamma_s #decay exponent of the hyperedge size distribution if it is a power law P(s)~s^(-gamma_s)
        self.alpha = alpha #the parameter of the nodes' radial distribution, adjusting the distribution of the hyperdegrees (larger values yield faster decays)
        self.edge_size_distr = edge_size_distr #"DD", "UNI", "BIN", or "POW"

        #arrange the nodes on the hyperbolic plane:
        self.R = 2 * np.log(8*self.N_nodes*(self.alpha**2)/(self.avg_hyp_deg*np.pi*((2*self.alpha-1)**2)))
        self.nodes_r = self.inverse_distr(uniform.rvs(loc = 0, scale = 1, size = self.N_nodes),
                                          self.alpha, self.R)
        self.nodes_theta = uniform.rvs(loc = 0, scale = 2 * np.pi, size = self.N_nodes)
        
        #calculate the hyperbolic distance between all node pairs:
        hyp_dists = np.diag(np.ones(self.N_nodes)*np.inf) #no node connects to itself -> set the corresponding distances to a large value
        for node1 in range(1,self.N_nodes):
            for node2 in range(node1):
                hyp_dists[node1,node2] = self.get_hyper_distance(self.nodes_theta[node1], self.nodes_theta[node2],
                                                                 self.nodes_r[node1], self.nodes_r[node2])
                hyp_dists[node2,node1] = hyp_dists[node1,node2]
        #print(" | DONE")

        #set the target number of hyperedges and the hyperedge sizes:
        if self.edge_size_distr == "DD": #Dirac delta edge size distribution -> the size is self.s_min for every hyperedge
            target_hyp_edge_num = np.ceil(self.N_nodes * self.avg_hyp_deg / self.s_min).astype(int)
            target_edge_sizes = [self.s_min for e in range(target_hyp_edge_num)] #the possible largest number of different hyperedges might be smaller than the target!
        elif self.edge_size_distr == "UNI": #uniform edge size distribution
            #first, create the corresponding discrete random variable class
            allowedSizes = list(range(self.s_min,self.s_max+1)) #list of the allowed hyperedge size values (integers)
            pmf = np.ones(len(allowedSizes))/len(allowedSizes) #the probability mass function
            uniformDistr = rv_discrete(values=(allowedSizes,pmf))
            target_hyp_edge_num = np.ceil(self.N_nodes * self.avg_hyp_deg / uniformDistr.stats(moments = 'm')).astype(int)
            target_edge_sizes = list(uniformDistr.rvs(size = target_hyp_edge_num)) #the possible largest number of different hyperedges might be smaller than the target!
        elif self.edge_size_distr == "BIN": #binomial edge size distribution in [self.s_min,self.s_max], obtained by shifting a binomial distribution in [0,self.s_max-self.s_min] with +self.s_min
            target_hyp_edge_num = np.ceil(self.N_nodes * self.avg_hyp_deg / self.s_avg).astype(int)
            nOfBinomDistr = self.s_max-self.s_min
            pOfBinomDistr = (self.s_avg-self.s_min)/(self.s_max-self.s_min)
                            #the p parameter of the binomial distribution is defined based on the following equation:
                            #self.s_avg := nOfBinomDistr*pOfBinomDistr + self.s_min
            target_edge_sizes = list(binom.rvs(n = nOfBinomDistr, p = pOfBinomDistr, size = target_hyp_edge_num) + self.s_min) #the possible largest number of different hyperedges might be smaller than the target!
                            #sample random integer from [0,self.s_max-self.s_min] according to a binomial distribution and shift them to the required range of [self.s_min,self.s_max] by increasing them with self.s_min
        elif self.edge_size_distr == "POW": #power-law decaying edge size distribution
            #first, create the corresponding discrete random variable class
            allowedSizes = list(range(self.s_min,self.s_max+1)) #list of the allowed hyperedge size values (integers)
            pmf = np.power(np.array(allowedSizes,dtype='float'),-self.gamma_s) #the probability mass function without normalization
            pmf /= pmf.sum() #normalization
            truncatedPowerLawDistr = rv_discrete(values=(allowedSizes,pmf))
            target_hyp_edge_num = np.ceil(self.N_nodes * self.avg_hyp_deg / truncatedPowerLawDistr.stats(moments = 'm')).astype(int)
            target_edge_sizes = list(truncatedPowerLawDistr.rvs(size = target_hyp_edge_num)) #the possible largest number of different hyperedges might be smaller than the target!

        #initialize the hypergraph:
        self.hyp_graph = hx.Hypergraph()
        self.hyp_graph.add_nodes(range(self.N_nodes))
        
        #generate the hyperedges:
        self.hyp_edge_sizes = []
        self.hyp_circ_centers_r = []
        self.hyp_circ_centers_theta = []
        size_num_dict = Counter(target_edge_sizes) #key=hyperedge size, value=required number of hyperedges of this size
        for hyp_edge_size in size_num_dict.keys():
            edge_list_perSize = []
            if self.N_nodes < size_num_dict[hyp_edge_size]: #the required number of hyperedges of the given size is larger than the number of network nodes
                print("WARNING: The target number of hyperedges of size " + str(hyp_edge_size) + " is reduced from " + str(size_num_dict[hyp_edge_size]) + " to the total number of possible hyperedge centers, i.e. N=" + str(self.N_nodes) + ".")
                size_num_dict[hyp_edge_size] = self.N_nodes
            edge_num_to_generate = size_num_dict[hyp_edge_size]
            possible_centers = list(range(self.N_nodes)) #the nodes that has not been tested yet as a center for the given hyperedge size
            while 0 < edge_num_to_generate:
                centers_to_test = np.random.permutation(possible_centers)[:edge_num_to_generate] #sample the required number of hyperedge center nodes randomly, without replacement (hyperedges having the same size AND center cannot be different!)
                for c in centers_to_test:
                    possible_centers.remove(c)
                    distance_order = np.argsort(hyp_dists[c,:])
                    new_edge = tuple(distance_order[:(hyp_edge_size - 1)]) + (c,)
                    new_edge_set = set(new_edge)
                    if new_edge_set not in edge_list_perSize: #the generated hyperedge has not been added to the edge list previously
                        edge_list_perSize.append(new_edge_set)
                        self.hyp_graph.add_edge(new_edge)
                        edge_num_to_generate = edge_num_to_generate-1
                        self.hyp_edge_sizes.append(hyp_edge_size)
                        self.hyp_circ_centers_r.append(self.nodes_r[c])
                        self.hyp_circ_centers_theta.append(self.nodes_theta[c])
                if len(possible_centers)==0:
                    if 0 < edge_num_to_generate:
                        print("WARNING: The number of hyperedges of size "+str(hyp_edge_size)+" is reduced from the target value "+str(size_num_dict[hyp_edge_size])+" to the possible largest number of different hyperedges of this size in the given hypergraph, namely to "+str(size_num_dict[hyp_edge_size]-edge_num_to_generate)+".")
                    break
        self.N_hyp_edges = len(self.hyp_edge_sizes)


    def get_hyperdegree(self):
        """returns a list of the hyperdegrees of each node"""
        return list(self.hyp_graph.degree_sequence().values())

    
    def get_degree(self):
        """returns a list of the degrees of each node"""
        degree = []
        for n in self.hyp_graph.get_nodes():
            neighs = set()
            for e in self.hyp_graph.get_edges():
                if n in e:
                    for i in e:
                        if i != n:
                            neighs.add(i)
            degree.append(len(neighs))
        return degree
    
    ##### PLOT COMMUNITIES IN THE HYPERGRAPH #####
    
    def __plotter(self, projection_edges: list = [], fig_size: float = 7.5, path_filename: str = "",
                             kwargs0: dict = {"node_color": "crimson", "edgecolors": "crimson"}) -> None:

        plt.figure(figsize = (fig_size, fig_size))
        ax1 = plt.subplot(111, polar = True, frame_on = False)
        ax1.scatter(self.nodes_theta, self.nodes_r, s = kwargs0["node_size"],
                    c = kwargs0["node_color"], edgecolors = kwargs0["edgecolors"], zorder=2)


        if projection_edges:
            self.__draw_edges(ax1, projection_edges)

        ax1.grid(False)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        if path_filename:
            plt.savefig(path_filename, dpi = 300, bbox_inches = 'tight')
        #plt.show()

    def __draw_edges(self, ax, edges):
        polars = np.vstack([self.nodes_r, self.nodes_theta]).T
        pos = dict(zip(range(self.hyp_graph.num_nodes()), polars))
        edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edges])
        for edge in edge_pos: 
            (r1, theta1), (r2, theta2) = edge

            ax.plot([theta1, theta2], [r1, r2], color='gray', alpha=0.3, linewidth=0.5, zorder=1)

    def __calculate_node_colors(self, comm_struct: list, col_palette: str) -> dict:

        clrs0 = sns.color_palette(col_palette, n_colors = len(comm_struct))  # a list of RGB tuples
        clrs0.append((0, 0, 0))          #black
        clrs0.append((0.95, 0.95, 0.95)) #white
        clrs0.append((0.6, 0.6, 0.6))    #grey
        colormap = dict()
        border_colormap = dict()
        for n in range(self.N_nodes):
            col = []
            bord_col = []
            for i in range(len(comm_struct)):
                if n in comm_struct[i]:
                    col.append(clrs0[i])

            # unclustered node: white with grey border
            if len(col) == 0:
                colormap.update({n: clrs0[-2]})
                border_colormap.update({n: clrs0[-1]})

            # node clustered in a single community: colored according to the given community
            elif len(col) == 1:
                colormap.update({n: col[0]})
                border_colormap.update({n: col[0]})

            # node clustered in multiple communities: grey with black border
            else:
                colormap.update({n: clrs0[-1]})
                border_colormap.update({n: clrs0[-3]})

        node_edge_color_dict = {"node_color": [v for k, v in sorted(colormap.items())],
                                "edgecolors": [v for k, v in sorted(border_colormap.items())]}

        return node_edge_color_dict

        
    def plot_hyperedge_communities(self, comm_struct: list,
                                   fig_size: float = 6., node_size: float = 10.,
                                   sns_color_palette: str = "Paired",
                                   path_filename: str = "") -> None:
        """
        This method plots the nodes of a hyperbolic hypergraph, visualizing the community structure by colors.
        It takes the community structure as the first argument.

        Note that if more than szám (or the number of colors in the given color palette) communities are given,
        some colors are re-used for multiple communities due to the limited number of colors in the palette.
        """

        keyword_args = self.__calculate_node_colors(comm_struct, sns_color_palette)
        keyword_args.update({"node_size": node_size})
            
        self.__plotter([], fig_size, path_filename, kwargs0 = keyword_args)
            

    def plot_projection_clique_communities(self, comm_struct: list, clique_proj: nx.Graph, 
                                             fig_size: float = 6., node_size: float = 10., sns_color_palette: str = "Paired",
                                             path_filename: str = "") -> None:
        """
        This method plots the clique projection of a hyperbolic hypergraph, visualizing the community structure by colors.
        It takes the community structure as the first and the projection (networkx.Graph object) as the second argument.

        Note that if more than 12 (or the number of colors in the given color palette) communities are given,
        some colors are re-used for multiple communities due to the limited number of colors in the palette.
        """

        keyword_args = self.__calculate_node_colors(comm_struct, sns_color_palette)
        keyword_args.update({"node_size": node_size})

        self.__plotter(list(clique_proj.edges()), fig_size, path_filename, kwargs0 = keyword_args)
