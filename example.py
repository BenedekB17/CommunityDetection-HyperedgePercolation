#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from clique_percolation_hypergraph import Hyperedge_percolation, Clique_percolation_projection
from hyperbolic_hypergraph_generation import Hypergraph_model
import numpy as np
import os
import platform

def clear_console():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

clear_console()  # Comment this line, if you do not want to clear the  terminal

#generate a hyperbolic hypergraph
print("Generating hyperbolic hypergraph...")
hypHypGraph = Hypergraph_model(N_nodes = 500, avg_hyp_deg = 10., alpha = 1.0, s_min = 3, s_max = 9, s_avg = 6., edge_size_distr = "BIN")
print(f"{'Number of hyperedges:':<31}{hypHypGraph.N_hyp_edges:>7}")
print(f"{'Average hyperdegree:':<31}{round(np.mean(hypHypGraph.get_hyperdegree()), 3):>7}")
print(f"{'Smallest hyperedge size:':<31}{min(hypHypGraph.hyp_edge_sizes):>7}")
print(f"{'Largest hyperedge size:':<31}{max(hypHypGraph.hyp_edge_sizes):>7}")
print(f"{'Average hyperedge size:':<31}{round(np.mean(hypHypGraph.hyp_edge_sizes), 3):>7}")


#search for communities in the given hypergraph
HP = Hyperedge_percolation(hypHypGraph.hyp_graph.get_edges(), random_shuffle=True)
CPP = Clique_percolation_projection(hypHypGraph.hyp_graph)

print("\nsearch for communities with the method focusing on the large hyperedges")
hyper_comm_struct = HP.hyperedge_percolation_largeEdgeFocus(6, None) #create a list of nodesets corresponding to communities
print(f"{'Number of detected communities:':<31}{len(hyper_comm_struct):>7}")
print(f"{'Smallest community size:':<31}{min(list(map(len, hyper_comm_struct))):>7}")
print(f"{'Largest community size:':<31}{max(list(map(len, hyper_comm_struct))):>7}")
print(f"{'Average community size:':<31}{round(np.mean(list(map(len, hyper_comm_struct))), 3):>7}")
hypHypGraph.plot_hyperedge_communities(hyper_comm_struct, sns_color_palette = "Paired", 
                                       node_size = 15, path_filename = os.getcwd() + "/largeEdgeFocus.png")

print("\nsearch for communities with the method focusing on the small hyperedges")
hyper_comm_struct = HP.hyperedge_percolation_smallEdgeFocus(6, 0.75) #create a list of nodesets corresponding to communities
print(f"{'Number of detected communities:':<31}{len(hyper_comm_struct):>7}")
print(f"{'Smallest community size:':<31}{min(list(map(len, hyper_comm_struct))):>7}")
print(f"{'Largest community size:':<31}{max(list(map(len, hyper_comm_struct))):>7}")
print(f"{'Average community size:':<31}{round(np.mean(list(map(len, hyper_comm_struct))), 3):>7}")
hypHypGraph.plot_hyperedge_communities(hyper_comm_struct, sns_color_palette = "Paired", 
                                       node_size = 15, path_filename = os.getcwd()+"/smallEdgeFocus.png")

print("\nsearch for communities on the clique projection of the hypergraph using the original k-clique percolation method")
projected_comm_struct = CPP.clique_percolation_on_projection(6) #create a list of nodesets corresponding to communities
print(f"{'Number of detected communities:':<31}{len(projected_comm_struct):>7}")
print(f"{'Smallest community size:':<31}{min(list(map(len, projected_comm_struct))):>7}")
print(f"{'Largest community size:':<31}{max(list(map(len, projected_comm_struct))):>7}")
print(f"{'Average community size:':<31}{round(np.mean(list(map(len, projected_comm_struct))), 3):>7}\n")
hypHypGraph.plot_projection_clique_communities(projected_comm_struct, CPP.clique_proj, sns_color_palette = "Paired", 
                                               node_size = 15, path_filename = os.getcwd() + "/originalMethodOnProjection.png") 