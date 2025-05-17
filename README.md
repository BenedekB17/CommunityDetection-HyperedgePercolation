# CommunityDetection-HyperedgePercolation
Python code for community detection in hypergraphs through hyperedge percolation, and for hypergraph generation on hyperbolic plane.

---

The example file generates a hypergraph on the native representation of the hyperbolic plane and searches for its communities with the hyperedge percolation method focusing on large hyperedges, with the hyperedge percolation method focusing on small hyperedges, and on the clique projection of the generated hypergraph using the original *k*-clique percolation method. The detected communities are visualized on the native representation of the hyperbolic plane.


## Community detection in hypergraphs based on hyperedge percolation
The hyperedge percolation method focusing on high-cardinality hyperedges should be used e.g. in the presence of a disproportionate amount of small hyperedges. The input parameters of this method are the minimum cardinality of the hyperedges to be considered (*k*, integer) and the minimum absolute intersection of two hyperedges that is required for joining a community (*I\_abs*, integer). The parameter *k* cannot be larger than the number of nodes in the largest hyperedge of the given hypergraph. The allowed smallest value of *I\_abs* is *1*, while its allowed largest value is *k-1*. 

The hyperedge percolation method focusing on low-cardinality hyperedges should be used e.g. when there are hyperedges presumably larger than the communities being sought, or even comparable in their size to the total number of nodes in the examined system. The input parameters of this method are the maximum cardinality of the hyperedges to be considered (*K*, integer) and the minimum relative intersection of a hyperedge and a community that is required for joining (*I\_rel*, float). The parameter *K* cannot be smaller than the number of nodes in the smallest hyperedge of the given hypergraph. The parameter *I\_rel* should be chosen from the range *[0.5,1.0)*. 

Both methods produce a list of node sets, where each set corresponds to a detected community. The detected communities might overlap, i.e. a node can appear in multiple node sets. There might be nodes that do not appear in any of the detected communities (these unclustered nodes might be identified as individual communities). Both community detection methods are deterministic, yielding the same community structure for a given hypergraph at every run. 


# Hyperbolic hypergraph generation
The tunable model parameters are the following:
- The total number of nodes: N\_nodes (integer). Note that it is not guaranteed that there will be no node that does not belong to any hyperedge. 
- The target average hyperdegree: avg\_hyp\_deg (float). This is the expected number of hyperedges containing a randomly chosen node. 
- The parameter of the nodes' radial distribution: alpha (float, larger than 0.5). A larger value yields a faster decay in the hyperdegree distribution. 
- The type of the hyperedge cardinality distribution: edge\_size\_distr (string).
	- If edge\_size\_distr is "DD": Dirac delta distribution parametrised by the single allowed hyperedge size (s\_min, integer)
	- If edge\_size\_distr is "UNI": uniform distribution parametrised by the allowed smallest (s\_min, integer) and largest (s\_max, integer) hyperedge size
	- If edge\_size\_distr is "BIN": binomial distribution parametrised by the allowed smallest (s\_min, integer) and largest (s\_max, integer) hyperedge size, and the expected average cardinality (s\_avg, float)
	- If edge\_size\_distr is "POW": power-law distribution parametrised by the allowed smallest (s\_min, integer) and largest (s\_max, integer) hyperedge size, and the decay exponent (gamma\_s, float) of the distribution *P(s)~s<sup>-gamma\_s<sup>*

Note that the maximum number of different hyperedges of a given size depends on the actual spatial arrangement of the nodes (and is usually smaller than the number *N\_nodes* of possible hyperedge centres), and if the required hyperedge cardinality distribution is not feasible in a given hypergraph, then this implementation does not stick strictly to the exact value of the expected total number of hyperedges but rather tries to adhere to the shape of the required hyperedge size distribution as much as possible. 

The hypergraph generation is not deterministic, each run yields a different hypergraph.


# Requirements
Python>=3.6, numpy>=1.3.0, seaborn>=0.2.0, matplotlib>=1.0.0, scipy>=0.18.0, networkx>=2.0, hypergraphx>=1.7.2


# Reference
[Kovács, B., Benedek, B. & Palla, G. *Community detection in hypergraphs through hyperedge percolation*. arXiv:2504.15213 [physics.soc-ph] (2025)](https://doi.org/10.48550/arXiv.2504.15213)

For any problem, please contact Bianka Kovács: <kovacs.bianka@ecolres.hu>
