# The DSFO framework

Folder with the DSFO implementation and utility functions (more details inside the code).

`dsfo.m:` Function implementing the DSFO framework taking as arguments:

        - data: Structure containing the data of the problem.

        - prob_params: Structure containing the problem parameters such as the number of nodes, the size of the filter, etc.

        - conv: Structure related the stopping criterion.

        - obj_eval: Function handle to the objective function evaluation. 

        - prob_solver: Function handle to the solver of the problem.

        - prob_resolve_uniqueness: (Optional) Function handle to the method for resolving uniqueness ambiguities.

`find_path.m:` Function finding the neighbors of node q and the shortest path to other every other node in the network.

`shortest_path.m:` Function computing the shortest path distance between a source node and all nodes in the network using Dijkstra's method. Note: This implementation is only for graphs for which the weight at each edge is equal to 1.

`make_sym.m:` Function to force symmetry.

`constr_Nu.m:` Function to obtain clusters of nodes for each neighbor.

`constr_Cq.m:` Function to construct the transition matrix between the local data and variables and the global ones.

`compress.m:` Function to compress the data.

`block_q.m:` Function to extract the block of X corresponding to node q.

**Dependencies:**



                                dsfo
                                  |
                                  |
            -----------------------------------------------------------
           |    |           |           |            |     |    |    |
           |    |           |           |            |     |    |    |
           |    v           v           v            v     |    v    |
           |find_path    constr_Nu    constr_Cq   compress | block_q |
           |    |                                    |     |         |
           |    |                                    |     |         |
           |    v                                    v     |         |
           | shortest_path                         make_sym|         |
           |                                               |         |
           |                                               |         |
           |                                               |         |
           |                                               |         |
           v                                               v         v
        obj_eval                                 prob_solver   prob_resolve_uniqueness