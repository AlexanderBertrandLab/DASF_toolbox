# The DSFO framework

The file `dsfo_toolbox.py` contains the DSFO implementation (more details inside the code).

`dsfo:` Function implementing the DSFO framework taking as arguments:

        - prob_params: Structure containing the problem parameters such as the number of nodes, the size of the filter, etc.
        
        - data: Structure containing the data of the problem.

        - prob_solver: Function handle to the solver of the problem.

        - conv: (Optional) Structure related the stopping criterion.

        - prob_select_sol: (Optional) Function handle to the method for resolving uniqueness ambiguities.

        - prob_eval: (Optional) Function handle to the objective function evaluation. 

`find_path:` Function finding the neighbors of node q and the shortest path to other every other node in the network.

`shortest_path:` Function computing the shortest path distance between a source node and all nodes in the network using Dijkstra's method. Note: This implementation is only for graphs for which the weight at each edge is equal to 1.

`find_clusters:` Function to obtain clusters of nodes for each neighbor.

`build_Cq:` Function to construct the transition matrix between the local data and variables and the global ones.

`compress:` Function to compress the data.

`block_q:` Function to extract the block of X corresponding to node q.

`dsfo_multivar:` Same function as `dsfo` but for problems with multiple variables (e.g. Canonical Correlation Analysis). The output `X` is a cell containing multiple variables.

`dsfo_block:` Same function as `dsfo` but the optimization variable is divided into cells *within the function* to explicitly emphasize and separate the block structure of the global variable `X=[X1;...;Xk;...XK]`. At the expense of a slightly less straightforward implementation than `dsfo`, it better represents how each node updates their local variable.

`update_X_block:` Only called from `dsfo_block`. Explicitly updating the `Xk` of each node `k` separately, where the global variable `X` is equal to `[X1;...;Xk;...XK]`, it allows to adapt the updating scheme depending on the user's application in an easier way than the implementation used in `dsfo`, resulting in more flexibility. 

**Dependencies:**



                                dsfo
                                  |
                                  |
           ----------------------------------------------------------------
           |    |           |           |            |     |    |    |    |
           |    |           |           |            |     |    |    |    |
           |    v           v           v            v     |    v    |    v
           |find_path  find_clusters   build_Cq   compress | block_q |  plot_dynamic
           |    |                                    |     |         |
           |    |                                    |     |         |
           |    v                                    v     |         |
           | shortest_path                         make_sym|         |
           |                                               |         |
           |                                               |         |
           |                                               |         |
           |                                               |         |
           v                                               v         v
        prob_eval                                 prob_solver   prob_select_sol





                              dsfo_block
                                  |
                                  |
           ------------------------------------------------------------------------------
           |    |        |          |          |     |    |    |         |    |         |
           |    |        |          |          |     |    |    |         |    |         |
           |    v        v          v          v     |    |    v         |    v         v
           |find_path find_clusters build_Cq compress|    |update_X_block| block_q plot_dynamic
           |    |                                    |    |          |   |         
           |    |                                    |    |          |   |         
           |    v                                    v    |          |   |         
           | shortest_path                        make_sym|          |   |
           |                                              |          |   |
           |                                              |          |   |
           |                                              |          |   |
           |                                              |          |   |
           v                                              v          v   v
        prob_eval                                    prob_solver   prob_select_sol
