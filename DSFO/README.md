# The DSFO framework

Folder with the DSFO implementation and utility functions (more details inside the code).

`dsfo:` Function implementing the DSFO framework taking as arguments:

        - data: Structure containing the data of the problem.

        - prob_params: Structure containing the problem parameters such as the number of nodes, the size of the filter, etc.

        - conv: Structure related the stopping criterion.

        - prob_solver: Function handle to the solver of the problem.

        - obj_eval: Function handle to the objective function evaluation. 

        - prob_resolve_uniqueness: (Optional) Function handle to the method for resolving uniqueness ambiguities.

        - X_star: (Optional) Matrix which solves the problem, for comparison purposes.

`shortest_path:` Shortest path algorithm

`make_sym:` Force matrix to be symmetric