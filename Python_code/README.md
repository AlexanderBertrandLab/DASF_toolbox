# Distributed Signal Fusion Optimization
 The distributed signal fusion optimization (DSFO) algorithm framework implementation.

 Given an optimization problem fitting the DSFO framework:

        P: min_X f_hat ( X.T @ y(t), X.T @ B, X.T @ Gamma @ X ) = f(X)
           s.t.  h_j ( X.T @ y(t), X .T @ B, X.T @ Gamma @ X ) <= 0 for inequalities j,
                 h_j ( X.T @ y(t), X.T @ B, X.T @ Gamma @ X ) = 0 for equalities j,

the DSFO algorithm solves the problem in a distributed setting such as a wireless sensor network consisting of nodes connected to each other in a certain way. This is done by creating a local problem at node `q` and iteration `i` and has the advantage that the local problem is a **parameterized** version of problem `P`. Therefore, a solver for problem `P` is used for the distributed implementation.

**Note:** There can be more than one `y(t)`, `B` and `Gamma` which are not represented for conciseness. 

The `dsfo` function implements the DSFO algorithm and is called in the following way:

        X_est,norm_diff,norm_err,f_seq=dsfo(prob_params,data,...
        prob_solver,conv,prob_select_sol,prob_eval)

Alternatively, `dsfo_block` can be used for the same result, the difference being an additional list for the variable `X` called `X_block` separating each local variable `Xk` of node `k`, such that `X=[X1;...;Xk;...;XK]` into separate blocks for better flexibility. At each iteration, `X_block` is updated from `update_X_block` where each block is updated separately. Modifying the updating scheme to adapt it to the user's applications is then made easier compared to `dsfo` at the expense of a slightly less straightforward code. This function is called in the same way as `dsfo`:

        X_est,norm_diff,norm_err,f_seq=dsfo_block(prob_params,data,...
        prob_solver,conv,prob_select_sol,prob_eval)

### I - Inputs

#### 1) Problem Parameters
`prob_params:` Dictionary related to the parameters of the problem, containing the following keys:
| Field | Description |
| --- | --- |
| `nbnodes` | Scalar containing the number of nodes. |
| `nbsensors_vec (nbnodes x 1)` | Vector containing the number of channels/sensors per node. |
| `nbsensors` | Scalar containing the total number of channels/sensors. Sum of the elements of `nbsensors_vec`. |
| `Q`| Scalar containing the number of columns of `X`, the optimization variable. |
| `nbsamples` | Scalar containing the number of time samples per iteration of the signals in the network (e.g. to compute an estimation of the correlation matrix). |
| `graph_adj (nbnodes x nbnodes)`| Adjacency matrix of the graph of the considered network, with `graph_adj[i,j]==1` if nodes `i`and `j` are connected and `0` otherwise. Moreover, `graph_adj[i,i]=0`. |
| `X_star (nbsensors x Q)` | **Optional.** Optimal argument for the optimization problem, computed using a centralized solver, for example `prob_solver` (see I-5) below). Used for comparison purposes, for example to compute the normalized error. |
| `compare_opt` | **Optional.** Binary value. If the optimal argument `X_star` is known and given, compute the normalized error `norm_err` at each iteration between `X^i` and `X_star` if this variable is "True" (see II-3) below). "False" by default. |
| `plot_dynamic`| **Optional.** Binary value. If the optimal argument `X_star` is known and given, show a dynamic plot comparing the first column of `X_star` to the first column of `X^i` if this variable is "True". "False" by default. |

#### 2) Data
 `data:` Dictionary related to the data of the problem, containing the following keys:

 | Field | Description |
 | --- | --- |
 | **Signals:** `Y_list` | List for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one element for each different signal. <br /> **Example:** If the problem depends on `X.T @ y(t)` and `X.T @ v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_list[0]=Y` and `Y_list[1]=V`. |
| **Linear terms:** `B_list` | List for deterministic linear terms, where each term has `nbsensors` rows. There is one element for each different parameter. <br />**Example:** If the problem depends on `X.T @ B` and `X.T @ c` then we have `B_list[0]=B` and `B_list[1]=c`. |
| **Quadratic terms:** `Gamma_list` | List for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one element for each different term. <br />**Example:** If the problem depends on `X.T @ X`, `X.T @ Gamma_1 @ X` and `X.T @ Gamma_2 @ X` then we have `Gamma_list[0]=identity(nbsensors)`, `Gamma_list[1]=Gamma_1` and `Gamma_list[2]=Gamma_2`. |
| **Global constants:** `Glob_Const_list` | List for global constants, i.e., terms that do not appear in the form `X.T @ ...`. There is one element for each different term. <br />**Example:** If the problem depends on `X.T @ X-A` and `X.T @ b-c` then we have `Glob_Const_list[0]=A` and `Glob_Const_list[1]=c`. |

If one or more of these do not appear in the problem, set their corresponding list to an empty one.

**Example:** `Gamma_list=[]` if no quadratic term `X.T @ Gamma @ X` appears in the problem.

#### 3) The problem solver

`prob_sover:` Function solving the optimization problem `P` in a centralized setting and having as output `X_star`, the argument solving the problem. The function should be of the form:

        X_star=prob_solver(data,prob_params)

where `data` and `prob_params` are the dictionaries defined above.

#### 4) Stopping and convergence criteria
`conv:` **Optional.** Dictionary related to the stopping and convergence criteria of the algorithm, containing the following keys:

| Field | Description |
| ---- | --- |
| `nbiter` | Maximum number of iterations. Stop the algorithm whenever it is achieved. |
| `tol_f`| Tolerance on the difference between consecutive objectives, i.e., `abs(f(X^(i+1))-f(X^i))`. If the difference is smaller than this variable, stop the algorithm. |
| `tol_X`| Tolerance on the difference between consecutive arguments, i.e., `||X^(i+1))-X^i||_F`. If the difference is smaller than this variable, stop the algorithm. |

By default, the algorithm stops at maximum 200 iterations. If one or more fields are provided and valid, the algorithm stops when the first stopping criterion is achieved.

#### 5) The function resolving uniqueness ambiguities

`prob_select_sol:` **Optional.** Function required only when the problem `P` has multiple solutions. Among potential solutions `[X1;...;Xq;...;XK]`, choose the one for which `||Xq_old-Xq||` is minimized when `q` is the updating node, where `X_old==[X1_old;...;Xq_old;...;XK_old]` is the global variable at the previous iteration. The function should be of the form:

        X=prob_select_sol(Xq_old,Xq,X)

`X` is the current global variable, i.e., the one chosen by `prob_solver` before resolving the ambiguity.

#### 6) The problem objective evaluator

`prob_eval:` **Optional.** Function evaluating the objective `f` of the problem `P` at the given `X`. The function should be of the form:
        
        f=prob_eval(X,data) % Equal to f(X)

where `data` is the dictionary defined above.

### II - Outputs
#### 1) Estimation of the optimal argument

`X_est:` Estimate of the optimal argument `X_star` obtained using the DSFO framework.

#### 2) Norm of difference of arguments

`norm_diff:` Vector containing the scaled norm of the difference between consecutive arguments, i.e., `||X^(i+1) - X^i||_F ** 2 / (nbsensors * Q)`.

#### 3) Normalized error

`norm_err:` **Computed only if** `X_star` **is provided and** `compare_opt` **is "true".** Vector containing the scaled norm of the difference between `X^i` and `X_star`, i.e., `||X^i - X_star||_F ** 2 / ||X_star||_F ** 2`.

#### 4) Sequence of objectives

`f_seq:`  **Computed only if** `prob_eval` **is provided.** Vector of the sequence of objectives, i.e., `f(X^i)`.

### III - Examples

See the folders `prob_QCQP`, `prob_SCQP` and `prob_TRO`.