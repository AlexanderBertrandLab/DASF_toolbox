# Distributed Adaptive Signal Fusion Algorithm
 The distributed adaptive signal fusion (DASF) algorithm framework implementation.

 ### Note

For detailed examples, see the folders `CCA`, `GEVD`, `LCMV`, `LS`, `QCQP`, `SCQP` and `TRO`.

Given an optimization problem fitting the DASF framework:

        P: min_X f_hat ( X'*y(t), X'*B, X'*Gamma*X ) = f(X)
           s.t.  h_j ( X'*y(t), X'*B, X'*Gamma*X ) <= 0 for inequalities j,
                 h_j ( X'*y(t), X'*B, X'*Gamma*X ) = 0 for equalities j,

the DASF algorithm solves the problem in a distributed setting such as a wireless sensor network consisting of nodes connected to each other in a certain way. This is done by creating a local problem at node `q` and iteration `i` and has the advantage that the local problem is a **parameterized** version of problem `P`. Therefore, a solver for problem `P` is used for the distributed implementation.

**Note:** There can be more than one `y(t)`, `B` and `Gamma` which are not represented for conciseness. 

The `dasf` function implements the DASF algorithm and is called in the following way:

        [X_est,norm_diff,norm_err,f_seq]=dasf(prob_params,data,...
        @prob_solver,conv,@prob_select_sol,@prob_eval)

Alternatively, `dasf_block` can be used for the same result, the difference being an additional cell structure for the variable `X` called `X_block` separating each local variable `Xk` of node `k`, such that `X=[X1;...;Xk;...;XK]` into separate blocks for better flexibility. At each iteration, `X_block` is updated from `update_X_block` where each block is updated separately. Modifying the updating scheme to adapt it to the user's applications is then made easier compared to `dasf` at the expense of a slightly less straightforward code. This function is called in the same way as `dasf`:

        [X_est,norm_diff,norm_err,f_seq]=dasf_block(prob_params,data,...
        @prob_solver,conv,@prob_select_sol,@prob_eval)

On the other hand, `dasf_multivar` is the same function as `dasf` but used for the case where there is a "set" of optimization variables, for example `(X,W)` in the case of the Canonical Correlation Analysis problem (see folder `prob_CCA` for an example). In this case, `prob_params` contains an extra filed called `nbvariables`, which gives the number of variables (for example `2` for CCA). If given, `X_star`should be a `nbvariables x 1` cell containing the optimal values. The output `X_est` is also a `nbvariables x 1` cell. The difference of arguments `norm_diff` and error `norm_err` (see II-2 and II-3) are computed over the full variable set.This function is called the same way as the previous ones:

        [X_est,norm_diff,norm_err,f_seq]=dasf_multivar(prob_params,data,...
        @prob_solver,conv,@prob_select_sol,@prob_eval)

### I - Inputs

#### 1) Problem Parameters
`prob_params:` Structure related to the parameters of the problem, containing the following fields:
| Field | Description |
| --- | --- |
| `nbnodes` | Scalar containing the number of nodes. |
| `nbsensors_vec (nbnodes x 1)` | Vector containing the number of channels/sensors per node. |
| `nbsensors` | Scalar containing the total number of channels/sensors. Sum of the elements of `nbsensors_vec`. |
| `Q`| Scalar containing the number of columns of `X`, the optimization variable. |
| `nbsamples` | Scalar containing the number of time samples per iteration of the signals in the network (e.g. to compute an estimation of the correlation matrix). |
| `graph_adj (nbnodes x nbnodes)`| Adjacency matrix of the graph of the considered network, with `graph_adj(i,j)==1` if nodes `i`and `j` are connected and `0` otherwise. Moreover, `graph_adj(i,i)=0`. |
| `nbvariables`| **Required only if `dasf_multivar` is used.** Number of different variables (e.g. the optimization variable set of the CCA problem is `(X,W)`, therefore `nbvariables==2`). |
| `X_init (nbsensors x Q)` | **Optional.** Initial value for `X`. If not specified, `X` is initialized randomly. **If `dasf_multivar` is used:** `X_init` is a `nbvariables x 1` cell where each entry is the  `(nbsensors x Q)` initial value matrix for the corresponding variable. |
| `X_star (nbsensors x Q)` | **Optional.** Optimal argument for the optimization problem, computed using a centralized solver, for example `prob_solver` (see I-5 below). Used for comparison purposes, for example to compute the normalized error. **If `dasf_multivar` is used:** `X_star` is a `nbvariables x 1` cell where each entry is the  `(nbsensors x Q)` optimal matrix for the corresponding variable. |
| `compare_opt` | **Optional.** Binary value. If the optimal argument `X_star` is known and given, compute the normalized error `norm_err` at each iteration between `X^i` and `X_star` if this variable is "true" (see II-3 below). "false" by default. |
| `plot_dynamic`| **Optional.** Binary value. If the optimal argument `X_star` is known and given, show a dynamic plot comparing the first column of `X_star` to the first column of `X^i` if this variable is "true". "false" by default. |

#### 2) Data
 `data:` Structure related to the data of the problem, containing the following fields:

 | Field | Description |
 | --- | --- |
 | **Signals:** `Y_cell` | Cell for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one cell for each different signal. <br /> **Example:** If the problem depends on `X'*y(t)` and `X'*v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_cell{1}=Y` and `Y_cell{2}=V`. |
| **Linear terms:** `B_cell` | Cell for deterministic linear terms, where each term has `nbsensors` rows. There is one cell for each different parameter. <br />**Example:** If the problem depends on `X'*B` and `X'*c` then we have `B_cell{1}=B` and `B_cell{2}=c`. |
| **Quadratic block-diagonal terms:** `Gamma_cell` | Cell for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X`, `X'*Gamma_1*X` and `X'*Gamma_2*X` then we have `Gamma_cell{1}=eye(nbsensors)`, `Gamma_cell{2}=Gamma_1` and `Gamma_cell{3}=Gamma_2`. **Note:** If we have `Gamma` such that `Gamma=B_1*B_2'`, it is equivalent to use linear terms instead of quadratic block-diagonal ones. The advantage with the quadratic ones is the reduction of the communication cost.|
| **Global constants:** `Glob_Const_cell` | Cell for global constants, i.e., terms that do not appear in the form `X'*...`. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X-A` and `X'*b-c` then we have `Glob_Const_cell{1}=A` and `Glob_Const_cell{2}=c`. |

If one or more of these do not appear in the problem, set their corresponding cell to an empty one.

**Example:** `Gamma_cell={}` if no quadratic block-diagonal term `X'*Gamma*X` appears in the problem.

**If `dasf_multivar` is used:** In this case, `data` is a `nbvariables x 1` cell containing the same structure described above for each variable.

#### 3) The problem solver

`prob_sover:` Function solving the optimization problem `P` in a centralized setting and having as output `X_star`, the argument solving the problem. The function should be of the form:

        X_star=prob_solver(data,prob_params)

where `data` and `prob_params` are the structures defined above.

**Note:** `dasf` takes the function handle `@prob_solver` as an argument, not the function itself.

#### 4) Stopping and convergence criteria
`conv:` **Optional.** Structure related to the stopping and convergence criteria of the algorithm, containing the following fields:

| Field | Description |
| ---- | --- |
| `nbiter` | Maximum number of iterations. Stop the algorithm whenever it is achieved. |
| `tol_f`| Tolerance on the difference between consecutive objectives, i.e., `abs(f(X^(i+1))-f(X^i))`. If the difference is smaller than this variable, stop the algorithm. |
| `tol_X`| Tolerance on the difference between consecutive arguments, i.e., `||X^(i+1))-X^i||_F`. If the difference is smaller than this variable, stop the algorithm. |

By default, the algorithm stops at maximum 200 iterations. If one or more fields are provided and valid, the algorithm stops when the first stopping criterion is achieved.

#### 5) The function resolving uniqueness ambiguities

`prob_select_sol:` **Optional/Problem dependent.** Function required only when the problem `P` has multiple solutions. Among potential solutions `X`, choose the one for which `||X-X_ref||` is minimized. The function should be of the form:

        X_tilde=prob_select_sol(X_ref,X,prob_params,q)

For flexibility, the updating node `q` can also be specified.

**Note:** `dasf` takes the function handle `@prob_select_sol` as an argument, not the function itself.

#### 6) The problem objective evaluator

`prob_eval:` **Optional.** Function evaluating the objective `f` of the problem `P` at the given `X`. The function should be of the form:
        
        f=prob_eval(X,data) % Equal to f(X)

where `data` is the structure defined above.

**Note:** `dasf` takes the function handle `@prob_eval` as an argument, not the function itself.

### II - Outputs
#### 1) Estimation of the optimal argument

`X_est:` Estimate of the optimal argument `X_star` obtained using the DASF framework.

#### 2) Norm of difference of arguments

`norm_diff:` Vector containing the scaled norm of the difference between consecutive arguments, i.e., `||X^(i+1)-X^i||_F^2/(nbsensors*Q)`.

#### 3) Normalized error

`norm_err:` **Computed only if** `X_star` **is provided and** `compare_opt` **is "true".** Vector containing the scaled norm of the difference between `X^i` and `X_star`, i.e., `||X^i-X_star||_F^2/||X_star||_F^2`.

#### 4) Sequence of objectives

`f_seq:`  **Computed only if** `prob_eval` **is provided.** Vector of the sequence of objectives, i.e., `f(X^i)`.
