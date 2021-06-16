# Distributed Signal Fusion Optimization
 The distributed signal fusion optimization (DSFO) algorithm framework implementation.

 Given an optimization problem fitting the DSFO framework:

        P: min_X f_hat ( X'*y(t), X'*B, X'*Gamma*X ) = f(X)
           s.t.  h_j ( X'*y(t), X'*B, X'*Gamma*X ) <= 0 for inequalities j,
                 h_j ( X'*y(t), X'*B, X'*Gamma*X ) = 0 for equalities j,

the DSFO algorithm solves the problem in a distributed setting such as a wireless sensor network consisting of nodes connected to each other in a certain way. This is done by creating a local problem at node `q` and iteration `i` and has the advantage that the local problem is a **parameterized** version of problem `P`. Therefore, a solver for problem `P` is used for the distributed implementation.

**Note:** There can be more than one `y(t)`, `B` and `Gamma` which are not represented for conciseness. 

The `dsfo` function implements the DSFO algorithm and is called in the following way:

        [X_est,f_seq,norm_diff,norm_err]=dsfo(prob_params,data,conv,...
        @obj_eval,@prob_solver,@prob_resolve_uniqueness)

### I - Inputs

#### 1) Problem Parameters
`prob_params:` This argument is a structure related to the parameters of the problem, containing the following fields:
| Field | Description |
| --- | --- |
| `nbnodes` | Scalar containing the number of nodes. |
| `nbsensors_vec (nbnodes x 1)` | Vector containing the number of channels/sensors per node. |
| `nbsensors` | Scalar containing the total number of channels/sensors. Sum of the elements of `nbsensors_vec`. |
| `Q`| Scalar containing the number of columns of `X`, the optimization variable. |
| `nbsamples` | Scalar containing the number of time samples per iteration of the signals in the network (e.g. to compute an estimation of the correlation matrix). |
| `graph_adj (nbnodes x nbnodes)`| Adjacency matrix of the graph of the considered network, with `graph_adj(i,j)==1` if nodes `i`and `j` are connected and `0` otherwise. Moreover, `graph_adj(i,i)=0`. |
| `X_star (nbsensors x Q)` | **Optional.** Optimal argument for the optimization problem, computed using a centralized solver, for example `prob_solver` (see I-5) below). Used for comparison purposes, for example to compute the normalized error. |
| `compare_opt` | **Optional.** If the optimal argument `X_star` is known and given, compute the normalized error `norm_err` at each iteration between `X^i` and `X_star` if this variable is equal to `1` (see II-4) below). |
| `plot_dynamic`| **Optional.** If the optimal argument `X_star` is known and given, show a dynamic plot comparing the first column of `X_star` to the first column of `X^i`. |

#### 2) Data
 `data:` This argument is a structure related to the data of the problem, containing the following fields:

 | Field | Description |
 | --- | --- |
 | **Signals:** `Y_cell` | Cell for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one cell for each different signal. <br /> **Example:** If the problem depends on `X'*y(t)` and `X'*v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_cell{1}=Y` and `Y_cell{2}=V`. |
| **Linear terms:** `B_cell` | Cell for deterministic linear terms, where each term has `nbsensors` rows. There is one cell for each different parameter. <br />**Example:** If the problem depends on `X'*B` and `X'*c` then we have `B_cell{1}=B` and `B_cell{2}=c`. |
| **Quadratic terms:** `Gamma_cell` | Cell for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X`, `X'*Gamma_1*X` and `X'*Gamma_2*X` then we have `Gamma_cell{1}=eye(nbsensors)`, `Gamma_cell{2}=Gamma_1` and `Gamma_cell{3}=Gamma_2`. |
| **Global constants:** `Glob_Const_cell` | Cell for global constants, i.e., terms that do not appear in the form `X'*...`. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X-A` and `X'*b-c` then we have `Glob_Const_cell{1}=A` and `Glob_Const_cell{2}=c`. |

If one or more of these do not appear in the problem, set their corresponding cell to an empty one.

**Example:** `Gamma_cell={}` if no quadratic term `X'*Gamma'*X` appears in the problem.

#### 3) Stopping and convergence criteria
`conv:` This argument is a structure related to the stopping and convergence criteria of the algorithm, containing the following fields:

| Field | Description |
| ---- | --- |
| `tol_f`| Tolerance on the difference between consecutive objectives, i.e., `abs(f(X^(i+1))-f(X^i))`. If the difference is smaller than this variable, stop the algorithm. |
| `nbiter` | Maximum number of iterations. Stop the algorithm whenever it is achieved. |

The condition achieved last between the two arguments will stop the algorithm. If only one argument is preferred, the other can be given a negative value, e.g. `-1`.

#### 4) The problem objective evaluator

`obj_eval:` Function evaluating the objective `f` of the problem `P` at the given `X`. The function should be of the form:
        
        f=obj_eval(X,data) % Equal to f(X)

where `data` is the structure defined above.

**Note:** `dsfo` takes the function handle `@obj_eval` as an argument, not the function itself.

#### 5) The problem solver

`prob_sover:` Function solving the optimization problem `P` in a centralized setting and having as outputs `X_star`, the argument solving the problem and `f_star==f(X_star)`. The function should be of the form:

        [X_star,f_star]=prob_solver(data,prob_params)

where `data` and `prob_params` are the structures defined above.

**Note:** `dsfo` takes the function handle `@prob_solver` as an argument, not the function itself.

#### 6) The function resolving uniqueness ambiguities

`prob_resolve_uniqueness:` **Optional.** Function required only when the problem `P` has multiple solutions. Among potential solutions `[X1;...;Xq;...;XK]`, choose the one for which `||Xq_old-Xq||` is minimized when `q` is the updating node, where `X_old==[X1_old;...;Xq_old;...;XK_old]` is the global variable at the previous iteration. The function should be of the form:

        X=prob_resolve_ambiguity(Xq_old,Xq,X)

`X` is the current global variable, i.e., the one chosen by `prob_solver` before resolving the ambiguity.

**Note:** `dsfo` takes the function handle `@prob_resolve_uniqueness` as an argument, not the function itself.

### II - Outputs
#### 1) Estimation of the optimal argument

`X_est:` Estimate of the optimal argument `X_star` obtained using the DSFO framework.

#### 2) Sequence of objectives

`f_seq:` Vector of the sequence of objectives, i.e., `f(X^i)`.

#### 3) Norm of difference of arguments

`norm_diff:` Vector containing the scaled norm of the difference between consecutive arguments, i.e., `||X^(i+1)-X^i||_F^2/(nbsensors*Q)`.

#### 4) Normalized error

`norm_err:` **Computed only if** `X_star` **is provided and** `compare_opt` **is equal to 1.** Vector containing the scaled norm of the difference between `X^i` and `X_star`, i.e., `||X^i-X_star||_F^2/||X_star||_F^2`.

### III - Examples

See the folders `prob_QCQP`, `prob_SCQP` and `prob_TRO`.