# Quadratically constrained quadratic problem
 
Folder implementing the following quadratically constrained quadratic problem (QCQP) in a distributed setting using the the DSFO framework:
``
P: min_X 0.5*|| E[X'*y(t)] ||^2-trace(X'*B) s.t. trace(X'*I*X)<=alpha^2, X'*c=d,
``

with the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y(t)` | `nbsensors x 1` for every `t` | Signal |
| `B` | `nbsensors x Q` | Linear term |
| `c` | `nbsensors x 1` | Linear term |
| `I` (identity) | `nbsensors x nbsensors` | Quadratic term |
| `alpha` | Scalar | Global constant |
| `d` | Scalar | Global constant |


The functions and files in this folder are:

`qcqp_solver.m:` Centralized algorithm for solving the QCQP:

        min_X 0.5*|| E[X'*y1(t)] ||^2-trace(X'*B1) s.t. trace(X'*Gamma1*X)<=gc1^2, X'*b2=gc2.

taking as input the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y1(t)` | `nbsensors x 1` for every `t` | Signal |
| `B1` | `nbsensors x Q` | Linear term |
| `b2` | `nbsensors x 1` | Linear term |
| `Gamma1` | `nbsensors x nbsensors` | Quadratic term |
| `gc1` | Scalar | Global constant |
| `gc2` | Scalar | Global constant |

`qcqp_eval.m:`  Evaluate the QCQP objective function.

`run_qcqp.m:` Script to run the DSFO algorithm to solve the QCQP in a randomly generated network.

`QCQP_script.mlx:` Matlab live script example.

**How to initialize** `data`**:** We remind the fields of the structure `data`:
| Field | Description |
 | --- | --- |
 | **Signals:** `Y_cell` | Cell for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one cell for each different signal. <br /> **Example:** If the problem depends on `X'*y(t)` and `X'*v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_cell{1}=Y` and `Y_cell{2}=V`. |
| **Linear terms:** `B_cell` | Cell for deterministic linear terms, where each term has `nbsensors` rows. There is one cell for each different parameter. <br />**Example:** If the problem depends on `X'*B` and `X'*c` then we have `B_cell{1}=B` and `B_cell{2}=c`. |
| **Quadratic terms:** `Gamma_cell` | Cell for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X`, `X'*Gamma_1*X` and `X'*Gamma_2*X` then we have `Gamma_cell{1}=eye(nbsensors)`, `Gamma_cell{2}=Gamma_1` and `Gamma_cell{3}=Gamma_2`. |
| **Global constants:** `Glob_Const_cell` | Cell for global constants, i.e., terms that do not appear in the form `X'*...`. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X-A` and `X'*b-c` then we have `Glob_Const_cell{1}=A` and `Glob_Const_cell{2}=c`. |

If one or more of these do not appear in the problem, set their corresponding cell to an empty one.

**Example:** `Gamma_cell={}` if no quadratic term `X'*Gamma*X` appears in the problem.

The function `qcqp_solver` depends on the signal `y1(t)`, the linear terms `B1` and `b2`, the quadratic term `Gamma1`, and the global constants `gc1` and `gc2`.

Looking at problem `P`, the relationship between the data in `P` and the solver are:

|`P <--> qcqp_solver`| `data` |
| --- | --- |
| `y(t)==y1(t)` | `Y=[y(1),...,y(nbsamples)]`<br />`data.Y_cell{1}=Y` |
| `B==B1` | `data.B_cell{1}=B` |
| `c==b2` | `data.B_cell{2}=c` |
| `I==Gamma1` | `data.Gamma_cell{1}=I` |
| `alpha==gc1`| `data.Glob_Const_cell{1}=alpha` |
| `d==gc2` | `data.Glob_Const_cell{2}=d` |