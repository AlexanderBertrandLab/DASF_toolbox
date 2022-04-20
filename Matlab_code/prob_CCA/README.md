# Canonical Correlation Analysis
 
Folder implementing the following Canonical Correlation Analysis (CCA) in a distributed setting using the the DASF framework:
``
P: min_(X,W) E[  v(t)'*W*X'*y(t) ] s.t. E[ X'*y(t)*y(t)'*X ]=I, E[ W'*v(t)*v(t)'*W ]=I,
``

with the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y(t)` | `nbsensors x 1` for every `t` | Signal |
| `v(t)` | `nbsensors x 1` for every `t` | Signal |

The functions and files in this folder are:

`cca_solver.m:` Centralized algorithm for solving the CCA:

        min_(X1,X2) E[  y2(t)'*X2*X1'*y2(t) ] s.t. E[ X1'*y1(t)*y1(t)'*X1 ]=I, E[ X2'*y2(t)*y2(t)'*X2 ]=I,

taking as input the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y1(t)` | `nbsensors x 1` for every `t` | Signal |
| `y2(t)` | `nbsensors x 1` for every `t` | Signal |

`cca_eval.m:`  Evaluate the CCA objective function.

`run_cca.m:` Script to run the DASF algorithm to solve the CCA in a randomly generated network.

`CCA_script.mlx:` Matlab live script example.

**How to initialize** `data`**:** In this case, `data` will be a cell containing `data_X` and `data_W` which are structures with the same fields as the `data` structure for problems using the `dasf` function. We have one structure per variable. We remind the fields of the structures `data_X` and `data_W`:
| Field | Description |
 | --- | --- |
 | **Signals:** `Y_cell` | Cell for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one cell for each different signal. <br /> **Example:** If the problem depends on `X'*y(t)` and `X'*v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_cell{1}=Y` and `Y_cell{2}=V`. |
| **Linear terms:** `B_cell` | Cell for deterministic linear terms, where each term has `nbsensors` rows. There is one cell for each different parameter. <br />**Example:** If the problem depends on `X'*B` and `X'*c` then we have `B_cell{1}=B` and `B_cell{2}=c`. |
| **Quadratic terms:** `Gamma_cell` | Cell for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X`, `X'*Gamma_1*X` and `X'*Gamma_2*X` then we have `Gamma_cell{1}=eye(nbsensors)`, `Gamma_cell{2}=Gamma_1` and `Gamma_cell{3}=Gamma_2`. |
| **Global constants:** `Glob_Const_cell` | Cell for global constants, i.e., terms that do not appear in the form `X'*...`. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X-A` and `X'*b-c` then we have `Glob_Const_cell{1}=A` and `Glob_Const_cell{2}=c`. |

If one or more of these do not appear in the problem, set their corresponding cell to an empty one.

**Example:** `Gamma_cell={}` if no quadratic term `X'*Gamma*X` appears in the problem.

The function `cca_solver` depends on the signals `y1(t)` and `y2(t)`. There are no linear or quadratic terms, nor global constants.

Looking at problem `P`, the relationship between the data in `P` and the solver are:

|`P <--> cca_solver`| `data` |
| --- | --- |
| `y(t)==y1(t)` | `Y=[y(1),...,y(nbsamples)]`<br />`data_X.Y_cell{1}=Y` |
| `v(t)==y2(t)` | `V=[v(1),...,v(nbsamples)]`<br />`data_W.Y_cell{1}=V` |
| | `data_X.B_cell={}`<br />`data_W.B_cell={}` |
| | `data_X.Glob_Const_cell={}`<br /> `data_W.Glob_Const_cell={}`|
| | `data_X.Gamma_cell={}`<br />`data_W.Gamma_cell={}` |

**Creating the data and parameters:** In the given example code, we take:
``
v(t)=A*d(t)+n(t),
``
where the entries of `d(t)` independently follow `N(0,0.5)`, and the entries of `n(t)` independently follow `N(0,0.1)` for each time instant `t`, where `N` denotes the Gaussian distribution. We take `d` to be `10`-dimensional. Additionally the entries of `A` are drawn independently from `U([-0.5,0.5])`. We take `y(t)=v(t-3)`.