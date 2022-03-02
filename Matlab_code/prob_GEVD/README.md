# The Generalized Eigenvalue Decomposition
 
Example implementing the Generalized Eigenvalue Decomposition (GEVD) in a distributed setting using the the DSFO framework:
``
P: max_X E[ || X'*y(t) ||^2 ] s.t. E[ X'*v(t)*v(t)'*X ]=I,
``

with the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y(t)` | `nbsensors x 1` for every `t` | Signal |
| `v(t)` | `nbsensors x 1` for every `t` | Signal |
| `I` (identity) | `nbsensors x nbsensors` | Quadratic term |

The functions and files in this folder are:

`gevd_solver.m:` Centralized algorithm for solving the GEVD problem: 
        
        max_X E[ || X'*y1(t) ||^2 ] s.t. E[ X'*y2(t)*y2(t)'*X ]=I.

taking as input the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y1(t)` | `nbsensors x 1` for every `t` | Signal |
| `y1(t)` | `nbsensors x 1` for every `t` | Signal |
| `Gamma1` | `nbsensors x nbsensors` | Quadratic term |

`gevd_eval.m:`  Evaluate the GEVD objective function.

`gevd_select_sol.m:`  Resolve the uniqueness ambiguity of the GEVD problem, i.e., invariance of the problem to the sign of the columns of `X`.

`run_gevd.m:` Script to run the DSFO algorithm to solve the GEVD problem in a randomly generated network.

`GEVD_script.mlx:` Matlab live script example.

**How to initialize** `data`**:** We remind the fields of the structure `data`:
| Field | Description |
 | --- | --- |
 | **Signals:** `Y_cell` | Cell for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one cell for each different signal. <br /> **Example:** If the problem depends on `X'*y(t)` and `X'*v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_cell{1}=Y` and `Y_cell{2}=V`. |
| **Linear terms:** `B_cell` | Cell for deterministic linear terms, where each term has `nbsensors` rows. There is one cell for each different parameter. <br />**Example:** If the problem depends on `X'*B` and `X'*c` then we have `B_cell{1}=B` and `B_cell{2}=c`. |
| **Quadratic terms:** `Gamma_cell` | Cell for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X`, `X'*Gamma_1*X` and `X'*Gamma_2*X` then we have `Gamma_cell{1}=eye(nbsensors)`, `Gamma_cell{2}=Gamma_1` and `Gamma_cell{3}=Gamma_2`. |
| **Global constants:** `Glob_Const_cell` | Cell for global constants, i.e., terms that do not appear in the form `X'*...`. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X-A` and `X'*b-c` then we have `Glob_Const_cell{1}=A` and `Glob_Const_cell{2}=c`. |

If one or more of these do not appear in the problem, set their corresponding cell to an empty one.

**Example:** `Gamma_cell={}` if no quadratic term `X'*Gamma*X` appears in the problem.

The function `gevd_solver` depends on the signals `y1(t)` and `y2(t)`. There are no linear or quadratic terms nor global constants.

Looking at problem `P`, the relationship between the data in `P` and the solver are:

|`P <--> gevd_solver`| `data` |
| --- | --- |
| `y(t)==y1(t)` | `Y=[y(1),...,y(nbsamples)]`<br />`data.Y_cell{1}=Y` |
| `v(t)==y2(t)` | `V=[V(1),...,V(nbsamples)]`<br />`data.Y_cell{2}=V` |
| `I==Gamma1` | `data.Gamma_cell{1}=I` |
|  | `data.B_cell={}` |
| | `data.Glob_Const_cell={}` |

**Creating the data and parameters:** In the given example code, we take:
``
v(t)=B*s(t)+n(t),
y(t)=A*d(t)+v(t),
``
where the entries of `s(t)` and `d(t)` independently follow `N(0,0.5)`, and the entries of `n(t)` independently follow `N(0,0.1)` for each time instant `t`, where `N` denotes the Gaussian distribution. We take `s` and `d` to be `5`-dimensional. Additionally the entries of `A` and `B` are drawn independently from `U([-0.5,0.5])`.