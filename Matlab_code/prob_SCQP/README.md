# Spherically Constrained Quadratic Problem
 
Folder implementing the following Spherically Constrained Quadratic Problem (SCQP) in a distributed setting using the the DASF framework:
``
P: min_X 0.5*E[ || X'*y(t) ||^2 ]+trace(X'*B) s.t. trace(X'*I*X)=1,
``
with the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y(t)` | `nbsensors x 1` for every `t` | Signal |
| `B` | `nbsensors x Q` | Linear term |
| `I` (identity) | `nbsensors x nbsensors` | Quadratic term |

The functions and files in this folder are:

`scqp_solver.m:` Centralized algorithm for solving the SCQP:

        min_X 0.5*E[ || X'*y1(t) ||^2 ]+trace(X'*B1) s.t. trace(X'*Gamma1*X)=1.

taking as input the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y1(t)` | `nbsensors x 1` for every `t` | Signal |
| `B1` | `nbsensors x Q` | Linear term |
| `Gamma1` | `nbsensors x nbsensors` | Quadratic term |

**Note:** Requires the Manopt toolbox:

Boumal, Nicolas, et al. "Manopt, a Matlab toolbox for optimization on manifolds." The Journal of Machine Learning Research 15.1 (2014): 1455-1459.
https://www.manopt.org

`scqp_eval.m:`  Evaluate the SCQP objective function.

`run_scqp.m:` Script to run the DASF algorithm to solve the SCQP in a randomly generated network.

`SCQP_script.mlx:` Matlab live script example.

**How to initialize** `data`**:** We remind the fields of the structure `data`:
| Field | Description |
 | --- | --- |
 | **Signals:** `Y_cell` | Cell for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one cell for each different signal. <br /> **Example:** If the problem depends on `X'*y(t)` and `X'*v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_cell{1}=Y` and `Y_cell{2}=V`. |
| **Linear terms:** `B_cell` | Cell for deterministic linear terms, where each term has `nbsensors` rows. There is one cell for each different parameter. <br />**Example:** If the problem depends on `X'*B` and `X'*c` then we have `B_cell{1}=B` and `B_cell{2}=c`. |
| **Quadratic terms:** `Gamma_cell` | Cell for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X`, `X'*Gamma_1*X` and `X'*Gamma_2*X` then we have `Gamma_cell{1}=eye(nbsensors)`, `Gamma_cell{2}=Gamma_1` and `Gamma_cell{3}=Gamma_2`. |
| **Global constants:** `Glob_Const_cell` | Cell for global constants, i.e., terms that do not appear in the form `X'*...`. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X-A` and `X'*b-c` then we have `Glob_Const_cell{1}=A` and `Glob_Const_cell{2}=c`. |

If one or more of these do not appear in the problem, set their corresponding cell to an empty one.

**Example:** `Gamma_cell={}` if no quadratic term `X'*Gamma*X` appears in the problem.

The function `scqp_solver` depends on the signal `y1(t)`, the linear term `B1` and the quadratic term `Gamma1`. There are no global constants.

Looking at problem `P`, the relationship between the data in `P` and the solver are:

|`P <--> scqp_solver`| `data` |
| --- | --- |
| `y(t)==y1(t)` | `Y=[y(1),...,y(nbsamples)]`<br />`data.Y_cell{1}=Y` |
| `B==B1` | `data.B_cell{1}=B` |
| `I==Gamma1` | `data.Gamma_cell{1}=I` |
| | `data.Glob_Const_cell={}` |

**Creating the data and parameters:** In the given example code, we take:
``
y(t)=A*s(t)+n(t),
``
where the entries of `s(t)` independently follow `N(0,0.5)`, and the entries of `n(t)` independently follow `N(0,0.1)` for each time instant `t`, where `N` denotes the Gaussian distribution. Additionally the entries of `A` are drawn independently from `U([-0.5,0.5])`. Finally, the entries of the linear term `B` follow independently `N(0,1)`.