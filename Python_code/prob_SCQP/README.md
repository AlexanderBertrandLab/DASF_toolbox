# Spherically constrained quadratic problem
 
Folder implementing the following spherically constrained quadratic problem (SCQP) in a distributed setting using the the DSFO framework:
``
P: min_X 0.5 * || E[X.T @ y(t)] || ** 2 + trace(X.T @ B) s.t. trace(X.T @ I @ X) = 1,
``
with the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y(t)` | `nbsensors x 1` for every `t` | Signal |
| `B` | `nbsensors x Q` | Linear term |
| `I` (identity) | `nbsensors x nbsensors` | Quadratic term |

The functions and files in this folder are:

`scqp_solver:` Centralized algorithm for solving the SCQP:

        min_X 0.5*|| E[X'*y1(t)] ||^2+trace(X'*B1) s.t. trace(X'*Gamma1*X)=1.

taking as input the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y1(t)` | `nbsensors x 1` for every `t` | Signal |
| `B1` | `nbsensors x Q` | Linear term |
| `Gamma1` | `nbsensors x nbsensors` | Quadratic term |

**Note:** Requires the Python pymanopt package:

Koep, Niklas, and Sebastian Weichwald. "Pymanopt: A Python Toolbox for Optimization on Manifolds using Automatic Differentiation." Journal of Machine Learning Research 17 (2016): 1-5.
https://www.pymanopt.org

`scqp_eval:`  Evaluate the SCQP objective function.

`run_scqp.py:` Script to run the DSFO algorithm to solve the SCQP in a randomly generated network.

`SCQP_script.ipynb:` Jupyter notebook example.

**How to initialize** `data`**:** We remind the fields of the structure `data`:
| Field | Description |
 | --- | --- |
 | **Signals:** `Y_list` | List for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one element for each different signal. <br /> **Example:** If the problem depends on `X.T @ y(t)` and `X.T @ v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_list[0]=Y` and `Y_list[1]=V`. |
| **Linear terms:** `B_list` | List for deterministic linear terms, where each term has `nbsensors` rows. There is one element for each different parameter. <br />**Example:** If the problem depends on `X.T @ B` and `X.T @ c` then we have `B_list[0]=B` and `B_list[1]=c`. |
| **Quadratic terms:** `Gamma_list` | List for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one element for each different term. <br />**Example:** If the problem depends on `X.T @ X`, `X.T @ Gamma_1 @ X` and `X.T @ Gamma_2 @ X` then we have `Gamma_list[0]=identity(nbsensors)`, `Gamma_list[1]=Gamma_1` and `Gamma_list[2]=Gamma_2`. |
| **Global constants:** `Glob_Const_list` | List for global constants, i.e., terms that do not appear in the form `X.T @ ...`. There is one element for each different term. <br />**Example:** If the problem depends on `X.T @ X-A` and `X.T @ b-c` then we have `Glob_Const_list[0]=A` and `Glob_Const_list[1]=c`. |

If one or more of these do not appear in the problem, set their corresponding list to an empty one.

**Example:** `Gamma_list=[]` if no quadratic term `X.T @ Gamma @ X` appears in the problem.

The function `scqp_solver` depends on the signal `y1(t)`, the linear term `B1` and the quadratic term `Gamma1`.

Looking at problem `P`, the relationship between the data in `P` and the solver are:

|`P <--> qcqp_solver`| `data` |
| --- | --- |
| `y(t)==y1(t)` | `Y=[y[0],...,y[nbsamples-1]]`<br />`data['Y_list'][0]=Y` |
| `B==B1` | `data['B_list'][0]=B` |
| `I==Gamma1` | `data['Gamma_list'][0]=I` |
| | `data['Glob_Const_list']=[]` |