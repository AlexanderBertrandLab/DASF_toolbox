# The Generalized Eigenvalue Decomposition
 
Example implementing the Generalized Eigenvalue Decomposition (GEVD) in a distributed setting using the the DSFO framework:
``
P: max_X E[ || X.T @ y(t) || ** 2 ] s.t. E[ X.T @ v(t) @ v(t).T @ X ] = I,
``

with the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y(t)` | `nbsensors x 1` for every `t` | Signal |
| `v(t)` | `nbsensors x 1` for every `t` | Signal |
| `I` (identity) | `nbsensors x nbsensors` | Quadratic term |

The functions and files in this folder are:

`gevd_solver:` Centralized algorithm for solving the GEVD problem: 
        
        max_X E[ || X.T @ y1(t) || ** 2 ] s.t. E[ X.T @ y2(t) @ y2(t).T @ X ] = I.

taking as input the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y1(t)` | `nbsensors x 1` for every `t` | Signal |
| `y2(t)` | `nbsensors x 1` for every `t` | Signal |
| `Gamma1` | `nbsensors x nbsensors` | Quadratic term |

`gevd_eval:`  Evaluate the GEVD objective function.

`gevd_select_sol:`  Resolve the uniqueness ambiguity of the GEVD problem, i.e., invariance of the problem to the sign of the columns of `X`.

`run_gevd.py:` Script to run the DSFO algorithm to solve the GEVD problem in a randomly generated network.

`GEVD_notebook.ipynb:` Jupyter notebook example.

**How to initialize** `data`**:** We remind the fields of the structure `data`:
| Field | Description |
 | --- | --- |
 | **Signals:** `Y_list` | List for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one element for each different signal. <br /> **Example:** If the problem depends on `X.T @ y(t)` and `X.T @ v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_list[0]=Y` and `Y_list[1]=V`. |
| **Linear terms:** `B_list` | List for deterministic linear terms, where each term has `nbsensors` rows. There is one element for each different parameter. <br />**Example:** If the problem depends on `X.T @ B` and `X.T @ c` then we have `B_list[0]=B` and `B_list[1]=c`. |
| **Quadratic terms:** `Gamma_list` | List for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one element for each different term. <br />**Example:** If the problem depends on `X.T @ X`, `X.T @ Gamma_1 @ X` and `X.T @ Gamma_2 @ X` then we have `Gamma_list[0]=identity(nbsensors)`, `Gamma_list[1]=Gamma_1` and `Gamma_list[2]=Gamma_2`. |
| **Global constants:** `Glob_Const_list` | List for global constants, i.e., terms that do not appear in the form `X.T @ ...`. There is one element for each different term. <br />**Example:** If the problem depends on `X.T @ X-A` and `X.T @ b-c` then we have `Glob_Const_list[0]=A` and `Glob_Const_list[1]=c`. |

If one or more of these do not appear in the problem, set their corresponding list to an empty one.

**Example:** `Gamma_list=[]` if no quadratic term `X.T @ Gamma @ X` appears in the problem.

The function `gevd_solver` depends on the signals `y1(t)` and `y2(t)`. There are no linear or quadratic terms nor global constants.

Looking at problem `P`, the relationship between the data in `P` and the solver are:

|`P <--> gevd_solver`| `data` |
| --- | --- |
| `y(t)==y1(t)` | `Y=[y[0],...,y[nbsamples-1]]`<br />`data['Y_list'][0]=Y` |
| `v(t)==y2(t)` | `V=[v[0],...,v[nbsamples-1]]`<br />`data['Y_list'][1]=V` |
| `I==Gamma1` | `data['Gamma_list'][0]=I` |
|  | `data['B_list']=[]` |
| | `data['Glob_Const_list']=[]` |

**Creating the data and parameters:** In the given example code, we take:
``
v(t) = B @ s(t) + n(t),
y(t) = A @ d(t) + v(t),
``
where the entries of `s(t)` and `d(t)` independently follow `N(0,0.5)`, and the entries of `n(t)` independently follow `N(0,0.1)` for each time instant `t`, where `N` denotes the Gaussian distribution. We take `s` and `d` to be `5`-dimensional. Additionally the entries of `A` and `B` are drawn independently from `U([-0.5,0.5])`.