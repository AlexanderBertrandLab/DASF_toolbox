# Linearly Constrained Minimum Variance
 
Folder implementing the following Linearly Constrained Minimum Variance (LCMV) in a distributed setting using the the DSFO framework:
``
P: min_X E[ || X.T @ y(t) || ** 2 ] s.t. X.T @ B = H,
``

with the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y(t)` | `nbsensors x 1` for every `t` | Signal |
| `B` | `nbsensors x Q` | Linear term |
| `H` | `Q x Q` | Global constant |


The functions and files in this folder are:

`lcmv_solver:` Centralized algorithm for solving the LCMV:

        min_X E[ || X.T @ y1(t) || ** 2 ] s.t. X.T @ B1 = GC1.

taking as input the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y1(t)` | `nbsensors x 1` for every `t` | Signal |
| `B1` | `nbsensors x Q` | Linear term |
| `GC1` | `Q x Q` | Global constant |

`lcmv_eval:`  Evaluate the LCMV objective function.

`run_lcmv.py:` Script to run the DSFO algorithm to solve the LCMV in a randomly generated network.

`LCMV_notebook.ipynb:` Jupyter notebook example.

**How to initialize** `data`**:** We remind the fields of the structure `data`:
| Field | Description |
 | --- | --- |
 | **Signals:** `Y_list` | List for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one element for each different signal. <br /> **Example:** If the problem depends on `X.T @ y(t)` and `X.T @ v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_list[0]=Y` and `Y_list[1]=V`. |
| **Linear terms:** `B_list` | List for deterministic linear terms, where each term has `nbsensors` rows. There is one element for each different parameter. <br />**Example:** If the problem depends on `X.T @ B` and `X.T @ c` then we have `B_list[0]=B` and `B_list[1]=c`. |
| **Quadratic terms:** `Gamma_list` | List for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one element for each different term. <br />**Example:** If the problem depends on `X.T @ X`, `X.T @ Gamma_1 @ X` and `X.T @ Gamma_2 @ X` then we have `Gamma_list[0]=identity(nbsensors)`, `Gamma_list[1]=Gamma_1` and `Gamma_list[2]=Gamma_2`. |
| **Global constants:** `Glob_Const_list` | List for global constants, i.e., terms that do not appear in the form `X.T @ ...`. There is one element for each different term. <br />**Example:** If the problem depends on `X.T @ X-A` and `X.T @ b-c` then we have `Glob_Const_list[0]=A` and `Glob_Const_list[1]=c`. |

If one or more of these do not appear in the problem, set their corresponding list to an empty one.

**Example:** `Gamma_list=[]` if no quadratic term `X.T @ Gamma @ X` appears in the problem.

The function `lcmv_solver` depends on the signal `y1(t)`, the linear term `B1` and the global constant `GC1`. There are no quadratic terms.

Looking at problem `P`, the relationship between the data in `P` and the solver are:

|`P <--> lcmv_solver`| `data` |
| --- | --- |
| `y(t)==y1(t)` | `Y=[y[0],...,y[nbsamples-1]]`<br />`data['Y_list'][0]=Y` |
| `B==B1` | `data['B_list'][0]=B` |
| `H==GC1`| `data['Glob_Const_list'][0]=H` |
|  | `data['Gamma_list']=[]` |

**Creating the data and parameters:** In the given example code, we take:
``
y(t) = A @ s(t) + n(t),
``
where the entries of `s(t)` independently follow `N(0,0.5)`, and the entries of `n(t)` independently follow `N(0,0.1)` for each time instant `t`, where `N` denotes the Gaussian distribution. We take `s` to be `10`-dimensional. Additionally the entries of `A` are drawn independently from `U([-0.5,0.5])`.

We take `B` to be the matrix equal to the first `Q` columns of `A`, while the entries of `H` independently follow `N(0,1)`.