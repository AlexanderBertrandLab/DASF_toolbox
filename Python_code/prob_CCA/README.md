# Canonical Correlation Analysis
 
Example implementing the Canonical Correlation Analysis (CCA) in a distributed setting using the the DASF framework:
``
P: max_(X,W) E[trace(X.T @ y(t) @ v(t).T @ W)]
    s.t. E[X.T @ y(t) @ y(t).T @ X] = I, E[W.T @ v(t) @ v(t).T @ W] = I,
``

with the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y(t)` | `nbsensors x 1` for every `t` | Signal |
| `v(t)` | `nbsensors x 1` for every `t` | Signal |

The functions and files in this folder are:

`cca_solver:` Centralized algorithm for solving the CCA problem: 
        
        max_(X1,X2) E[trace(X1.T @ y1(t) @ y2(t).T @ X2)]
    s.t. E[X1.T @ y1(t) @ y1(t).T @ X1] = I, E[X2.T @ y2(t) @ y2(t).T @ X2] = I.

taking as input the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y1(t)` | `nbsensors x 1` for every `t` | Signal |
| `y2(t)` | `nbsensors x 1` for every `t` | Signal |

`cca_eval:`  Evaluate the CCA objective function.

`cca_select_sol:`  Resolve the uniqueness ambiguity of the CCA problem, i.e., invariance of the problem to the sign of the columns of `X`.

`run_cca.py:` Script to run the DASF algorithm to solve the CCA problem in a randomly generated network.

`CCA_notebook.ipynb:` Jupyter notebook example.

**How to initialize** `data`**:** In this case, `data` will be a list containing `data_X` and `data_W` which are dictionaries with the same fields as the `data` structure for problems using the `dasf` function. We have one dictionary per variable. We remind the fields of the dictionaries `data_X` and `data_W`:
| Field | Description |
 | --- | --- |
 | **Signals:** `Y_list` | List for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one element for each different signal. <br /> **Example:** If the problem depends on `X.T @ y(t)` and `X.T @ v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_list[0]=Y` and `Y_list[1]=V`. |
| **Linear terms:** `B_list` | List for deterministic linear terms, where each term has `nbsensors` rows. There is one element for each different parameter. <br />**Example:** If the problem depends on `X.T @ B` and `X.T @ c` then we have `B_list[0]=B` and `B_list[1]=c`. |
| **Quadratic terms:** `Gamma_list` | List for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one element for each different term. <br />**Example:** If the problem depends on `X.T @ X`, `X.T @ Gamma_1 @ X` and `X.T @ Gamma_2 @ X` then we have `Gamma_list[0]=identity(nbsensors)`, `Gamma_list[1]=Gamma_1` and `Gamma_list[2]=Gamma_2`. |
| **Global constants:** `Glob_Const_list` | List for global constants, i.e., terms that do not appear in the form `X.T @ ...`. There is one element for each different term. <br />**Example:** If the problem depends on `X.T @ X-A` and `X.T @ b-c` then we have `Glob_Const_list[0]=A` and `Glob_Const_list[1]=c`. |

If one or more of these do not appear in the problem, set their corresponding list to an empty one.

**Example:** `Gamma_list=[]` if no quadratic term `X.T @ Gamma @ X` appears in the problem.

The function `cca_solver` depends on the signals `y1(t)` and `y2(t)`. There are no linear or quadratic terms, nor global constants.

Looking at problem `P`, the relationship between the data in `P` and the solver are:

| `P <--> cca_solver` | `data` |
|---------------------| --- |
| `y(t)==y1(t)`       | `Y=[y[0],...,y[nbsamples-1]]`<br />`data_X['Y_list'][0]=Y` |
| `v(t)==y2(t)`       | `V=[v[0],...,v[nbsamples-1]]`<br />`data_W['Y_list'][0]=V` |
|                     | `data['B_list']=[]` |
|                     | `data['Gamma_list']=[]` |
|                     | `data['Glob_Const_list']=[]` |

**Creating the data and parameters:** In the given example code, we take:
``
v(t) = A @ d(t) + n(t),
``
where the entries of `d(t)` independently follow `N(0,0.5)`, and the entries of `n(t)` independently follow `N(0,0.1)` for each time instant `t`, where `N` denotes the Gaussian distribution. We take `d` to be `10`-dimensional. Additionally the entries of `A` are drawn independently from `U([-0.5,0.5])`. We take `y(t)=v(t-3)`.