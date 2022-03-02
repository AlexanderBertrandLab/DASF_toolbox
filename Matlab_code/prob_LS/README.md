# Least Squares
 
Folder implementing the following Least Squares (LS) Problem in a distributed setting using the the DSFO framework:
``
P: min_X E[ || d(t)-X'*y(t) ||^2 ],
``

with the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y(t)` | `nbsensors x 1` for every `t` | Signal |
| `d(t)` | `Q x 1` for every `t` | Global constant |


The functions and files in this folder are:

`ls_solver.m:` Centralized algorithm for solving the LS:

        min_X E[ || gc1(t)-X'*y1(t) ||^2 ].

taking as input the following data:

|Data|Size|Type|
| --- | --- | --- |
| `y1(t)` | `nbsensors x 1` for every `t` | Signal |
| `gc1(t)` | `Q x 1` for every `t` | Global constant |


`ls_eval.m:`  Evaluate the LS objective function.

`run_ls.m:` Script to run the DSFO algorithm to solve the LS in a randomly generated network.

`LS_script.mlx:` Matlab live script example.

**How to initialize** `data`**:** We remind the fields of the structure `data`:
| Field | Description |
 | --- | --- |
 | **Signals:** `Y_cell` | Cell for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one cell for each different signal. <br /> **Example:** If the problem depends on `X'*y(t)` and `X'*v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_cell{1}=Y` and `Y_cell{2}=V`. |
| **Linear terms:** `B_cell` | Cell for deterministic linear terms, where each term has `nbsensors` rows. There is one cell for each different parameter. <br />**Example:** If the problem depends on `X'*B` and `X'*c` then we have `B_cell{1}=B` and `B_cell{2}=c`. |
| **Quadratic terms:** `Gamma_cell` | Cell for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X`, `X'*Gamma_1*X` and `X'*Gamma_2*X` then we have `Gamma_cell{1}=eye(nbsensors)`, `Gamma_cell{2}=Gamma_1` and `Gamma_cell{3}=Gamma_2`. |
| **Global constants:** `Glob_Const_cell` | Cell for global constants, i.e., terms that do not appear in the form `X'*...`. There is one cell for each different term. <br />**Example:** If the problem depends on `X'*X-A` and `X'*b-c` then we have `Glob_Const_cell{1}=A` and `Glob_Const_cell{2}=c`. |

If one or more of these do not appear in the problem, set their corresponding cell to an empty one.

**Example:** `Gamma_cell={}` if no quadratic term `X'*Gamma*X` appears in the problem.

The function `ls_solver` depends on the signal `y1(t)` and the global constant `gc1(t)`. There are no linear or quadratic terms.

Looking at problem `P`, the relationship between the data in `P` and the solver are:

|`P <--> ls_solver`| `data` |
| --- | --- |
| `y(t)==y1(t)` | `Y=[y(1),...,y(nbsamples)]`<br />`data.Y_cell{1}=Y` |
| `d(t)==gc1(t)`| `D=[d(1),...,d(nbsamples)]`<br />`data.Glob_Const_cell{1}=D`|
| | `data.B_cell={}` |
| | `data.Gamma_cell={}` |

**Creating the data and parameters:** In the given example code, we take:
``
y(t)=A*d(t)+n(t),
``
where the entries of `d(t)` independently follow `N(0,0.5)`, and the entries of `n(t)` independently follow `N(0,0.1)` for each time instant `t`, where `N` denotes the Gaussian distribution. We take `d` to be `Q`-dimensional. Additionally the entries of `A` are drawn independently from `U([-0.5,0.5])`.