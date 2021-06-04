# The trace ratio optimization problem
 
Folder implementing the trace ratio optimization (TRO) problem:
$$
\text{arg} \max_X \frac{\text{tr}(X^TR_{\mathbf{yy}}X)}{\text{tr}(X^TR_{\mathbf{vv}}X)}\\
\text{subject to } X^TX=I,
$$

where $R_{\mathbf{yy}}=\mathbb{E}[\mathbf{yy}^T]$ and $R_{\mathbf{vv}}=\mathbb{E}[\mathbf{vv}^T]$ in a distributed setting using the the DSFO framework (more details inside the code).

`tro_solver:` Centralized algorithm for solving the TRO problem.

`tro_eval:`  Evaluate the TRO objective function

`tro_resolve_uniqueness:`  Resolve the uniqueness ambiguity of the TRO problem, i.e., invariance of the problem to the sign of the columns of $X$.

`run_tro:` Script to run the DSFO algorithm to solve the TRO problem in a randomly generated network.