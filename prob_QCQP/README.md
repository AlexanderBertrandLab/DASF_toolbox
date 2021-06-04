# Quadratically constrained quadratic problem
 
Folder implementing the following QCQP:
$$
\text{arg} \max_X \text{tr}(X^TR_{\mathbf{yy}}X)-\text{tr}(B^TX)\\
\text{subject to } ||X||_F^2\leq \alpha^2,\\
X^T\mathbf{c}=\mathbf{d},
$$

where $R_{\mathbf{yy}}=\mathbb{E}[\mathbf{yy}^T]$ in a distributed setting using the the DSFO framework (more details inside the code).

`qcqp_solver:` Centralized algorithm for solving the QCQP.

`qcqp_eval:`  Evaluate the QCQP objective function

`run_qcqp:` Script to run the DSFO algorithm to solve the QCQP in a randomly generated network.
