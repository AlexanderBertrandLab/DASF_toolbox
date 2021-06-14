# The trace ratio optimization problem
 
Folder implementing the trace ratio optimization (TRO) problem:
``
arg max_X trace(X'*R_yy*X)/trace(X'*R_vv*X) subject to X'*X=I,
``

where ``R_yy=E[yy']`` and ``R_vv=E[vv']``, in a distributed setting using the the DSFO framework (more details inside the code).

`tro_solver:` Centralized algorithm for solving the TRO problem.

`tro_eval:`  Evaluate the TRO objective function

`tro_resolve_uniqueness:`  Resolve the uniqueness ambiguity of the TRO problem, i.e., invariance of the problem to the sign of the columns of $X$.

`run_tro:` Script to run the DSFO algorithm to solve the TRO problem in a randomly generated network.

`TRO_script:` Matlab live script example.