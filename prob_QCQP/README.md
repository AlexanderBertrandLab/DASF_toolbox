# Quadratically constrained quadratic problem
 
Folder implementing the following quadratically constrained quadratic problem (QCQP):
``
arg max_X 0.5*trace(X'*R_yy*X)-trace(X'*B) subject to ||X||_F^2<=alpha^2, X'*c=d,
``

where ``R_yy=E[yy']``, in a distributed setting using the the DSFO framework (more details inside the code).

`qcqp_solver:` Centralized algorithm for solving the QCQP.

`qcqp_eval:`  Evaluate the QCQP objective function

`run_qcqp:` Script to run the DSFO algorithm to solve the QCQP in a randomly generated network.

`QCQP_script:` Matlab live script example.