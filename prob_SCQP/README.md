# Spherically constrained quadratic problem
 
Folder implementing the following spherically constrained quadratic problem (SCQP):
``
arg max_X 0.5*trace(X'*R_yy*X)+trace(X'*B) subject to ||X||_F^2=1,
``

where ``R_yy=E[yy']``, in a distributed setting using the the DSFO framework (more details inside the code).

`scqp_solver:` Centralized algorithm for solving the SCQP.

`scqp_eval:`  Evaluate the SCQP objective function

`run_scqp:` Script to run the DSFO algorithm to solve the SCQP in a randomly generated network.

`SCQP_script:` Matlab live script example.