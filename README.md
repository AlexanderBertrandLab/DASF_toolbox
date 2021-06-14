# Distributed Signal Fusion Optimization
 The distributed signal fusion optimization (DSFO) algorithm framework implementation.

 Given an optimization problem fitting the DSFO framework the `dsfo` algorithm solves the problem using the DSFO algorithm taking as arguments:

 `data:` This argument is a structure related to the data of the problem, containing the following fields:

        - Y_cell
Cell for stochastic signals, where each signal is a `nbsensors x nbsamples` matrix corresponding to time samples of multi-channel signals in the network. There is one cell for each different signal.
For example: The problem depends on `X'*y(t)` and `X'*v(t)` then `Y` and `V` contain the time samples of `y` and `v` respectively and we have `Y_cell{1}=Y` and `Y_cell{2}=V`.

        - B_cell
Cell for deterministic constant parameters, where each parameter has `nbsensors` rows. There is one cell for each different parameter.
For example: The problem depends on `X'*B` and `X'*c` then we have `B_cell{1}=B` and `B_cell{2}=c`.

        - Gamma_cell
Cell for deterministic quadratic block-diagonal terms, where each term is a `nbsensors x nbsensors` matrix. There is one cell for each different term.
For example: The problem depends on `X'*X`, `X'*Gamma_1*X` and `X'*Gamma_2*X` then we have `Gamma_cell{1}=eye(nbsensors)`, `Gamma_cell{2}=Gamma_1` and `Gamma_cell{3}=Gamma_2`.

        - Glob_Const_cell
Cell for global constants, i.e., terms that do not appear in the form `X'*...`. There is one cell for each different term.
For example: The problem depends on `X'*X-A` and `X'*b-c` then we have `Glob_Const_cell{1}=A` and `Glob_Const_cell{2}=c`.


`prob_params:` This argument is a structure related to the parameters of the problem, containing the following fields:

        - obj_eval: Function handle to the objective function evaluation. 

        - prob_resolve_uniqueness: (Optional) Function handle to the method for resolving uniqueness ambiguities.

        - X_star: (Optional) Matrix which solves the problem, for comparison purposes.

