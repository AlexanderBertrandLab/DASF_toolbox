# Distributed Signal Fusion Optimization
 The distributed signal fusion optimization (DSFO) algorithm framework implementation.

 Given an optimization problem fitting the DSFO framework:

        P: min_X f_hat ( X'*y(t), X'*B, X'*Gamma*X ) = f(X)
           s.t.  h_j ( X'*y(t), X'*B, X'*Gamma*X ) <= 0 for inequalities j,
                 h_j ( X'*y(t), X'*B, X'*Gamma*X ) = 0 for equalities j,

the DSFO algorithm solves the problem in a distributed setting such as a wireless sensor network consisting of nodes connected to each other in a certain way. This is done by creating a local problem at node `q` and iteration `i` and has the advantage that the local problem is a **parameterized** version of problem `P`. Therefore, a solver for problem `P` is used for the distributed implementation.

**Note:** There can be more than one `y(t)`, `B` and `Gamma` which are not represented for conciseness. 

The `dsfo` function implements the DSFO algorithm and is called in the following way:
**Matlab:**

        [X_est,norm_diff,norm_err,f_seq]=dsfo(prob_params,data,...
        @prob_solver,conv,@prob_select_sol,@prob_eval)

**Python:**

        X_est,norm_diff,norm_err,f_seq=dsfo(prob_params,data,...
        prob_solver,conv,prob_select_sol,prob_eval)

