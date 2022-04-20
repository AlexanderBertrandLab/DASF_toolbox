# Distributed Adaptive Signal Fusion Algorithm
 The distributed adaptive signal fusion (DASF) algorithm framework implementation.

 Given an optimization problem fitting the DASF framework:

        P: min_X f_hat ( X'*y(t), X'*B, X'*Gamma*X ) = f(X)
           s.t.  h_j ( X'*y(t), X'*B, X'*Gamma*X ) <= 0 for inequalities j,
                 h_j ( X'*y(t), X'*B, X'*Gamma*X ) = 0 for equalities j,

the DASF algorithm solves the problem in a distributed setting such as a wireless sensor network consisting of nodes connected to each other in a certain way. This is done by creating a local problem at node `q` and iteration `i` and has the advantage that the local problem is a **parameterized** version of problem `P`. Therefore, a solver for problem `P` is used for the distributed implementation.

**Note:** There can be more than one `y(t)`, `B` and `Gamma` which are not represented for conciseness. 

The `dasf` function implements the DASF algorithm and is called in the following way:

**Matlab:**

        [X_est,norm_diff,norm_err,f_seq]=dasf(prob_params,data,...
        @prob_solver,conv,@prob_select_sol,@prob_eval)

**Python:**

        X_est,norm_diff,norm_err,f_seq=dasf(prob_params,data,...
        prob_solver,conv,prob_select_sol,prob_eval)

