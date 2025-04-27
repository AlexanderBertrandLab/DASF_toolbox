# Distributed Adaptive Signal Fusion Algorithm

**News:**

- The DASF toolbox has been featured in the [AI toolbox list](https://www.flandersairesearch.be/en/research/list-of-toolboxes/dasf) of the Flanders AI Research Program! 
- v2 is here! The DASF toolbox gets a brand new look with a much more practical and flexible design! With this version, Python is the main supported language, while the previous version can be accessed from the archive, where the MATLAB implementation can also be found.

[![Documentation Status](https://dasf-toolbox.readthedocs.io/en/latest/)](https://dasf-toolbox.readthedocs.io/en/latest/)

![](https://github.com/CemMusluoglu/DASF_toolbox/blob/main/assets/dasf_gif.gif)

 The distributed adaptive signal fusion (DASF) algorithm framework implementation based on [1] and [2].

 Given an optimization problem fitting the DASF framework:

        P: min_X f_hat ( X'*y(t), X'*B, X'*Gamma*X ) = f(X)
           s.t.  h_j ( X'*y(t), X'*B, X'*Gamma*X ) <= 0 for inequalities j,
                 h_j ( X'*y(t), X'*B, X'*Gamma*X ) = 0 for equalities j,

the DASF algorithm solves the problem in a distributed setting such as a wireless sensor network consisting of nodes connected to each other in a certain way. This is done by creating a local problem at node `q` and iteration `i` and has the advantage that the local problem is a **parameterized** version of problem `P`. Therefore, a solver for problem `P` is used for the distributed implementation. **Note:** There can be more than one `y(t)`, `B` and `Gamma` which are not represented for conciseness.



**References:**

[1] C. A. Musluoglu and A. Bertrand "A Unified Algorithmic Framework for Distributed Adaptive Signal and Feature Fusion Problems - Part I: Algorithm Derivation", IEEE Transactions on Signal Processing, 2023, doi: https://doi.org/10.1109/TSP.2023.3275272.

[2] C. A. Musluoglu, C. Hovine and A. Bertrand "A Unified Algorithmic Framework for Distributed Adaptive Signal and Feature Fusion Problems - Part II: Convergence Properties", IEEE Transactions on Signal Processing, 2023, doi: https://doi.org/10.1109/TSP.2023.3275273.