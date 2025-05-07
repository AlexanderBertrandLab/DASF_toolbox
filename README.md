# DASF toolbox

The distributed adaptive signal fusion (DASF) algorithm framework solves feature-partitioned/spatial filtering problems in a distributed fashion without centralizing the raw data. The implementation is based on the published work in [1] and [2]. To cite this toolbox, please use:

      @misc{dasftoolbox,
            title={DASF toolbox},
            author={Musluoglu, Cem Ates and Bertrand, Alexander},
            howpublished = "\url{https://github.com/AlexanderBertrandLab/DASF_toolbox}",
            year={2022}
      }

**News:**

- The DASF toolbox has been featured in the [AI toolbox list](https://www.flandersairesearch.be/en/research/list-of-toolboxes/dasf) of the Flanders AI Research Program! 
- A brand new version is here! The DASF toolbox gets a new look with a more practical and flexible design! With this version, Python is the main supported language, while the previous version can be accessed from the archive (the deprecated folder), where the MATLAB implementation can also be found.

**Installation:**

You can clone this repository and then locally build the package using:

`pip install -e .`

A documentation is available in the link below (still under construction):

[![Documentation Status](https://readthedocs.org/projects/dasf-toolbox/badge/?version=latest)](https://dasf-toolbox.readthedocs.io/en/latest/)

![](https://github.com/CemMusluoglu/DASF_toolbox/blob/main/assets/dasf_gif.gif)

**Description:**

Formally, the DASF framework is built to solve optimization problems of the following form:

        P: min_X f_hat ( X'*y(t), X'*B) = f(X)
           s.t.  h_j ( X'*y(t), X'*B) <= 0 for inequalities j,
                 h_j ( X'*y(t), X'*B) = 0 for equalities j.

Many spatial filtering applications require the optimal filter to be selected based on a criterion following the form above, such as denoising, dimensionality reduction, beamforming, etc. of multi-channel signals. The DASF algorithm solves such problem in a distributed way for settings such as wireless sensor networks. It does so in a sequential and iterative fashion, where a smaller/compressed/local version of the original problem is formed at an updating node `q` and iteration `i` and solved to then partially find the optimal variable. Iterating this procedure at different updating nodes leads to convergence to the optimal filter in most practical scenarios. One main advantage of the DASF algorithm is that these local problems are always **parameterized** versions of problem `P`. Therefore, a solver for problem `P` is used within the DASF algorithm.

**Note:** In the problem above, There can be more than one `y(t)` and `B` which are not represented for conciseness.



**References:**

[1] C. A. Musluoglu and A. Bertrand, "A Unified Algorithmic Framework for Distributed Adaptive Signal and Feature Fusion Problems—Part I: Algorithm Derivation," in IEEE Transactions on Signal Processing, vol. 71, pp. 1863-1878, 2023, doi: https://doi.org/10.1109/TSP.2023.3275272.

[2] C. A. Musluoglu, C. Hovine and A. Bertrand, "A Unified Algorithmic Framework for Distributed Adaptive Signal and Feature Fusion Problems — Part II: Convergence Properties," in IEEE Transactions on Signal Processing, vol. 71, pp. 1879-1894, 2023, doi: https://doi.org/10.1109/TSP.2023.3275273.