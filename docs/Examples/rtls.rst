RTLS
==========================

Example for the regularized total least squares (RTLS) problem given as:

.. math::

    \begin{aligned}
        \max_{X}\; & \frac{\mathbb{E}[\|X^T \mathbf{y}(t)-\mathbf{d}(t)\|^2]}{1+X^T\Gamma X} \\
        \text{s.t. } & \|X^T L\|^2 \leq \delta^2.
    \end{aligned}

.. automodule:: dasftoolbox.optimization_problems.rtls_problem
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: dasftoolbox.data_retrievers.rtls_data_retriever
    :members:
    :undoc-members:
    :show-inheritance: