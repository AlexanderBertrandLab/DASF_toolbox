LCMV
==========================

Example for the linearly constrained minimum variance (LCMV) problem given as:

.. math::

    \begin{aligned}
        \min_{X}\; & \mathbb{E}[\| X^T \mathbf{y}(t)\|^2] \\
        \text{s.t. } & X^TB=H.
    \end{aligned}

.. automodule:: dasftoolbox.optimization_problems.lcmv_problem
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: dasftoolbox.data_retrievers.lcmv_data_retriever
    :members:
    :undoc-members:
    :show-inheritance: