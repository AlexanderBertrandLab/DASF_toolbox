CCA
==========================

Example for the canonical correlation analysis (CCA) problem given as:

.. math::

    \begin{aligned}
        \max_{X,W}\; & \mathbb{E}[\text{trace}(X^T \mathbf{y}(t) \mathbf{v}^T(t) W)] \\
        \text{s.t. } & \mathbb{E}[X^T \mathbf{y}(t) \mathbf{y}^T(t) X] = I,\\
        & \mathbb{E}[W^T \mathbf{v}(t) \mathbf{v}^T(t) W] = I.
    \end{aligned}

.. automodule:: dasftoolbox.optimization_problems.cca_problem
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: dasftoolbox.data_retrievers.cca_data_retriever
    :members:
    :undoc-members:
    :show-inheritance: