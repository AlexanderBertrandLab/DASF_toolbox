GEVD
==========================

Example for the generalized eigenvalue decomposition (GEVD) problem given as:

.. math::

    \begin{aligned}
        \max_X\; &\mathbb{E}[\| X^T \mathbf{y}(t)\|^2] \\
        \text{s.t. } & \mathbb{E}[X^T \mathbf{v}(t)\mathbf{v}^T(t) X] = I.
    \end{aligned}

.. automodule:: dasftoolbox.optimization_problems.gevd_problem
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: dasftoolbox.data_retrievers.gevd_data_retriever
    :members:
    :undoc-members:
    :show-inheritance: