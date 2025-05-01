ICA
==========================

Example for the independent component analysis (ICA) problem given as:

.. math::

    \begin{aligned}
        \max_X\; &\sum_m \mathbb{E}[F(X_m^T \mathbf{y}(t))] \\
        \text{s.t. } & \mathbb{E}[X^T \mathbf{y}(t)\mathbf{y}^T(t) X] = I,
    \end{aligned}

where :math:`X_m` is the :math:`m`-th column of :math:`X` and :math:`F` is the negentropy function.


.. automodule:: dasftoolbox.optimization_problems.ica_problem
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: dasftoolbox.data_retrievers.ica_data_retriever
    :members:
    :undoc-members:
    :show-inheritance: