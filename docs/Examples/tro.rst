TRO
==========================

Example for the trace ratio optimization (TRO) problem given as:

.. math::

    \begin{aligned}
        \max_{X}\; & \frac{\mathbb{E}[\|X^T \mathbf{y}(t)\|^2]}{\mathbb{E}[\|X^T \mathbf{v}(t)\|^2]} \\
        \text{s.t. } & X^T \Gamma X = I.
    \end{aligned}

.. automodule:: dasftoolbox.optimization_problems.tro_problem
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: dasftoolbox.data_retrievers.tro_data_retriever
    :members:
    :undoc-members:
    :show-inheritance: