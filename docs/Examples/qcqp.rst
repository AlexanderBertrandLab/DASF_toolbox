QCQP
==========================

Example for the quadratically constrained quadratic problem (QCQP) given as:

.. math::

    \begin{aligned}
        \min_{X}\; & \frac{1}{2}\mathbb{E}[\| X^T \mathbf{y}(t)\|^2] - \text{trace}(X^T B) \\
        \text{s.t. } & \text{trace}(X^T \Gamma  X) \leq \alpha^2,\\
        & X^T \mathbf{c} = \mathbf{d}.
    \end{aligned}

.. automodule:: dasftoolbox.optimization_problems.qcqp_problem
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: dasftoolbox.data_retrievers.qcqp_data_retriever
    :members:
    :undoc-members:
    :show-inheritance: