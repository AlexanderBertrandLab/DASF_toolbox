import numpy as np
from problem_settings import (
    ProblemInputs,
    ProblemParameters,
    ConvergenceParameters,
)
from abc import ABC, abstractmethod


class OptimizationProblem:
    def __init__(
        self,
        problem_parameters: ProblemParameters,
        initial_estimate: np.ndarray | None = None,
    ) -> None:
        self.problem_parameters = problem_parameters
        self.initial_estimate = initial_estimate

    @abstractmethod
    def solve(
        self, problem_inputs: ProblemInputs | list[ProblemInputs]
    ) -> np.ndarray | list[np.ndarray]:
        pass

    @abstractmethod
    def evaluate_objective(
        X: np.ndarray | list[np.ndarray],
        problem_inputs: ProblemInputs | list[ProblemInputs],
    ) -> float:
        pass

    @abstractmethod
    def generate_synthetic_inputs(self) -> ProblemInputs | list[ProblemInputs]:
        pass

    def resolve_ambiguity(self, X_ref, X, prob_params, q):
        return X


class MMSEProblem(OptimizationProblem):
    def __init__(self) -> None:
        super().__init__()

    def solve(problem_inputs: ProblemInputs) -> np.ndarray:
        """Solve the MMSE problem min E[||d(t) - X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_data[0]
        D = problem_inputs.global_constants[0]

        N = np.size(Y, 1)

        Ryy = Y @ Y.T / N
        Ryy = (Ryy + Ryy.T) / 2
        Ryd = Y @ D.T / N

        X_star = np.linalg.inv(Ryy) @ Ryd

        return X_star

    def evaluate_objective(X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the MMSE objective E[||d(t) - X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_data[0]
        D = problem_inputs.global_constants[0]
        N = np.size(Y, 1)

        Ryy = Y @ Y.T / N
        Ryy = (Ryy + Ryy.T) / 2
        Rdd = D @ D.T / N
        Rdd = (Rdd + Rdd.T) / 2
        Ryd = Y @ D.T / N

        f = np.trace(X.T @ Ryy @ X) - 2 * np.trace(X.T @ Ryd) + np.trace(Rdd)

        return f

    def generate_synthetic_inputs(
        self,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        offset: float = 0.5,
        nb_sources: int | None = None,
    ) -> ProblemInputs:
        """Generate synthetic inputs for the MMSE problem."""
        rng = np.random.default_rng()

        Q = self.problem_parameters.nb_filters
        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = Q

        D = rng.normal(loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples))
        A = rng.uniform(low=-offset, high=offset, size=(nb_sensors, nb_sources))
        noise = rng.normal(
            loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples)
        )

        Y = A @ D + noise

        mmse_inputs = ProblemInputs(fused_data=[Y], global_constants=[D])

        return mmse_inputs


class LCMVProblem(OptimizationProblem):
    def __init__(self) -> None:
        super().__init__()

    def solve(problem_inputs: ProblemInputs) -> np.ndarray:
        """Solve the LCMV problem min E[||X.T @ y(t)||**2] s.t. X.T @ B = H."""
        Y = problem_inputs.fused_data[0]
        B = problem_inputs.fused_constants[0]
        H = problem_inputs.global_constants[0]

        N = np.size(Y, 1)

        Ryy = Y @ Y.T / N
        Ryy = (Ryy + Ryy.T) / 2

        X_star = (
            np.linalg.inv(Ryy) @ B @ np.linalg.inv(B.T @ np.linalg.inv(Ryy) @ B) @ H.T
        )

        return X_star

    def evaluate_objective(X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the LCMV objective E[||X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_data[0]
        N = np.size(Y, 1)

        Ryy = Y @ Y.T / N
        Ryy = (Ryy + Ryy.T) / 2

        f = np.trace(X.T @ Ryy @ X)

        return f

    def generate_synthetic_inputs(
        self,
        L: int,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        offset: float = 0.5,
        nb_sources: int | None = None,
    ) -> ProblemInputs:
        """Generate synthetic inputs for the LCMV problem."""
        rng = np.random.default_rng()

        Q = self.problem_parameters.nb_filters
        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = Q

        D = rng.normal(loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples))
        A = rng.uniform(low=-offset, high=offset, size=(nb_sensors, nb_sources))
        noise = rng.normal(
            loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples)
        )

        Y = A @ D + noise
        B = A[:, 0:L]
        H = rng.standard_normal(size=(Q, L))

        lcmv_inputs = ProblemInputs(
            fused_data=[Y], fused_constants=[B], global_constants=[H]
        )

        return lcmv_inputs


class GEVDProblem(OptimizationProblem):
    def __init__(self) -> None:
        super().__init__()

    def solve(self, problem_inputs: ProblemInputs) -> np.ndarray:
        """Solve the GEVD problem max E[||X.T @ y(t)||**2] s.t. E[X.T @ v(t) @ v(t).T @ X] = I."""
        Y = problem_inputs.fused_data[0]
        V = problem_inputs.fused_data[1]

        Q = self.problem_parameters.nb_filters
        N = np.size(Y, 1)

        Ryy = Y @ Y.T / N
        Ryy = (Ryy + Ryy.T) / 2
        Rvv = V @ V.T / N
        Rvv = (Rvv + Rvv.T) / 2

        eigvals, eigvecs = np.linalg.eigh(Ryy, Rvv)
        indices = np.argsort(eigvals)[::-1]

        X_star = eigvecs[:, indices[0:Q]]

        return X_star

    def evaluate_objective(X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the GEVD objective E[||X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_data[0]
        N = np.size(Y, 1)

        Ryy = Y @ Y.T / N
        Ryy = (Ryy + Ryy.T) / 2

        f = np.trace(X.T @ Ryy @ X)

        return f

    def resolve_ambiguity(self, X_ref, X, prob_params, q):
        """Resolve the sign ambiguity for the GEVD problem."""
        Q = self.problem_parameters.nb_filters

        for l in range(Q):
            if np.linalg.norm(X_ref[:, l] - X[:, l]) > np.linalg.norm(
                -X_ref[:, l] - X[:, l]
            ):
                X[:, l] = -X[:, l]

        return X

    def generate_synthetic_inputs(
        self,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        offset: float = 0.5,
        latent_dim: int = 10,
        nb_sources: int | None = None,
    ) -> ProblemInputs:
        """Generate data for the GEVD problem."""
        rng = np.random.default_rng()

        Q = self.problem_parameters.nb_filters
        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = Q

        D = rng.normal(loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples))
        S = rng.normal(
            loc=0, scale=np.sqrt(signal_var), size=(latent_dim - nb_sources, nb_samples)
        )
        A = rng.uniform(low=-offset, high=offset, size=(nb_sensors, nb_sources))
        B = rng.uniform(
            low=offset, high=offset, size=(nb_sensors, latent_dim - nb_sources)
        )
        noise = rng.normal(
            loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples)
        )
        V = B @ S + noise
        Y = A @ D + V

        gevd_inputs = ProblemInputs(fused_data=[Y, V])

        return gevd_inputs


class CCAProblem(OptimizationProblem):
    def __init__(self) -> None:
        super().__init__()

    def solve(self, problem_inputs: list[ProblemInputs]) -> np.ndarray:
        """Solve the CCA problem max_(X,W) E[trace(X.T @ y(t) @ v(t).T @ W)]
        s.t. E[X.T @ y(t) @ y(t).T @ X] = I, E[W.T @ v(t) @ v(t).T @ W] = I."""
        inputs_X = problem_inputs[0]
        inputs_W = problem_inputs[1]

        Y = inputs_X.fused_data[0]
        V = inputs_W.fused_data[0]

        Q = self.problem_parameters.nb_filters
        N = np.size(Y, 1)

        Ryy = Y @ Y.T / N
        Ryy = (Ryy + Ryy.T) / 2
        Rvv = V @ V.T / N
        Rvv = (Rvv + Rvv.T) / 2
        Ryv = Y @ V.T / N
        Rvy = Ryv.T

        inv_Rvv = np.linalg.inv(Rvv)
        inv_Rvv = (inv_Rvv + inv_Rvv.T) / 2
        A_X = Ryv @ inv_Rvv @ Rvy
        A_X = (A_X + A_X.T) / 2

        eigvals_X, eigvecs_X = np.linalg.eigh(A_X, Ryy)
        indices_X = np.argsort(eigvals_X)[::-1]
        eigvals_X = eigvals_X[indices_X]
        eigvecs_X = eigvecs_X[:, indices_X]

        X = eigvecs_X[:, 0:Q]
        eigvecs_W = (
            inv_Rvv @ Rvy @ eigvecs_X @ np.diag(1 / np.sqrt(np.absolute(eigvals_X)))
        )
        W = eigvecs_W[:, 0:Q]
        X_star = [X, W]

        return X_star

    def evaluate_objective(
        X: list[np.ndarray], problem_inputs: list[ProblemInputs]
    ) -> float:
        """Evaluate the CCA objective E[trace(X.T @ y(t) @ v(t).T @ W)]."""
        inputs_X = problem_inputs[0]
        inputs_W = problem_inputs[1]

        Y = inputs_X.fused_data[0]
        V = inputs_W.fused_data[0]
        N = np.size(Y, 1)

        Ryv = Y @ V.T / N

        f = np.trace(X[0].T @ Ryv @ X[1])

        return f

    def resolve_ambiguity(self, X_ref, X, prob_params, q):
        """Resolve the sign ambiguity for the CCA problem."""
        X = X[0]
        W = X[1]
        X_ref = X_ref[0]
        Q = self.problem_parameters.nb_filters

        for l in range(Q):
            if np.linalg.norm(X_ref[:, l] - X[:, l]) > np.linalg.norm(
                -X_ref[:, l] - X[:, l]
            ):
                X[:, l] = -X[:, l]
                W[:, l] = -W[:, l]

        return [X, W]

    def generate_synthetic_inputs(
        self,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        offset: float = 0.5,
        lags: int = 1,
        nb_sources: int | None = None,
    ) -> list[ProblemInputs]:
        """Generate synthetic inputs for the CCA problem."""
        rng = np.random.default_rng()

        Q = self.problem_parameters.nb_filters
        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = Q

        D = rng.normal(
            loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples + lags)
        )
        A = rng.uniform(low=-offset, high=offset, size=(nb_sensors, nb_sources))
        noise = rng.normal(
            loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples + lags)
        )

        signal = A @ D + noise
        Y = signal[:, 0:nb_samples]
        V = signal[:, lags:None]

        cca_inputs_X = ProblemInputs(fused_data=[Y])
        cca_inputs_W = ProblemInputs(fused_data=[V])

        return [cca_inputs_X, cca_inputs_W]
