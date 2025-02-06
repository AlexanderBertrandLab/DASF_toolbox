import numpy as np
from problem_settings import (
    ProblemInputs,
    ProblemParameters,
    ConvergenceParameters,
)
from abc import ABC, abstractmethod
from utils import make_symmetric, autocorrelation_matrix, cross_correlation_matrix
import scipy.optimize as opt
import warnings
import pymanopt
from pymanopt.manifolds import Sphere
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions
import autograd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationProblem:
    def __init__(
        self,
        nb_filters: int,
        convergence_parameters: ConvergenceParameters | None = None,
        problem_parameters: ProblemParameters | None = None,
        initial_estimate: np.ndarray | None = None,
    ) -> None:
        self.nb_filters = nb_filters
        self.convergence_parameters = convergence_parameters
        self.problem_parameters = problem_parameters
        self.initial_estimate = initial_estimate
        self._X_star = None

    @abstractmethod
    def solve(
        self, problem_inputs: ProblemInputs | list[ProblemInputs]
    ) -> np.ndarray | list[np.ndarray]:
        pass

    @abstractmethod
    def evaluate_objective(
        self,
        X: np.ndarray | list[np.ndarray],
        problem_inputs: ProblemInputs | list[ProblemInputs],
    ) -> float:
        pass

    @property
    def X_star(self):
        if self._X_star is None:
            logger.warning("The problem has not been solved yet.")
        return self._X_star

    # def resolve_ambiguity(self, X_ref, X, prob_params, q):
    #     return X


class MMSEProblem(OptimizationProblem):
    def __init__(self, nb_filters: int) -> None:
        super().__init__(nb_filters=nb_filters)

    def solve(self, problem_inputs: ProblemInputs) -> np.ndarray:
        """Solve the MMSE problem min E[||d(t) - X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_data[0]
        D = problem_inputs.global_constants[0]

        Ryy = autocorrelation_matrix(Y)
        Ryd = cross_correlation_matrix(Y, D)

        self._X_star = np.linalg.inv(Ryy) @ Ryd

        return self._X_star

    def evaluate_objective(self, X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the MMSE objective E[||d(t) - X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_data[0]
        D = problem_inputs.global_constants[0]

        Ryy = autocorrelation_matrix(Y)
        Rdd = autocorrelation_matrix(D)
        Ryd = cross_correlation_matrix(Y, D)

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

        if self.problem_parameters is None:
            raise ValueError(
                "Problem parameters must be set to generate synthetic inputs."
            )

        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = self.nb_filters

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

        Ryy = autocorrelation_matrix(Y)

        X_star = (
            np.linalg.inv(Ryy) @ B @ np.linalg.inv(B.T @ np.linalg.inv(Ryy) @ B) @ H.T
        )

        return X_star

    def evaluate_objective(X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the LCMV objective E[||X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_data[0]

        Ryy = autocorrelation_matrix(Y)

        f = np.trace(X.T @ Ryy @ X)

        return f

    def generate_synthetic_inputs(
        self,
        nb_steering: int,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        offset: float = 0.5,
        nb_sources: int | None = None,
    ) -> ProblemInputs:
        """Generate synthetic inputs for the LCMV problem."""
        rng = np.random.default_rng()

        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = self.nb_filters

        D = rng.normal(loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples))
        A = rng.uniform(low=-offset, high=offset, size=(nb_sensors, nb_sources))
        noise = rng.normal(
            loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples)
        )

        Y = A @ D + noise
        B = A[:, 0:nb_steering]
        H = rng.standard_normal(size=(self.nb_filters, nb_steering))

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

        Ryy = autocorrelation_matrix(Y)
        Rvv = autocorrelation_matrix(V)

        eigvals, eigvecs = np.linalg.eigh(Ryy, Rvv)
        indices = np.argsort(eigvals)[::-1]

        X_star = eigvecs[:, indices[0 : self.nb_filters]]

        return X_star

    def evaluate_objective(X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the GEVD objective E[||X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_data[0]

        Ryy = autocorrelation_matrix(Y)

        f = np.trace(X.T @ Ryy @ X)

        return f

    def resolve_ambiguity(self, X_ref, X, prob_params, q):
        """Resolve the sign ambiguity for the GEVD problem."""

        for l in range(self.nb_filters):
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

        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = self.nb_filters

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

        Ryy = autocorrelation_matrix(Y)
        Rvv = autocorrelation_matrix(V)
        Ryv = cross_correlation_matrix(Y, V)
        Rvy = Ryv.T

        inv_Rvv = np.linalg.inv(Rvv)
        inv_Rvv = (inv_Rvv + inv_Rvv.T) / 2
        A_X = Ryv @ inv_Rvv @ Rvy
        A_X = (A_X + A_X.T) / 2

        eigvals_X, eigvecs_X = np.linalg.eigh(A_X, Ryy)
        indices_X = np.argsort(eigvals_X)[::-1]
        eigvals_X = eigvals_X[indices_X]
        eigvecs_X = eigvecs_X[:, indices_X]

        X = eigvecs_X[:, 0 : self.nb_filters]
        eigvecs_W = (
            inv_Rvv @ Rvy @ eigvecs_X @ np.diag(1 / np.sqrt(np.absolute(eigvals_X)))
        )
        W = eigvecs_W[:, 0 : self.nb_filters]
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

        Ryv = cross_correlation_matrix(Y, V)

        f = np.trace(X[0].T @ Ryv @ X[1])

        return f

    def resolve_ambiguity(self, X_ref, X, prob_params, q):
        """Resolve the sign ambiguity for the CCA problem."""
        X = X[0]
        W = X[1]
        X_ref = X_ref[0]
        Q = np.size(X, 1)

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
        lags: int = 3,
        nb_sources: int | None = None,
    ) -> list[ProblemInputs]:
        """Generate synthetic inputs for the CCA problem."""
        rng = np.random.default_rng()
        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = self.nb_filters

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


class TROProblem(OptimizationProblem):
    def __init__(
        self,
        convergence_parameters: ConvergenceParameters | None,
        initial_estimate: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            convergence_parameters=convergence_parameters,
            initial_estimate=initial_estimate,
        )

    def solve(self, problem_inputs: ProblemInputs) -> np.ndarray:
        """Solve the TRO problem max E[||X.T @ y(t)||**2] / E[||X.T @ v(t)||**2] s.t. X.T @ Gamma @ X = I."""
        Y = problem_inputs.fused_data[0]
        V = problem_inputs.fused_data[1]
        Gamma = problem_inputs.fused_quadratics[0]

        rng = np.random.default_rng()
        if self.initial_estimate is None:
            X = rng.normal(
                size=(
                    np.size(Y, 0),
                    self.nb_filters,
                )
            )
        else:
            X = self.initial_estimate
        f = self.evaluate_objective(X, problem_inputs)

        convergence_parameters = (
            self.convergence_parameters
            if self.convergence_parameters
            else ConvergenceParameters()
        )

        Ryy = autocorrelation_matrix(Y)
        Rvv = autocorrelation_matrix(V)

        U_c, S_c, _ = np.linalg.svd(Gamma)

        Y_t = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Y
        V_t = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ V

        Kyy = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Ryy @ U_c @ np.diag(np.sqrt(1 / S_c))
        Kvv = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Rvv @ U_c @ np.diag(np.sqrt(1 / S_c))

        i = 0
        while i < convergence_parameters.max_iterations:
            eigvals, eigvecs = np.linalg.eig(Kyy - f * Kvv)
            indices = np.argsort(eigvals)[::-1]

            X_old = X
            X = eigvecs[:, indices[0 : self.nb_filters]]
            f_old = f
            Y_list_t = [Y_t, V_t]
            problem_inputs_t = ProblemInputs(fused_data=Y_list_t)
            f = self.evaluate_objective(X, problem_inputs_t)

            i += 1

            if (
                np.absolute(f - f_old) <= convergence_parameters.objective_tolerance
            ) or (
                np.linalg.norm(X - X_old, "fro")
                <= convergence_parameters.argument_tolerance
            ):
                break

        X_star = U_c @ np.diag(np.sqrt(1 / S_c)) @ X

        return X_star

    def evaluate_objective(self, X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the TRO objective E[||X.T @ y(t)||**2] / E[||X.T @ v(t)||**2]."""

        Y = problem_inputs.fused_data[0]
        V = problem_inputs.fused_data[1]

        Ryy = autocorrelation_matrix(Y)
        Rvv = autocorrelation_matrix(V)

        f = np.trace(X.T @ Ryy @ X) / np.trace(X.T @ Rvv @ X)

        return f

    def resolve_ambiguity(self, X_ref, X, prob_params, q):
        """Resolve the sign ambiguity for the TRO problem."""

        for l in range(self.nb_filters):
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
        latent_dimensions: int = 10,
        nb_sources: int | None = None,
    ) -> list[ProblemInputs]:
        """Generate synthetic inputs for the TRO problem."""
        rng = np.random.default_rng()

        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = self.nb_filters

        D = rng.normal(loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples))
        S = rng.normal(
            loc=0,
            scale=np.sqrt(signal_var),
            size=(latent_dimensions - nb_sources, nb_samples),
        )
        A = rng.uniform(low=-offset, high=offset, size=(nb_sensors, nb_sources))
        B = rng.uniform(
            low=offset, high=offset, size=(nb_sensors, latent_dimensions - nb_sources)
        )
        noise = rng.normal(
            loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples)
        )

        V = B @ S + noise
        Y = A @ D + V

        Gamma = np.eye(nb_sensors)

        tro_inputs = ProblemInputs(fused_data=[Y, V], fused_quadratics=[Gamma])

        return tro_inputs


class QCQPProblem(OptimizationProblem):
    def __init__(self) -> None:
        super().__init__()

    def solve(self, problem_inputs: ProblemInputs) -> np.ndarray:
        """Solve the QCQP problem min 0.5 * E[||X.T @ y(t)||**2] - trace(X.T @ B) s.t. trace(X.T @ Gamma @ X) <= alpha**2; X.T @ c = d."""

        def X_fun(mu: float, problem_inputs: ProblemInputs) -> np.ndarray:
            Y = problem_inputs.fused_data[0]
            B = problem_inputs.fused_constants[0]
            c = problem_inputs.fused_constants[1]
            Gamma = problem_inputs.fused_quadratics[0]
            d = problem_inputs.global_constants[1]

            Ryy = autocorrelation_matrix(Y)

            M = Ryy + mu * Gamma
            M_inv = np.linalg.inv(M)
            w = (B.T @ M_inv.T @ c - d) / (c.T @ M_inv @ c)
            X = M_inv @ (B - c @ w.T)

            return X

        def norm_fun(mu: float, problem_inputs: ProblemInputs) -> float:
            Gamma = problem_inputs.fused_quadratics[0]
            alpha = problem_inputs.global_constants[0]
            X = X_fun(mu, problem_inputs)
            norm = np.trace(X.T @ Gamma @ X) - alpha**2

            return norm

        c = problem_inputs.fused_constants[1]
        Gamma = problem_inputs.fused_quadratics[0]
        alpha = problem_inputs.global_constants[0]
        d = problem_inputs.global_constants[1]

        U_c, S_c, _ = np.linalg.svd(Gamma)

        sqrt_Gamma = (U_c @ np.diag(np.sqrt(S_c))).T

        if (
            alpha**2
            == np.linalg.norm(d) ** 2
            / np.linalg.norm(np.linalg.inv(sqrt_Gamma).T @ c) ** 2
        ):
            X_star = np.linalg.inv(Gamma) @ c @ d.T / np.linalg.norm(sqrt_Gamma.T @ c)
        elif (
            alpha**2
            > np.linalg.norm(d) ** 2
            / np.linalg.norm(np.linalg.inv(sqrt_Gamma).T @ c) ** 2
        ):
            if norm_fun(0, problem_inputs) < 0:
                X_star = X_fun(0, problem_inputs)
            else:
                mu_star = opt.fsolve(norm_fun, 0, problem_inputs)
                X_star = X_fun(mu_star, problem_inputs)
        else:
            warnings.warn("Infeasible")

        return X_star

    def evaluate_objective(X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the QCQP objective 0.5 * E[||X.T @ y(t)||**2] - trace(X.T @ B)."""
        Y = problem_inputs.fused_data[0]
        B = problem_inputs.fused_constants[0]

        Ryy = autocorrelation_matrix(Y)

        f = 0.5 * np.trace(X.T @ Ryy @ X) - np.trace(X.T @ B)

        return f

    def generate_synthetic_inputs(
        self,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        offset: float = 0.5,
        nb_sources: int | None = None,
    ) -> ProblemInputs:
        """Generate synthetic inputs for the QCQP problem."""

        def create_signal(
            signal_var: float,
            noise_var: float,
            offset: float,
            nb_sources: int,
            nb_sensors: int,
            nb_samples: int,
        ):
            """Create signals for the QCQP problem."""
            rng = np.random.default_rng()

            S = rng.normal(
                loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples)
            )
            A = rng.uniform(low=-offset, high=offset, size=(nb_sensors, nb_sources))
            noise = rng.normal(
                loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples)
            )

            Y = A @ S + noise

            return Y

        rng = np.random.default_rng()

        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        Y = create_signal(nb_sensors, nb_samples)
        Ryy = Y @ Y.T / nb_samples
        Ryy = (Ryy + Ryy.T) / 2
        B = rng.standard_normal(size=(nb_sensors, self.nb_filters))
        c = rng.standard_normal(size=(nb_sensors, 1))
        d = rng.standard_normal(size=(Q, 1))
        w = (B.T @ np.linalg.inv(Ryy).T @ c - d) / (c.T @ np.linalg.inv(Ryy) @ c)
        X = np.linalg.inv(Ryy) @ (B - c @ w.T)

        toss = rng.integers(0, 1, endpoint=True)
        if toss == 0:
            alpha = rng.standard_normal()
            alpha = alpha**2
        else:
            alpha = rng.standard_normal()
            alpha = alpha**2
            alpha = np.sqrt(np.linalg.norm(X, ord="fro") ** 2 + alpha**2)

        while alpha**2 < np.linalg.norm(d) ** 2 / np.linalg.norm(c) ** 2:
            c = rng.standard_normal(size=(nb_sensors, 1))
            d = rng.standard_normal(size=(self.nb_filters, 1))
            w = (B.T @ np.linalg.inv(Ryy).T @ c - d) / (c.T @ np.linalg.inv(Ryy) @ c)
            X = np.linalg.inv(Ryy) @ (B - c @ w.T)
            toss = rng.integers(0, 1, endpoint=True)
            if toss == 0:
                alpha = rng.standard_normal()
                alpha = alpha**2
            else:
                alpha = rng.standard_normal()
                alpha = alpha**2
                alpha = np.sqrt(np.linalg.norm(X, ord="fro") ** 2 + alpha**2)

        qcqp_inputs = ProblemInputs(
            fused_data=[Y],
            fused_constants=[B, c],
            fused_quadratics=[np.eye(nb_sensors)],
            global_constants=[alpha, d],
        )

        return qcqp_inputs


class SCQPProblem(OptimizationProblem):
    def __init__(self) -> None:
        super().__init__()

    def solve(self, problem_inputs: ProblemInputs) -> np.ndarray:
        """Solve the SCQP problem min 0.5 * E[||X.T @ y(t)||**2] + trace(X.T @ B) s.t. trace(X.T @ Gamma @ X)=1."""
        Y = problem_inputs.fused_data[0]
        B = problem_inputs.fused_constants[0]
        Gamma = problem_inputs.fused_quadratics[0]

        manifold = Sphere(np.size(B, 0), np.size(B, 1))

        Ryy = autocorrelation_matrix(Y)

        Gamma = make_symmetric(Gamma)

        L = np.linalg.cholesky(Gamma)
        Ryy_t = np.linalg.inv(L) @ Ryy @ np.linalg.inv(L).T
        Ryy_t = make_symmetric(Ryy_t)
        B_t = np.linalg.inv(L) @ B

        @pymanopt.function.autograd(manifold)
        def cost(X):
            return 0.5 * autograd.numpy.trace(X.T @ Ryy_t @ X) + autograd.numpy.trace(
                X.T @ B_t
            )

        problem = Problem(manifold=manifold, cost=cost)
        problem.verbosity = 0

        solver = TrustRegions(verbosity=0)
        X_star = solver.run(problem).point
        X_star = np.linalg.inv(L.T) @ X_star

        return X_star

    def evaluate_objective(X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the SCQP objective 0.5 * E[||X.T @ y(t)||**2] + trace(X.T @ B)."""
        Y = problem_inputs.fused_data[0]
        B = problem_inputs.fused_constants[0]

        Ryy = autocorrelation_matrix(Y)

        f = 0.5 * np.trace(X.T @ Ryy @ X) + np.trace(X.T @ B)

        return f

    def generate_synthetic_inputs(
        self,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        offset: float = 0.5,
        nb_sources: int | None = None,
    ) -> ProblemInputs:
        """Create data for the SCQP problem."""
        rng = np.random.default_rng()

        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = self.nb_filters

        S = rng.normal(loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples))
        A = rng.uniform(low=-offset, high=offset, size=(nb_sensors, nb_sources))
        noise = rng.normal(
            loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples)
        )

        Y = A @ S + noise
        B = rng.standard_normal(size=(nb_sensors, self.nb_filters))

        scqp_inputs = ProblemInputs(
            fused_data=[Y], fused_constants=[B], fused_quadratics=[np.eye(nb_sensors)]
        )

        return scqp_inputs


class RTLSProblem(OptimizationProblem):
    def __init__(
        self,
        convergence_parameters: ConvergenceParameters | None,
        initial_estimate: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            convergence_parameters=convergence_parameters,
            initial_estimate=initial_estimate,
        )

    def solve(self, problem_inputs: ProblemInputs) -> np.ndarray:
        """Solve the RTLS problem max rho = E[|X.T @ y(t) - d(t)|**2] / (1 + X.T @ X) s.t. ||X.T @ L|| <= delta ** 2."""
        Y = problem_inputs.fused_data[0]
        L = problem_inputs.fused_constants[0]
        Gamma = problem_inputs.fused_quadratics[0]
        d = problem_inputs.global_constants[0]
        delta = problem_inputs.global_constants[1]

        rng = np.random.default_rng()
        if self.initial_estimate is None:
            X = rng.normal(
                size=(
                    np.size(Y, 0),
                    self.nb_filters,
                )
            )
        else:
            X = self.initial_estimate
        i = 0

        f = self.evaluate_objective(X, problem_inputs)

        convergence_parameters = (
            self.convergence_parameters
            if self.convergence_parameters
            else ConvergenceParameters()
        )

        Ryy = autocorrelation_matrix(Y)
        ryd = np.sum(Y * d, axis=1) / np.size(Y, 1)
        LL = L @ L.T
        LL = make_symmetric(LL)
        Gamma = make_symmetric(Gamma)

        while i < convergence_parameters.max_iterations:
            X_old = X
            X_f = np.linalg.inv(Ryy - f * Gamma) @ ryd
            if X_f.T @ LL @ X_f < delta**2:
                X = X_f
            else:
                obj = (
                    lambda l: np.linalg.norm(
                        L.T @ np.linalg.inv(Ryy - f * Gamma + l * LL) @ ryd
                    )
                    ** 2
                    - delta**2
                )
                opt_sol = opt.root_scalar(obj, bracket=[0, 1000], method="bisect", x0=0)
                l_star = opt_sol.root

                X = np.linalg.inv(Ryy - f * Gamma + l_star * LL) @ ryd

            f_old = f
            f = self.evaluate_objective(X, problem_inputs)

            if (
                np.absolute(f - f_old) <= convergence_parameters.objective_tolerance
            ) or (
                np.linalg.norm(X - X_old, "fro")
                <= convergence_parameters.argument_tolerance
            ):
                break

            i += 1

            X_star = np.expand_dims(X, axis=1)

        return X_star

    def evaluate_objective(X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the RTLS objective E[|X.T @ y(t) - d(t)|**2] / (1 + X.T @ Gamma @ X)."""
        Y = problem_inputs.fused_data[0]
        Gamma = problem_inputs.fused_quadratics[0]
        d = problem_inputs.global_constants[0]

        Ryy = autocorrelation_matrix(Y)
        ryd = np.sum(Y * d, axis=1) / np.size(Y, 1)
        rdd = np.sum(d * d, axis=1) / np.size(d, 1)

        f = (X.T @ Ryy @ X - 2 * X.T @ ryd + rdd) / (X.T @ Gamma @ X + 1)

        return f

    def generate_synthetic_inputs(
        self,
        signal_var: float = 0.5,
        noise_var: float = 0.2,
        mixture_var: float = 0.3,
        nb_sources: int | None = None,
    ) -> ProblemInputs:
        """Generate synthetic inputs for the RTLS problem."""
        rng = np.random.default_rng()

        nb_samples = self.problem_parameters.nb_samples
        nb_sensors = self.problem_parameters.network_graph.nb_sensors_total

        if nb_sources is None:
            nb_sources = self.nb_filters

        s = rng.normal(loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples))
        s = s - s.mean(axis=1, keepdims=True)
        s = s * np.sqrt(
            signal_var * np.ones((nb_sources, 1)) / s.var(axis=1, keepdims=True)
        )
        Pi_s = rng.normal(
            loc=0, scale=np.sqrt(mixture_var), size=(nb_sensors, nb_sources)
        )
        noise = rng.normal(
            loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples)
        )
        noise = noise - noise.mean(axis=1, keepdims=True)
        noise = noise * np.sqrt(
            noise_var * np.ones((nb_sensors, 1)) / noise.var(axis=1, keepdims=True)
        )

        Y = Pi_s @ s + noise

        d_noisevar = 0.02
        d_noise = rng.normal(
            loc=0, scale=np.sqrt(d_noisevar), size=(nb_sources, nb_samples)
        )
        d_noise = d_noise - d_noise.mean(axis=1, keepdims=True)
        d_noise = d_noise * np.sqrt(
            d_noisevar * np.ones((nb_sources, 1)) / d_noise.var(axis=1, keepdims=True)
        )

        d = s + d_noise

        return Y, d
