import math

import torch

from dataclasses import dataclass
from torch.quasirandom import SobolEngine
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from approximate_gp import *
from botorch.models.model import Model
from botorch.acquisition.objective import ScalarizedObjective


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False
    mdn: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    mdn=False,
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    dtype=torch.float32,
    device=torch.device('cpu')
):
    
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    if mdn:
        weights = torch.ones_like(x_center)
    else:
        # ONLY NECESSARY FOR MULTIPLE LENGTH SCALES:
        # weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        # print("weights", weights)
        # print("weights shape", weights.shape)
        # weights = weights / weights.mean()
        # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        weights = torch.ones_like(x_center)

    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        if mdn:
            ei = ExpectedImprovementMDN(model.cuda(), Y.max().cuda(), maximize=True)

            X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        else:
            ei = qExpectedImprovement(model.cuda(), Y.max(), maximize=True)
            # Y.max() is best seen so far
            print("got qei")
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]).cuda(),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
            # optimize_acqf requires model to have a posterior...
            # works for mdn until rsample() fails to produce tensor...
    # print("X_next", X_next) # 1x128, and normal looking numbers... as expected 
    return X_next


class ExpectedImprovementMDN(AnalyticAcquisitionFunction):
    r"""Single-outcome Expected Improvement (analytic).
    Computes classic Expected Improvement over the current best observed value,
    using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.
    `EI(x) = E(max(y - best_f, 0)), y ~ f(x)`
    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.2)
        >>> ei = EI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).
        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    # @t_batch_mode_transform(expected_q=1, assert_output_shape=False)

    def forward(self, X: Tensor) -> Tensor:
        mix = self.model(X.cuda())
        #print("mix", mix)
            # MixtureSameFamily(
            #   Categorical(logits: torch.Size([4000, 10])), #Weights 
            #   Normal(loc: torch.Size([4000, 10]), scale: torch.Size([4000, 10])))
        # print("comp dist", mix.component_distribution)
            #comp dist Normal(loc: torch.Size([4000, 10]), scale: torch.Size([4000, 10]))
        dists = mix.component_distribution
        means = dists.loc # 4000, 10
        wts = mix.mixture_distribution.logits # 4000, 10
        # print("wts shape", wts.shape)
        # wts shape torch.Size([400, 10])
        num_in_mix = dists.loc.shape[-1]
        # print("Num in mix", num_in_mix) # 10
        total_ei = 0
        for i in range(num_in_mix): 
            means = dists.loc[:,i].cuda()
            sigmas = dists.scale[:,i].cuda()
            # means, sigmas torch.Size([400])

            ## Taken from EI code:
            # https://github.com/pytorch/botorch/blob/main/botorch/acquisition/analytic.py
            u = (means - self.best_f.expand_as(means)) / sigmas
            normal = torch.distributions.normal.Normal(torch.zeros_like(u), torch.ones_like(u))
            ucdf = normal.cdf(u)
            updf = torch.exp(normal.log_prob(u))
            ei = sigmas * (updf + u * ucdf)
            # print("ei", ei.shape)
            # ei torch.Size([400])
            ##

            weighted_ei = ei * wts[:,i]
            # print("wtd ei", weighted_ei.shape)
            # wtd ei torch.Size([400])
            total_ei = total_ei + weighted_ei

        # print("total ei", total_ei.shape)
        # total ei torch.Size([400]) or torch.Size([10]) w/ bsz = 11, num_resutarts = 10! 
        return total_ei

# optimize_acqf:
# https://github.com/pytorch/botorch/blob/73698a16f0283b53ee55a8e052b4651c2538270e/botorch/optim/optimize.py

# rsample example (multivariate normal): 
# https://github.com/cornellius-gp/gpytorch/blob/661359cdd7cd56178ca63bde341fa0749c332cfa/gpytorch/distributions/multivariate_normal.py#L18
