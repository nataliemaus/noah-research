import torch
from torch.distributions.distribution import Distribution
from torch.distributions import Categorical
from torch.distributions import constraints
from typing import Dict


class MixtureSameFamily(Distribution):
    r"""
    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all component are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` component) and a component
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.

    Examples::

        # Construct Gaussian Mixture Model in 1D consisting of 5 equally
        # weighted normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
        >>> gmm = MixtureSameFamily(mix, comp)

        # Construct Gaussian Mixture Modle in 2D consisting of 5 equally
        # weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Independent(D.Normal(
                     torch.randn(5,2), torch.rand(5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

        # Construct a batch of 3 Gaussian Mixture Models in 2D each
        # consisting of 5 random weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.rand(3,5))
        >>> comp = D.Independent(D.Normal(
                    torch.randn(3,5,2), torch.rand(3,5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

    Args:
        mixture_distribution: `torch.distributions.Categorical`-like
            instance. Manages the probability of selecting component.
            The number of categories must match the rightmost batch
            dimension of the `component_distribution`. Must have either
            scalar `batch_shape` or `batch_shape` matching
            `component_distribution.batch_shape[:-1]`
        component_distribution: `torch.distributions.Distribution`-like
            instance. Right-most batch dimension indexes component.
    """
    arg_constraints: Dict[str, constraints.Constraint] = {}
    has_rsample = True

    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 reparameterize = True,
                 validate_args=None):
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution
        # print("creating MIX!")

        if not isinstance(self._mixture_distribution, Categorical):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of torch.distribtutions.Categorical")

        if not isinstance(self._component_distribution, Distribution):
            raise ValueError("The Component distribution need to be an "
                             "instance of torch.distributions.Distribution")

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._component_distribution.batch_shape[:-1]
        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                                 "compatible with `component_distribution."
                                 "batch_shape`({1})".format(mdbs, cdbs))

        # Check that the number of mixture component matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution component` ({0}) does not"
                             " equal `component_distribution.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self._num_component = km

        self.reparameterize = reparameterize

        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)
        super(MixtureSameFamily, self).__init__(batch_shape=cdbs,
                                                event_shape=event_shape,
                                                validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        batch_shape_comp = batch_shape + (self._num_component,)
        new = self._get_checked_instance(MixtureSameFamily, _instance)
        new._component_distribution = \
            self._component_distribution.expand(batch_shape_comp)
        new._mixture_distribution = \
            self._mixture_distribution.expand(batch_shape)
        new._num_component = self._num_component
        new._event_ndims = self._event_ndims
        event_shape = new._component_distribution.event_shape
        super(MixtureSameFamily, new).__init__(batch_shape=batch_shape,
                                               event_shape=event_shape,
                                               validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    def support(self):
        # FIXME this may have the wrong shape when support contains batched
        # parameters
        return self._component_distribution.support

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def component_distribution(self):
        return self._component_distribution

    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(probs * self.component_distribution.mean,
                         dim=-1 - self._event_ndims)  # [B, E]

    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(probs * self.component_distribution.variance,
                                  dim=-1 - self._event_ndims)
        var_cond_mean = torch.sum(probs * (self.component_distribution.mean -
                                           self._pad(self.mean)).pow(2.0),
                                  dim=-1 - self._event_ndims)
        return mean_cond_var + var_cond_mean

    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.cdf(x)
        mix_prob = self.mixture_distribution.probs

        return torch.sum(cdf_x * mix_prob, dim=-1)

    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1)))
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        print("trying to rsample...")

        # get samples without gradients first:
        samples = self.sample(sample_shape=sample_shape)
        # now we need to add derivatives to samples
        # and return samples with deriviatives...

        # From paper:
        # https://papers.nips.cc/paper/2018/file/92c8c96e4c37100777c7190b76d28233-Paper.pdf
        # dy/dphi = -(d cdf(y) / dy)^-1 * (d cdf(y) / dphi)
        # in above equation, y is samples, and phi is mixture distribution parameters
        cdf = self.cdf(samples)
        external_grad = torch.ones(cdf.shape)
        cdf.backward(gradient=external_grad)
        d_cdf_dy = samples.grad
        d_cdf_dphi = phi.grad
        dy_dphi = -1*(d_cdf_dy)**-1 * d_cdf_dphi
        # I think i need to do this for each phi param seperately,
        # and then put in one big matrix?...

        #set new calculated grad for samples
        samples.grad = dy_dphi
        return samples

        r"""Sample from the posterior (with gradients).
        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler`.
                This is used for deterministic optimization.
        Returns:
            A `sample_shape x event`-dim Tensor of samples from the posterior.
        """

    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_ndims)

    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        x = x.reshape(xs[:-1] + torch.Size(pad_ndims * [1]) +
                      xs[-1:] + torch.Size(self._event_ndims * [1]))
        return x

    def __repr__(self):
        args_string = '\n  {},\n  {}'.format(self.mixture_distribution,
                                             self.component_distribution)
        return 'MixtureSameFamily' + '(' + args_string + ')'