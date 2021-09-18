import torch
import logging
from collections import OrderedDict
from botorch.posteriors.posterior import Posterior
from mixture_same_family import MixtureSameFamily

from botorch.models.model import Model

from base import DenseNetwork


# class MDN(torch.nn.Module): 
# #extend model instead, then define posterior method that calls forward? JK bigger problem... 
class MDN(Model):
    def __init__(self, input_dim, num_train, output_dim=None, hidden_dims=(1000, 1000, 500, 50), mixture_size=10):
        super().__init__()

        self.output_dim = output_dim if output_dim is not None else 1
        self.flat_output = output_dim is None
        self.mixture_size = mixture_size
       # self.num_outputs = 1 

        # Feature extractor
        self.hidden_layers = DenseNetwork(input_dim, hidden_dims)
        hidden_dim = self.hidden_layers.output_dim

        # Final output layer: mean and variance
        n_outputs = 3 * self.output_dim * mixture_size
        if hidden_dim < n_outputs:
            logging.warn(f'Hidden dimension {hidden_dim} is smaller than required {n_outputs} outputs.')
        self.final_layer = torch.nn.Linear(hidden_dim, n_outputs)

        # Parameters for the features / noise
        self.register_parameter("_noise", torch.nn.Parameter(torch.zeros(self.output_dim)))

        self.register_parameter("var_w", torch.nn.Parameter(torch.zeros(self.output_dim)))
        self.register_parameter("var_b", torch.nn.Parameter(torch.zeros(self.output_dim)))

    def forward(self, x):
        features = self.hidden_layers(x)
        mean, _std, weight_logits = self.final_layer(features).reshape((x.shape[0], 3, self.output_dim, self.mixture_size)).transpose(0, 1)
        std = torch.nn.functional.softplus(_std) + torch.nn.functional.softplus(self._noise)[:, None]

        if not self.training:
            w = torch.nn.functional.softplus(self.var_w) + 1.
            b = torch.nn.functional.softplus(self.var_b)

            logging.debug(f'var_w = {w.mean().item()}, var_b = {b.mean().item()}')
            std = w[:, None] * std + b[:, None]

        if self.flat_output:
            weight_logits = weight_logits[:, 0, :]
            mean = mean[:, 0, :]
            std = std[:, 0, :]

        # Output
        mixture_distribution = torch.distributions.Categorical(logits=weight_logits)
        component_distribution = torch.distributions.Normal(mean, std)
        output = MixtureSameFamily(mixture_distribution, component_distribution)
        return output

    def initialize(self, train_x):
        return self

    def loss(self, output, target):
        losses = -output.log_prob(target)
        return losses.mean()

    def parameters_nowd(self):
        return [self._noise]

    def parameters_wd(self):
        return list(set(self.parameters()) - set(self.parameters_nowd()) - set(self.parameters_valid()))

    def parameters_valid(self):
        return [self.var_w, self.var_b]
    
    def posterior(self, X):
        dist = self(X)
        # post = Posterior()
        # post.posterior = dist  #FIGURE OUT HOW TO WRAP!!
        # need to set sampler...
        dist.dtype = dist.mean.dtype
        dist.device = torch.device('cuda')
        dist.base_sample_shape = dist._component_distribution.event_shape
        return dist

    @property
    def num_outputs(self): 
        return self.output_dim

# samples = self.sampler(posterior)
# AttributeError: 'MixtureSameFamily' object has no attribute 'base_sample_shape'

# posterior = GPyTorchPosterior(mvn=mvn)
# return GPyTorchPosterior(
#             mvn=MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
#         )

# Model Class:
# https://github.com/pytorch/botorch/blob/main/botorch/models/model.py
