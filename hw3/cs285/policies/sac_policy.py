from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        entropy = self.log_alpha.exp()
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        action_distribution = self.forward(ptu.from_numpy(obs))
        if sample:
            action = ptu.to_numpy(action_distribution.sample())
        else:
            action = ptu.to_numpy(action_distribution.mean())
        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> torch.distributions.Distribution:
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file

        mean = self.mean_net(observation)
        std_indv = self.logstd.clamp(*torch.exp(self.log_std_bounds))
        std = std_indv.repeat(mean.shape[0], 1)        
        action_distribution = sac_utils.SquashedNormal(mean, std)

        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        obs = ptu.from_numpy(obs)

        action_distribution = self.forward(obs)

        actions = action_distribution.sample()

        q1, q2 = critic(obs, actions)

        q = torch.min(q1, q2)

        log_prob = action_distribution.log_prob(actions)

        actor_loss = -1 * (q - self.alpha * log_prob).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = torch.nn.MSELoss(self.log_alpha, self.target_entropy)
        alpha_loss.backward()
        

        return ptu.to_numpy(actor_loss), ptu.to_numpy(alpha_loss), ptu.to_numpy(self.alpha)

