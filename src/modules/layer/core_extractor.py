import torch
import torch.nn as nn
import math
from torch.cuda import device

from torch.nn import LayerNorm
from utils.th_utils import orthogonal_init_

class CoreExtractor(nn.Module):
    def __init__(self, input_shape, args):
        super(CoreExtractor, self).__init__()
        self.args = args
        # core extraction module
        self.dominators = math.ceil(args.core_agent_ratio * args.n_agents)
        self.followers = args.n_agents - self.dominators

        if args.core_extractor_type == 'random':
            return
        elif args.core_extractor_type == 'nn':
            self.fc1 = nn.Linear(input_shape, args.core_hidden_dim)
        elif args.core_extractor_type == 'svd':
            self.fc1 = nn.Linear(args.n_agents, args.core_hidden_dim)
        self.fc2 = nn.Linear(args.core_hidden_dim, args.core_hidden_dim)
        self.fc3 = nn.Linear(args.core_hidden_dim, args.n_agents)
        self.core_extractor = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)
            orthogonal_init_(self.fc3, gain=args.gain)

    def forward(self, x):
        """
        return: dominators obs & followers obs as agent network's input
        """
        if self.args.core_extractor_type == 'nn':
            return self._forward_nn(x)
        elif self.args.core_extractor_type == 'svd':
            return self._forward_svd(x)
        elif self.args.core_extractor_type == 'random':
            return self._forward_random(x)

    def _forward_nn(self, inputs):
        # [batch_size, n_agents, obs_dim]
        b, a, e = inputs.size()
        extractor_output = self.core_extractor(inputs)
        sorted_s, sorted_idx = torch.sort(extractor_output.mean(dim=-1), dim=-1, descending=True)
        # [1, dominators_num]
        dominators_idx = sorted_idx[:, :self.dominators]
        # [1, followers_num]
        followers_idx = sorted_idx[:, self.dominators:]
        return dominators_idx, followers_idx

    def _forward_svd(self, inputs):
        b, a, e = inputs.size()
        # SVD
        uu, ss, vv = torch.linalg.svd(inputs.to('cpu'))
        extractor_output = self.core_extractor(ss.to(inputs.device))
        sorted_s, sorted_idx = torch.sort(extractor_output, dim=-1, descending=True)
        # [1, dominators_num]
        dominators_idx = sorted_idx[:, :self.dominators]
        # [1, followers_num]
        followers_idx = sorted_idx[:, self.dominators:]
        return dominators_idx, followers_idx

    def _forward_random(self, inputs):
        b, a, e = inputs.size()
        rand = torch.rand((b, a), device=inputs.device)
        sorted_s, sorted_idx = torch.sort(rand, dim=-1, descending=True)
        dominators_idx, followers_idx = sorted_idx[:, :self.dominators], sorted_idx[:, self.dominators:]
        return dominators_idx, followers_idx