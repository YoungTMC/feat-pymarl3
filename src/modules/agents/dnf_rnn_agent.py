import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from modules.layer.mat import Encoder
import math


class CoreRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CoreRNNAgent, self).__init__()
        self.args = args
        # core extraction module
        self.dominators = math.ceil(args.dominator_num)
        self.followers = args.n_agents - self.dominators
        # [1, n_agents, obs_dim] -> [1, n_agents]
        self.core_extractor = nn.Sequential(
            nn.Linear(args.n_agents, args.core_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.core_hidden_dim, args.core_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.core_hidden_dim, args.n_agents)
        )
        self.encoder = Encoder(
            args.state_shape, args.obs_shape, args.n_actions, input_shape,
            args.n_block, args.n_embd, args.n_head, 
            args.n_agents, args.encode_state
        )
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def extractor_forward(self, inputs):
        """
        return: dominators obs & followers obs as agent network's input
        """
        b, a, e = inputs.size()
        # SVD
        s = torch.zeros([b, a], device=inputs.device)
        for i in range(b):
            uu, ss, vv = torch.linalg.svd(inputs[i])
            s.index_put_((torch.tensor([i]),), ss)
        extractor_output = self.core_extractor(s)

        sorted_s, sorted_idx = torch.sort(extractor_output, dim=-1, descending=True)
        # [1, dominators_num]
        dominators_idx = sorted_idx[:, :self.dominators]
        # [1, followers_num]
        followers_idx = sorted_idx[:, self.dominators:]
        return dominators_idx, followers_idx

    def dominator_forward(self, inputs, hidden_state, follower_actions):
        """
        Args:
            inputs: dominators' observations
            hidden_state: RNN hidden state
            follower_actions: actions generated by SMAC rule-based AI for followers
        """
        v_local, inputs_rep = self.encoder.forward(inputs, inputs, follower_actions)
        return self._forward(inputs_rep, hidden_state)

    def follower_forward(self, inputs, hidden_state):
        return self._forward(inputs, hidden_state)

    def _forward(self, inputs, hidden_state):
        # TODO the inputs of followers maybe None.
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)
