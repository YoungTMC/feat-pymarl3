import torch
import math

from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class CoreMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(CoreMAC, self).__init__(scheme, groups, args)
        self.dominators_idx, self.followers_idx = self._init_idx()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chose_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chose_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        device = agent_inputs.device
        # [batch_size, agent_num, obs_dim + (agent_id_num)]
        b, a, e = agent_inputs.size()
        # get dominators and followers
        self.dominators_idx, self.followers_idx = self.agent.extractor_forward(agent_inputs)
        d_mask = torch.zeros(agent_inputs.shape[:2], dtype=torch.bool, device=device)
        d_mask.scatter_(1, self.dominators_idx, True)
        f_mask = torch.zeros(agent_inputs.shape[:2], dtype=torch.bool, device=device)
        f_mask.scatter_(1, self.followers_idx, True)
        # inputs
        dominators_inputs = agent_inputs[d_mask].view(b, self.core_agents, e)
        followers_inputs = agent_inputs[f_mask].view(b, self.n_agents - self.core_agents, e)
        # hidden states
        dominators_hidden_states = self.hidden_states[d_mask].view(b, self.core_agents, self.args.rnn_hidden_dim)
        followers_hidden_states = self.hidden_states[f_mask].view(b, self.n_agents - self.core_agents,
                                                                  self.args.rnn_hidden_dim)

        combined_inputs = torch.cat([dominators_inputs, followers_inputs], dim=1)
        combined_hidden_states = torch.cat([dominators_hidden_states, followers_hidden_states], dim=1)
        # 主导者和跟随者都用agent_network决策
        combined_outputs, combined_hidden_states = self.agent.follower_forward(combined_inputs, combined_hidden_states)
        return combined_outputs

    # TODO 完成follower用规则智能体决策的代码。说明：followers决策 -> 将followers的动作作为输入拼接到dominators的输入中，再输入到dominators的网络中进行决策。
    def forward_rule_follower(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        device = agent_inputs.device
        # [batch_size, agent_num, obs_dim + (agent_id_num)]
        b, a, e = agent_inputs.size()
        # get dominators and followers
        self.dominators_idx, self.followers_idx = self.agent.extractor_forward(agent_inputs)
        d_mask = torch.zeros(agent_inputs.shape[:2], dtype=torch.bool, device=device)
        d_mask.scatter_(1, self.dominators_idx, True)
        f_mask = torch.zeros(agent_inputs.shape[:2], dtype=torch.bool, device=device)
        f_mask.scatter_(1, self.followers_idx, True)
        # inputs
        dominators_inputs = agent_inputs[d_mask].view(b, self.core_agents, e)
        followers_inputs = agent_inputs[f_mask].view(b, self.n_agents - self.core_agents, e)
        # hidden states
        dominators_hidden_states = self.hidden_states[d_mask].view(b, self.core_agents, self.args.rnn_hidden_dim)
        # 跟随者用规则智能体决策
        followers_actions = self.forward_rule_follower(followers_inputs)
        # 主导者用agent_network决策，但输入中包含跟随者的动作
        dominators_outputs, _ = self.agent.dominator_forward(dominators_inputs, dominators_hidden_states, followers_actions)
        return dominators_outputs, followers_actions
    def forward_rule_follower(self, followers_inputs):
        """
        Rule-based decision making for followers
        Args:
            followers_inputs: Tensor of shape [batch_size, n_followers, obs_dim]
        Returns:
            actions: Tensor of shape [batch_size, n_followers] containing selected actions
        """
        # Simple rule: choose action with maximum value in the last obs_dim/2 features
        batch_size, n_followers, obs_dim = followers_inputs.shape
        half_dim = obs_dim // 2
        action_values = followers_inputs[:, :, -half_dim:]
        actions = torch.argmax(action_values, dim=-1)
        return actions

    def _init_idx(self):
        self.core_agents = math.ceil(self.args.core_agent_ratio * self.n_agents)
        dominators_idx = torch.zeros(self.core_agents)
        followers_idx = torch.zeros(self.n_agents - self.core_agents)
        print("dominators_idx: {}, followers_idx: {}", dominators_idx, followers_idx)
        return dominators_idx, followers_idx
