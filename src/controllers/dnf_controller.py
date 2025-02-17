import torch
import math

from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class CoreMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(CoreMAC, self).__init__(scheme, groups, args)
        self.dominator_num = self.args.dominator_num
        self.dominators_idx, self.followers_idx = self._init_idx()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, ps=None):
        if t_ep == 0:
            self.set_evaluation_mode()
        if ps is not None:
            self.ps = ps
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        use_rule_follower = getattr(self.args, "use_rule_follower", False)
        if use_rule_follower:
            qvals, followers_actions = self.forward(ep_batch, t_ep, test_mode=test_mode, use_rule_follower=use_rule_follower)
            followers_actions = followers_actions.squeeze(-1)
            # 获取dominator的动作
            dominator_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env,
                                                                   test_mode=test_mode,
                                                                   use_rule_follower=True,
                                                                   dominators_idx=self.dominators_idx)
            
            # 创建完整的动作tensor
            batch_size = avail_actions.shape[0]
            n_agents = self.args.n_agents
            all_actions = torch.zeros((batch_size, n_agents), dtype=torch.long, device=qvals.device)
            
            # 填充dominator的动作
            for b_idx in range(batch_size):
                for d_idx, agent_id in enumerate(self.dominators_idx[b_idx]):
                    all_actions[b_idx, agent_id] = dominator_actions[b_idx, d_idx]
                
                # 填充follower的动作
                for f_idx, agent_id in enumerate(self.followers_idx[b_idx]):
                    all_actions[b_idx, agent_id] = followers_actions[b_idx, f_idx]
            
            return all_actions
        else:
            qvals = self.forward(ep_batch, t_ep, test_mode=test_mode, use_rule_follower=use_rule_follower)
            chose_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env,
                                                               test_mode=test_mode, 
                                                               use_rule_follower=False)
            return chose_actions

    def forward(self, ep_batch, t, test_mode=False, use_rule_follower=False):
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
        dominators_inputs = agent_inputs[d_mask].view(b, self.dominator_num, e)
        followers_inputs = agent_inputs[f_mask].view(b, self.n_agents - self.dominator_num, e)
        # hidden states
        dominators_hidden_states = self.hidden_states[d_mask].view(b, self.dominator_num, self.args.rnn_hidden_dim)

        if use_rule_follower:
            # 使用SMAC的规则智能体为followers生成动作
            followers_actions = []
            for b_idx in range(b):
                batch_actions = []
                for f_idx in range(self.followers_idx.shape[1]):
                    agent_id = self.followers_idx[b_idx][f_idx].item()
                    # 获取当前follower的观察
                    obs = followers_inputs[b_idx, f_idx].cpu().numpy()
                    # 通过已存在的进程管道发送请求获取规则动作
                    if hasattr(self, 'ps') and self.ps is not None:
                        parent_conn = self.ps[b_idx]
                        parent_conn.send(("get_agent_action_heuristic", (agent_id, None)))
                        action = parent_conn.recv()
                    else:
                        raise RuntimeError("Process connections not initialized. Make sure ps is set.")
                    batch_actions.append(action)
                followers_actions.append(batch_actions)
            
            # 转换为三维tensor: [batch_size, n_followers, 1]
            followers_actions = torch.tensor(followers_actions, dtype=torch.float, device=device).unsqueeze(-1)
            
            # 主导者用agent_network决策，但输入中包含跟随者的动作
            outputs, _ = self.agent.dominator_forward(
                dominators_inputs, 
                dominators_hidden_states, 
                followers_actions
            )
            return outputs, followers_actions
        else:
            # 主导者和跟随者都用agent_network决策
            followers_hidden_states = self.hidden_states[f_mask].view(b, self.n_agents - self.dominator_num, self.args.rnn_hidden_dim)
            combined_inputs = torch.cat([dominators_inputs, followers_inputs], dim=1)
            combined_hidden_states = torch.cat([dominators_hidden_states, followers_hidden_states], dim=1)
            outputs, _ = self.agent.follower_forward(combined_inputs, combined_hidden_states)
            return outputs

    def _init_idx(self):
        dominators_idx = torch.zeros(self.dominator_num)
        followers_idx = torch.zeros(self.n_agents - self.dominator_num)
        print("dominators_idx: {}, followers_idx: {}", dominators_idx, followers_idx)
        return dominators_idx, followers_idx
