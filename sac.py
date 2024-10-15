import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import deque
import time
import pandas as pd
from DDPG_classes import Actor, Critic, OUNoise

class Agent():
    def __init__(self, input_size, output_size, options):
        self.tau = 0.005
        self.discount = 0.99
        self.iters = 0
        self.terminate = False
        self.minibatch_size = 128
        self.replay_memory_size = 100_000
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(input_size, output_size).to(self.device)
        self.critic_1 = Critic(input_size + output_size, 1).to(self.device)
        self.critic_2 = Critic(input_size + output_size, 1).to(self.device)
        self.critic_1_target = Critic(input_size + output_size, 1).to(self.device)
        self.critic_2_target = Critic(input_size + output_size, 1).to(self.device)
        
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        self.target_entropy = -torch.prod(torch.Tensor(output_size).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)
        
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.99)
        self.critic_1_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_1_optimizer, step_size=1000, gamma=0.99)
        self.critic_2_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_2_optimizer, step_size=1000, gamma=0.99)
        
        self.ounoise = OUNoise(output_size)
        self.alpha = 0.2  # Fixed entropy coefficient
        self.target_update_interval = 1
        self.gamma = 0.99
        self.tau = 0.005
        self.reward_scale = 0.1  # New: scale rewards

    def choose_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        if state.device != self.device:
            state = state.to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action, _, _ = self.actor.sample(state)
        return action.cpu().numpy().flatten()

    def learn(self):
        if len(self.replay_memory) < self.minibatch_size:
            return 0, 0, 0  # Return zeros if we don't have enough samples

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.sample_batch()

        # Scale rewards
        reward_batch = reward_batch * self.reward_scale

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target = self.critic_1_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_2_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target

        qf1 = self.critic_1(state_batch, action_batch)
        qf2 = self.critic_2(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value.detach())
        qf2_loss = F.mse_loss(qf2, next_q_value.detach())

        # Update critics
        self.critic_1_optimizer.zero_grad()
        qf1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)  # New: gradient clipping
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        qf2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)  # New: gradient clipping
        self.critic_2_optimizer.step()

        # Update actor
        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi = self.critic_1(state_batch, pi)
        qf2_pi = self.critic_2(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # New: gradient clipping
        self.actor_optimizer.step()

        if self.iters % self.target_update_interval == 0:
            self.soft_update(self.critic_1_target, self.critic_1)
            self.soft_update(self.critic_2_target, self.critic_2)

        self.iters += 1

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def sample_batch(self):
        batch = random.sample(self.replay_memory, self.minibatch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Ensure all states have the same shape
        state = [s.squeeze() if s.ndim == 2 else s for s in state]
        next_state = [s.squeeze() if s.ndim == 2 else s for s in next_state]

        # Convert to numpy arrays first, moving tensors to CPU if necessary
        # for s in state:
        #     print(s.shape)
        state = np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in state])
        action = np.array([a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in action])
        reward = np.array(reward).reshape(-1, 1)
        next_state = np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in next_state])
        done = np.array(done).reshape(-1, 1)

        # Convert numpy arrays to PyTorch tensors
        return (
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(action).to(self.device),
            torch.FloatTensor(reward).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor(done).to(self.device)
        )

    def update_replay_memory(self, transition):
        state, action, reward, next_state, done = transition
        # Ensure consistent shapes and move tensors to CPU if necessary
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else state.squeeze() if isinstance(state, np.ndarray) else state
        action = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        reward = reward.cpu().item() if isinstance(reward, torch.Tensor) else reward.item() if isinstance(reward, np.ndarray) else reward
        next_state = next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state.squeeze() if isinstance(next_state, np.ndarray) else next_state
        done = done.cpu().item() if isinstance(done, torch.Tensor) else done.item() if isinstance(done, np.ndarray) else done
        
        self.replay_memory.append((state, action, reward, next_state, done))

def create_Df(df, data, columns):
    data = pd.DataFrame([data], columns=[columns])
    df = pd.concat([df, data])
    return df



