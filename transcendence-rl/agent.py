from model.model import DQN
import numpy as np
import numpy.ma as ma
import math
import torch
import torch.optim as optim
from utils import flatten_dict_concat
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class Agent(object):
  def __init__(self, env, model, device, batch_size=128,
               gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=1000, 
               tau=0.005, lr=1e-4):
    self.env = env
    self.device = device
    self.batch_size = batch_size
    self.gamma = gamma
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_decay = eps_decay
    self.tau = tau
    self.lr = lr

    # setup env
    # get observation and action space size
    n_actions = self.env.action_space.n
    # Get the number of state observations
    state, info = env.reset(options={"case_num": 0})
    n_observations = len(flatten_dict_concat(state))

    # setup model
    if model.lower() == 'dqn':
      self.policy_net = DQN(n_observations, n_actions).to(device)
      self.target_net = DQN(n_observations, n_actions).to(device)
      self.target_net.load_state_dict(self.policy_net.state_dict())
      self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
      self.memory = ReplayMemory(10000)
    else:
      raise NotImplementedError
    
    self.episode_durations = []
    self.steps_done = 0
  
  def select_action(self, state):
    sample = np.random.random()
    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        math.exp(-1. * self.steps_done / self.eps_decay)
    self.steps_done += 1
    mask = self.env.unwrapped.get_action_mask()
    available_actions = ma.masked_array(np.arange(self.env.action_space.n), mask=~mask).compressed()
    if sample > eps_threshold:
        # get best q value-action from policy net
        with torch.no_grad():
            q_values = self.policy_net(state)
        max_q_value = float('-inf')
        for action in available_actions:
            q_value = q_values[0][action]
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action
        return torch.tensor([[best_action]], device=self.device, dtype=torch.long)
    else:
        # explore
        selected_action = np.random.choice(available_actions)
        return torch.tensor([[selected_action]], device=self.device, dtype=torch.long)
  
  def learn(self):
    if len(self.memory) < self.batch_size:
        return
    transitions = self.memory.sample(self.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(self.batch_size, device=self.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()
