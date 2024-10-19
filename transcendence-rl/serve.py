import torch
from model.model import DQN
import gymnasium as gym
from transcendence_gym.transcendence_gym import TranscendenceEnv
from utils import flatten_dict_concat
import json
PATH = f'./model/dqn.pth'

if __name__ == '__main__':
  env = gym.make('transcendence-sim-v0')
  # get observation and action space size
  n_actions = env.action_space.n
  state, info = env.reset(options={"case_num": 0})
  print(state)
  state = flatten_dict_concat(state)
  n_observations = len(state)
  print(state)
  print(state.tolist())
  p = "./test.json"
  with open(p, "w") as f:
    json.dump(state.tolist(), f)
  # load model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  policy_net = DQN(n_observations, n_actions).to(device)
  policy_net.load_state_dict(torch.load(PATH, weights_only=True))
  policy_net.eval()

  


