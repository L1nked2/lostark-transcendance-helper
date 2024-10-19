import torch
import numpy as np
from model.model import DQN
ACTION_SPACE_SIZE = 131
OBSERVATION_SPACE_SIZE = 362

PATH = f'./model/dqn.pth'

def predict(given_observation):
  given_observation = np.array(given_observation, dtype=np.float32)
  given_observation = torch.from_numpy(given_observation)
  # load model
  policy_net = DQN(OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE)
  policy_net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu'), weights_only=True))
  policy_net.eval()

  # predict

  with torch.no_grad():
    max_q = torch.max(policy_net.forward(given_observation), 0)
  max_q_value, action_idx = max_q[0].item(), max_q[1].item()

  return policy_net.forward(given_observation).tolist()
 

