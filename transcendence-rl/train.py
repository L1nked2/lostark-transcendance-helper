import numpy as np
import gymnasium as gym
from transcendence_gym.transcendence_gym import TranscendenceEnv
from itertools import count
import torch
from agent import Agent
from tqdm import tqdm
from utils import flatten_dict_concat

PATH = f'./model/dqn.pth'

if __name__ == '__main__':
    env = gym.make('transcendence-sim-v0')
    options = {
      "case_num": 0,
    }

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(env, 'dqn', device, batch_size=BATCH_SIZE,
                  gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, 
                  tau=TAU, lr=LR)
    
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50
    print("episodes: ", num_episodes)
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get its state
        state, info = env.reset(options={"case_num": i_episode % TranscendenceEnv.EQUIPMENT_NUM * TranscendenceEnv.STAGE_NUM})
        state = flatten_dict_concat(state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            observation = flatten_dict_concat(observation)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.learn()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                agent.episode_durations.append(t + 1)
                break

    torch.save(agent.policy_net.state_dict(), PATH)              
    print('Complete')