import numpy as np
import gymnasium as gym
from transcendence_gym.transcendence_gym import TranscendenceEnv


if __name__ == '__main__':
    env = gym.make('transcendence-sim-v0')
    options = {
      "case_num": 0,
    }
    
