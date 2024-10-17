import numpy as np
# import pygame
from enum import Enum
import json
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from transcendence_gym.card import Card
from transcendence_gym.board import Board
from transcendence_gym.constants import CardType, BaseBlockType, SpecialBlockType, DistortApplicapableEffects, MAX_BOARD_SIZE, BASE_CARD_TYPE_SIZE
from transcendence_gym.game import Game

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='transcendence-sim-v0',
    entry_point='transcendence_gym.transcendence_gym:TranscendenceEnv',
)

class TranscendenceEnv(gym.Env):
    CARD_PROBABILITY_TABLE = [0.150, 0.115, 0.095, 0.055, 0.105, 0.070, 0.090, 0.070, 0.100, 0.150]
    BLOCK_PROBABILITY_TABLE = [0.160, 0.160, 0.160, 0.235, 0.170, 0.115]
    DEFAULT_GOLD_RESET = 925
    DEFAULT_GOLD_USE_CARD = 140
    DEFAULT_GOLD_WIN = 1000000
    with open('transcendence_gym/presets.json') as f:
        PRESETS = json.load(f)
        EQUIPMENT_NUM = len(PRESETS["boards"]) 
        STAGE_NUM = len(PRESETS["boards"]["head"])
    
    def __init__(self):

        # use card_1 or card_2 on somewhere: 8 * 8 * 2 = 128
        # change card_1 or card_2: 2
        # reset: 1
        # total 131
        self.action_space = spaces.Discrete(131)
        # 32 types of cards, 5 cards of queue
        # 8 * 8 board, 10 types of blocks
        # available count for 3-star step, 0-11
        # available count for draw new card, 0-11
        # total 75 = 2 + 3 + 64 + 3 + 1 + 1 + 1
        CARD_TYPES = len(CardType)
        DEFAULT_BLOCK_TYPES = len(BaseBlockType)
        SPECIAL_BLOCK_TYPES = len(SpecialBlockType)
        self.observation_space = spaces.Dict({"hand":spaces.MultiDiscrete([CARD_TYPES, CARD_TYPES], dtype=np.uint8), 
                                              "queue":spaces.MultiDiscrete([CARD_TYPES for _ in range(3)], dtype=np.uint8), 
                                              "board":spaces.MultiDiscrete(np.full((MAX_BOARD_SIZE, MAX_BOARD_SIZE), DEFAULT_BLOCK_TYPES), dtype=np.uint8),
                                              "special_block":spaces.MultiDiscrete((MAX_BOARD_SIZE, MAX_BOARD_SIZE, SPECIAL_BLOCK_TYPES), dtype=np.uint8),
                                              "remaining_step":spaces.Discrete(12), 
                                              "drawable_count":spaces.Discrete(12),
                                              "protection_level":spaces.Discrete(11)})
        self.rng = np.random.default_rng()
        
    def _get_obs(self):
        observation = {
            "hand": np.array(self.game.card_queue.hand, dtype=np.uint8),
            "queue": np.array(self.game.card_queue.queue, dtype=np.uint8),
            "board": self.game.board.values,
            "special_block": self.game.board.special_block,
            "remaining_step": self.game.remaining_step,
            "drawable_count": self.game.drawable_count,
            "protection_level": self.game.protection_level
        }
        return observation

    def _get_info(self):
        block_unique, block_counts = np.unique(self.game.board.values, return_counts=True)
        remaining_blocks = dict(zip(block_unique, block_counts))
        return {"remaining_blocks": remaining_blocks}

    def _draw_card(self):
        # loop until card queue is filled
        while not self.game.card_queue.isValid():
            # draw new card according to card probability
            new_card = self.rng.choice(BASE_CARD_TYPE_SIZE, p=TranscendenceEnv.CARD_PROBABILITY_TABLE)
            self.game.fillCard(new_card)

    def _set_special_block(self):
        # clear prev special block and add new special block
        self.game.board.special_block = np.array([0, 0, 0], dtype=np.uint8)
        candidates = np.transpose(np.where(self.game.board.values == 3))
        if len(candidates) > 0:
          pos = self.rng.choice(candidates)
          self.game.board.special_block[0:2] = pos
          self.game.board.special_block[2] = self.rng.choice(range(len(SpecialBlockType)), p=TranscendenceEnv.BLOCK_PROBABILITY_TABLE)

    def reset(self, seed=None, options=None):
        if options and "case_num" in options:
            self.case_num = options["case_num"]
        else:
            self.case_num = self.rng.choice(TranscendenceEnv.EQUIPMENT_NUM * TranscendenceEnv.STAGE_NUM)
        equipment_types = list(TranscendenceEnv.PRESETS["boards"].keys())
        equipment_type = equipment_types[self.case_num // TranscendenceEnv.STAGE_NUM]
        self.stage = self.case_num % TranscendenceEnv.STAGE_NUM + 1
        self.game = Game(equipment_type, self.stage)
        self._draw_card()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def is_action_available(self, action):
      if action < 128:
          card = self.game.card_queue.hand[0] if action < 64 else self.game.card_queue.hand[1]
          x = action//8 if action < 64 else (action-64)//8
          y = action%8 if action < 64 else (action-64)%8
          if self.game.board[x, y] < 2:
              return False
          elif self.game.board[x, y] == 2 and \
                not (Card(card).getCardEffect() in DistortApplicapableEffects):
              return False
      elif (action == 128 or action == 129) and self.game.drawable_count <= 0:
          return False
      elif action == 130:
          pass
      else:
          return False
      return True

    def step(self, action):
        # check action is available
        if not self.is_action_available(action):
            reward = 0
        else:
            reward = -self.DEFAULT_GOLD_USE_CARD
            # parse action
            # type 1: use card
            if action < 128:
                slot = 0 if action < 64 else 1
                x = action//8 if action < 64 else (action-64)//8
                y = action%8 if action < 64 else (action-64)%8
                self.game.useCard(slot, x, y)
                self._set_special_block()
            # type 2: replace card, it's ensured that drawable_count > 0
            elif action == 128 or action == 129:
                self.game.replaceCard(action - 128)
            # type 3: reset and start new one
            elif action == 130:
                self.game.restore()
                reward = -self.DEFAULT_GOLD_RESET
            else:
                raise ValueError("Invalid action: {}".format(action))
            self._draw_card()
        observation = self._get_obs()
        terminated = self.game.isTerminated
        truncated = False#self.game.remaining_step <= 0
        if terminated:
            reward = self.DEFAULT_GOLD_WIN
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        pass
