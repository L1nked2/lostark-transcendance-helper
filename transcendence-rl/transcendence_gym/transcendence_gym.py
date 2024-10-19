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
from transcendence_gym.constants import CardType, BaseBlockType, SpecialBlockType, DistortApplicapableEffects, MAX_BOARD_SIZE, BASE_CARD_TYPE_SIZE, TOTAL_CARD_TYPE_SIZE, HAND_SIZE, QUEUE_SIZE
from transcendence_gym.game import Game

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='transcendence-sim-v0',
    entry_point='transcendence_gym.transcendence_gym:TranscendenceEnv',
)

class TranscendenceEnv(gym.Env):
    CARD_PROBABILITY_TABLE = [0.150, 0.115, 0.095, 0.055, 0.105, 0.070, 0.090, 0.070, 0.100, 0.150]
    BLOCK_PROBABILITY_TABLE = [0.160, 0.160, 0.160, 0.235, 0.170, 0.115]
    DEFAULT_GOLD_USE_CARD = 0.02
    DEFAULT_GOLD_WIN = 1.0
    DEFAULT_GOLD_BLOCK = 0.01
    with open('transcendence_gym/presets.json') as f:
        PRESETS = json.load(f)
        EQUIPMENT_NUM = len(PRESETS["boards"]) 
        STAGE_NUM = len(PRESETS["boards"]["head"])
    
    def __init__(self):
        self.board_entry_size = MAX_BOARD_SIZE * MAX_BOARD_SIZE
        self.use_action_size = self.board_entry_size * HAND_SIZE
        # use card_1 or card_2 on somewhere: 8 * 8 * 2 = 128
        # change card_1 or card_2: 2
        # reset: 1
        # total 131
        self.action_space = spaces.Discrete(self.use_action_size + HAND_SIZE + 1)
        # observation:
        # board size = 64
        # card type size = 32
        # available count for 3-star step, 0~11
        # available count for draw new card, 0~11
        # available count for protection level, 0~10
        # normal block board + distorted block board + special block board + special block type
        # + hand0 + hand1 + queue0 + queue1 + queue2
        # + remaining step + drawable count + protection level
        # total 329 = 64 + 64 + 64 + 6 + 32 + 32 + 32 + 32 + 1 + 1 + 1
        SPECIAL_BLOCK_TYPE_SIZE = len(SpecialBlockType)
        self.observation_space = spaces.Dict({"normal_block_board": spaces.MultiBinary((MAX_BOARD_SIZE, MAX_BOARD_SIZE)),
                                              "distorted_block_board": spaces.MultiBinary((MAX_BOARD_SIZE, MAX_BOARD_SIZE)),
                                              "special_block_board": spaces.MultiBinary((MAX_BOARD_SIZE, MAX_BOARD_SIZE)),
                                              "special_block_type": spaces.MultiBinary(SPECIAL_BLOCK_TYPE_SIZE),
                                              "hand_0": spaces.MultiBinary(TOTAL_CARD_TYPE_SIZE),
                                              "hand_1": spaces.MultiBinary(TOTAL_CARD_TYPE_SIZE),
                                              "queue_0": spaces.MultiBinary(TOTAL_CARD_TYPE_SIZE),
                                              "queue_1": spaces.MultiBinary(TOTAL_CARD_TYPE_SIZE),
                                              "queue_2": spaces.MultiBinary(TOTAL_CARD_TYPE_SIZE),
                                              "remaining_step": spaces.Discrete(12), 
                                              "drawable_count": spaces.Discrete(12),
                                              "protection_level": spaces.Discrete(11)})
        self.rng = np.random.default_rng()
        
    def _get_obs(self):
        board = self.game.board.values
        normal_only = np.vectorize(lambda x: 1 if x == BaseBlockType.NORMAL else 0)
        distorted_only = np.vectorize(lambda x: 1 if x == BaseBlockType.DISTORTED else 0)
        special_board = np.zeros((MAX_BOARD_SIZE, MAX_BOARD_SIZE), dtype=np.uint8)
        if self.game.board.special_block[2] != SpecialBlockType.NONE:
          special_board[self.game.board.special_block[0], self.game.board.special_block[1]] = 1
        special_block_type = np.zeros(len(SpecialBlockType), dtype=np.uint8)
        special_block_type[self.game.board.special_block[2]] = 1
        hand = [np.zeros(TOTAL_CARD_TYPE_SIZE, dtype=np.uint8) for _ in range(HAND_SIZE)]
        for i in range(HAND_SIZE):
            hand[i][self.game.card_queue.hand[0]] = 1
        queue = [np.zeros(TOTAL_CARD_TYPE_SIZE, dtype=np.uint8) for _ in range(QUEUE_SIZE)]
        for i in range(QUEUE_SIZE):
            queue[i][self.game.card_queue.queue[i]] = 1
        observation = {
            "normal_block_board": normal_only(board),
            "distorted_block_board": distorted_only(board),
            "special_block_board": special_board,
            "special_block_type": special_block_type,
            "hand_0": hand[0],
            "hand_1": hand[1],
            "queue_0": queue[0],
            "queue_1": queue[1],
            "queue_2": queue[2],
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
          self.game.board.special_block[2] = self.rng.choice(range(1, len(SpecialBlockType)), p=TranscendenceEnv.BLOCK_PROBABILITY_TABLE)

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
    
    def get_action_mask(self):
      if 0 < self.game.remaining_step:
        mask = np.ones((1, self.action_space.n), dtype=bool)
        for action in range(mask.shape[1]):
          if action < self.use_action_size:
              slot = action // self.board_entry_size
              use_action = action % self.board_entry_size
              x = use_action // MAX_BOARD_SIZE
              y = use_action % MAX_BOARD_SIZE
              card = self.game.card_queue.hand[slot]
              if self.game.board[x, y] < 2:
                  mask[0, action] = False
              elif self.game.board[x, y] == 2 and \
                    not (Card(card).getCardEffect() in DistortApplicapableEffects):
                  mask[0, action] = False
          elif action < self.use_action_size + HAND_SIZE:
              if self.game.drawable_count <= 0:
                  mask[0, action] = False
          elif action == self.use_action_size + HAND_SIZE:
              pass
          else:
              raise ValueError("Invalid action: {}".format(action))
      else:
        mask = np.zeros((1, self.action_space.n), dtype=bool)
        mask[0, self.use_action_size + HAND_SIZE] = True
      return mask

    def step(self, action):
        reward = 0
        prev_normal_block_count = self.game.normal_block_count
        # parse action
        # type 1: use card
        if action < self.use_action_size:
            slot = action // self.board_entry_size
            use_action = action % self.board_entry_size
            x = use_action // MAX_BOARD_SIZE
            y = use_action % MAX_BOARD_SIZE
            self.game.useCard(slot, x, y)
            self._set_special_block()
            normal_block_count_diff = self.game.normal_block_count - prev_normal_block_count
            reward += -self.DEFAULT_GOLD_BLOCK * normal_block_count_diff / (self.game.remaining_step+1) * 10
        # type 2: replace card, it's ensured that drawable_count > 0
        elif self.use_action_size <= action < self.use_action_size + HAND_SIZE:
            self.game.replaceCard(action - self.use_action_size)
        # type 3: reset and start new one
        elif action == self.use_action_size + HAND_SIZE:
            self.game.restore()
        else:
            raise ValueError("Invalid action: {}".format(action))
        self._draw_card()
        observation = self._get_obs()
        terminated = self.game.is_terminated
        if terminated:
            reward = self.DEFAULT_GOLD_WIN if self.game.remaining_step >= 0 else 0
        info = self._get_info()
        return observation, reward, terminated, False, info

    def render(self):
        pass
