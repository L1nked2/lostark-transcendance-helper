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
from transcendence_gym.constants import BlockType, DistortApplicapableEffects

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='transcendence-sim-v0',
    entry_point='transcendence_gym.transcendence_gym:TranscendenceEnv',
)



class TranscendenceEnv(gym.Env):
    CARD_TYPES = 32
    BOARD_SIZE = 8
    DEFAULT_BLOCK_TYPES = 4
    SPECIAL_BLOCK_TYPES = 10
    CARD_PROBABILITY_TABLE = [0.150, 0.115, 0.095, 0.055, 0.105, 0.070, 0.090, 0.070, 0.100, 0.150]
    BLOCK_PROBABILITY_TABLE = [0.160, 0.160, 0.160, 0.235, 0.170, 0.115]
    with open('transcendence_gym/presets.json') as f:
        PRESETS = json.load(f)

    @property
    def protection_level(self):
        if 1 <= self.stage <= 3:
            protection_level = (self.trial_count - 2)//2
        elif 4 <= self.stage <= 5:
            protection_level = (self.trial_count - 3)//2
        elif 6 <= self.stage <= 7:
            protection_level = (self.trial_count - 4)//2
        return protection_level if protection_level > 0 else 0
    
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
        CARD_TYPES = TranscendenceEnv.CARD_TYPES
        BOARD_SIZE = TranscendenceEnv.BOARD_SIZE
        DEFAULT_BLOCK_TYPES = TranscendenceEnv.DEFAULT_BLOCK_TYPES
        SPECIAL_BLOCK_TYPES = TranscendenceEnv.SPECIAL_BLOCK_TYPES
        self.observation_space = spaces.Dict({"hand":spaces.MultiDiscrete([CARD_TYPES, CARD_TYPES], dtype=np.uint8), 
                                              "queue":spaces.MultiDiscrete([CARD_TYPES for _ in range(3)], dtype=np.uint8), 
                                              "board":spaces.MultiDiscrete(np.full((BOARD_SIZE, BOARD_SIZE), DEFAULT_BLOCK_TYPES), dtype=np.uint8),
                                              "special_block":spaces.MultiDiscrete((BOARD_SIZE, BOARD_SIZE, SPECIAL_BLOCK_TYPES), dtype=np.uint8),
                                              "remaining_step":spaces.Discrete(12), 
                                              "drawable_count":spaces.Discrete(12),
                                              "protection_level":spaces.Discrete(11)})
        

    def _get_obs(self):
        observation = {
            "hand": np.array(self.hand, dtype=np.uint8),
            "queue": np.array(self.queue, dtype=np.uint8),
            "board": self.board.values,
            "special_block": self.board.special_block,
            "remaining_step": self.remaining_step,
            "drawable_count": self.drawable_count,
            "protection_level": self.protection_level
        }
        return observation

    def _get_info(self):
        block_unique, block_counts = np.unique(self.board, return_counts=True)
        remaining_blocks = dict(zip(block_unique, block_counts))
        return {"remaining_blocks": remaining_blocks}
    
    def _load_board_and_step(self, equipment_type_idx, level):
        presets = TranscendenceEnv.PRESETS
        equipment_types = list(presets["boards"].keys())
        equipment_type = equipment_types[equipment_type_idx]
        self.board = Board(np.array(presets["boards"][equipment_type][str(level)], dtype=np.uint8))
        self.remaining_step = presets["available_steps"][equipment_type][str(level)]
    
    def _restore_board_and_hand(self, case_num):
        # restore board
        self.board = Board(np.zeros((8, 8), dtype=np.uint8))
        self.board.special_block = np.zeros(3, dtype=np.uint8)
        if case_num is None:
            case_num = self.rng.choice(5*7)
        self._load_board_and_step(case_num//7, case_num%7+1)
        # fill hand and queue
        self.hand = deque(maxlen=2)
        self.queue = deque(maxlen=3)
        self._fill_hand()
        # apply protection level
        self.drawable_count = 2 + self.protection_level
        

    def _fill_queue(self):
        # loop until queue is filled
        while len(self.queue) < 3:
            # draw card
            new_card = self.rng.choice(10, p=TranscendenceEnv.CARD_PROBABILITY_TABLE)
            self.queue.appendleft(new_card)

    def _fill_hand(self):
        self._fill_queue()
        # loop until hand is filled
        while len(self.hand) < 2:
            # fill empty hand
            self.hand.append(self.queue.pop())        
            # merge card if possible
            if len(self.hand) == 2 and self.hand[0] % 10 == self.hand[1] % 10 and max(self.hand[0], self.hand[1]) < 30:
                if self.hand[0] >= self.hand[1]:
                    self.hand[0] += 10
                    self.hand.pop()
                else:
                    self.hand[1] += 10
                    self.hand.popleft()
            # fill queue again
            self._fill_queue()

    def _use_card(self, slot, target_x, target_y):
        assert slot == 0 or slot == 1
        # use card to board
        target_card_num = self.hand.popleft() if slot == 0 else self.hand.pop()
        card = Card(target_card_num)
        broken_block_effect = self.board.applyEffect(card.getCardEffect(), target_x, target_y)
        self.remaining_step -= 1

        # handle special block effect
        # 4: 강화
        if broken_block_effect == 4:
          if self.hand[0] < 20:
            self.hand[0] += 10
        # 5: 복제
        elif broken_block_effect == 5:
          self.hand[0] = target_card_num
        # 6: 신비
        elif broken_block_effect == 6:
          new_card = self.rng.choice(2)
          if new_card == 0:
            self.hand[0] = 31
          else:
            self.hand[0] = 32
        # 7: 추가
        elif broken_block_effect == 7:
          self.drawable_count += 1
        # 8: 재배치
        elif broken_block_effect == 8:
          fb = self.board.flatten()
          idx, = np.nonzero(fb)
          fb[idx] = fb[self.rng.permutation(idx)]
          self.board = fb.reshape((8, 8))
        # 9: 축복
        elif broken_block_effect == 9:
          self.remaining_step += 1
        
        # clear prev special block and add new special block
        self.board.special_block = np.array([0, 0, 0], dtype=np.uint8)
        candidates = np.transpose(np.where(self.board.values == 3))
        if len(candidates) > 0:
          sp = self.rng.choice(candidates)
          self.board.special_block[0:2] = sp
          self.board.special_block[2] = self.rng.choice(range(4, 10), p=TranscendenceEnv.BLOCK_PROBABILITY_TABLE)

        # draw new card
        self._fill_hand()

    def _replace_card(self, slot):
        assert slot == 0 or slot == 1
        assert self.drawable_count > 0
        # change card
        self.drawable_count -= 1
        if slot == 0:
            self.hand.popleft()
        else:
            self.hand.pop()
        # draw new card
        self._fill_hand()

    def reset(self, seed=None, options=None):
        assert options and "case_num" in options
        case_num = options["case_num"]
        assert case_num is None or 0 <= case_num < 5 * 7
        self.rng = np.random.default_rng(seed=seed)
        self.stage = case_num%7+1
        self.trial_count = 0
        self._restore_board_and_hand(case_num)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.hand_backup = self.hand.copy()
        try:
          # parse action
          # type 1: use card
          if action < 128:
              if action < 64:
                  self._use_card(0, action//8, action%8)
              else:
                  self._use_card(1, (action-64)//8, (action-64)%8)
          # type 2: replace card, it's ensured that drawable_count > 0
          elif action == 129 or action == 130:
              self._replace_card(action - 129)
          elif action == 131:
              # reset and start new one
              raise NotImplementedError
        except:
            self.hand = self.hand_backup
        observation = self._get_obs()
        terminated = np.all(self.board.values < 3)
        reward = 1 if terminated else 0
        info = self._get_info()
        return observation, reward, terminated, False, info

    def get_action_mask(self, observation):
        mask = np.ones(self.action_space.n)
        hand_l, hand_r = observation["hand"][0], observation["hand"][1]
        for i in range(64):
            if observation["board"][i//8, i%8] < 2:
                mask[i] = 0
                mask[i+64] = 0
            elif observation["board"][i//8, i%8] == 2:
              if not (Card(hand_l).getCardEffect() in DistortApplicapableEffects):
                mask[i] = 0
              if not (Card(hand_r).getCardEffect() in DistortApplicapableEffects):
                mask[i+64] = 0
        if self.drawable_count == 0:
            mask[128] = 0
            mask[129] = 0
        # to be implemented
        mask[130] = 0

        return mask

    def render(self):
        pass
