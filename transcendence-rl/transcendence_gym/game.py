from transcendence_gym.board import Board
from transcendence_gym.cardqueue import CardQueue
from transcendence_gym.constants import CardType, BASE_CARD_TYPE_SIZE, MAX_CARD_STRENGTH, SpecialBlockType, MAX_BOARD_SIZE
import json
import numpy as np

class Game(object):
  with open('transcendence_gym/presets.json') as f:
    PRESETS = json.load(f)
  def __init__(self, equipment_type, stage):
    self.equipment_type = equipment_type
    self.stage = stage
    self.trial_count = -1
    self.restore()
    self.rng = np.random.default_rng()

  @property
  def protection_level(self):
      if 1 <= self.stage <= 3:
          protection_level = (self.trial_count - 2)//2
      elif 4 <= self.stage <= 5:
          protection_level = (self.trial_count - 3)//2
      elif 6 <= self.stage <= 7:
          protection_level = (self.trial_count - 4)//2
      return min(protection_level, 10) if protection_level > 0 else 0  

  def restore(self):
    presets = Game.PRESETS
    self.board = Board(np.array(presets["boards"][self.equipment_type][str(self.stage)], dtype=np.uint8))
    self.remaining_step = presets["available_steps"][self.equipment_type][str(self.stage)]
    self.board.special_block = np.zeros(3, dtype=np.uint8)
    self.card_queue = CardQueue()
    self.trial_count += 1
    self.drawable_count = 2 + self.protection_level
  
  @property
  def isTerminated(self):
    return np.all(self.board.values < 3)
  
  def fillCard(self, card_num):
    self.card_queue.replaceCard(card_num)
  
  def useCard(self, idx, target_x, target_y):
    card = self.card_queue.popCardFromHand(idx)
    block_effect = self.board.applyEffect(card.getCardEffect(), target_x, target_y)
    self.remaining_step -= 1
    self.applyBlockEffect(block_effect, card, idx)
  
  def replaceCard(self, idx):
    self.card_queue.popCardFromHand(idx)
    self.drawable_count -= 1

  def applyBlockEffect(self, block_effect, card, used_card_idx):
    target_card_idx = 0 if used_card_idx == 1 else 0
    # handle special block effect
    # 강화
    if block_effect == SpecialBlockType.ENHANCE:
      if self.card_queue.hand[target_card_idx] // BASE_CARD_TYPE_SIZE < MAX_CARD_STRENGTH:
        self.card_queue.hand[target_card_idx] += BASE_CARD_TYPE_SIZE
    # 복제
    elif block_effect == SpecialBlockType.DUPLICATE:
      self.card_queue.setHand(target_card_idx, card.card_num)
      self.card_queue.mergeCard()
    # 신비
    elif block_effect == SpecialBlockType.MYSTIC:
      new_card_num = self.rng.choice([CardType.EXPULSION, CardType.RESONANCE_OF_WORLD_TREE])
      self.card_queue.setHand(target_card_idx, new_card_num)
    # 추가
    elif block_effect == SpecialBlockType.ADDITION:
      self.drawable_count += 1
    # 재배치
    elif block_effect == SpecialBlockType.REARRANGE:
      fb = self.board.values.flatten()
      idx, = np.nonzero(fb)
      fb[idx] = fb[self.rng.permutation(idx)]
      self.board.values = fb.reshape((MAX_BOARD_SIZE, MAX_BOARD_SIZE))
    # 축복
    elif block_effect == SpecialBlockType.BLESS:
      self.remaining_step += 1
    