import numpy as np
from transcendence_gym.constants import (
  MAX_BOARD_SIZE, DEFAULT_DISTORT_SIDE_EFFECT_BLOCK_NUM, 
  DistortModeType, BaseBlockType, DistortApplicapableEffects
)

class Board(object):
  def __init__(self, board_values:np.ndarray=None):
    self._special_block = np.zeros(3, dtype=np.uint8)
    self.rng = np.random.default_rng()
    if not (board_values is None):
      self.size = len(board_values)
      ###assert size, type
      self.values = board_values
    else:
       self.size = MAX_BOARD_SIZE
       self.values = np.full((MAX_BOARD_SIZE, MAX_BOARD_SIZE), BaseBlockType.NORMAL)
  
  def __getitem__(self, item):
    if isinstance(item, tuple):
      return self.values[item]
  
  def __setitem__(self, item, value):
    if isinstance(item, tuple):
      self.values[item] = value

  def copy(self):
     copied = Board(self.values.copy())
     copied.special_block = self.special_block
     return copied
  
  @property
  def special_block(self):
     return self._special_block
  
  @special_block.setter
  def special_block(self, value):
     ###assert size, type
    self._special_block = value
  
  def _removeBlock(self, x, y, distort_mode:str = "default"):
    # Check if the distort_mode is valid
    if distort_mode not in DistortModeType:
        raise ValueError(f"Invalid distort mode: {distort_mode}. Must be one of {DistortModeType._member_map_.keys()}")

    # Check if the block is valid
    if not (0 <= x < len(self.values) and 0 <= y < len(self.values[0])):
        return -1, -1
    if self.values[x][y] == BaseBlockType.EMPTY or self.values[x][y] == BaseBlockType.DESTROYED:
        return -1, -1

    # Try to remove the block
    if self.values[x][y] == BaseBlockType.DISTORTED:
        if distort_mode == DistortModeType.BREAK:
            self.values[x][y] = BaseBlockType.DESTROYED
            return x, y
        elif distort_mode == DistortModeType.IGNORE:
            pass
        elif distort_mode == DistortModeType.DEFAULT:
            # distort block damaged, restore destroyed blocks
            positions = np.transpose(np.nonzero(self.values == BaseBlockType.DESTROYED))
            new_blocks_pos = self.rng.choice(positions, size=min(DEFAULT_DISTORT_SIDE_EFFECT_BLOCK_NUM, len(positions)), replace=False, shuffle=False)
            for pos in new_blocks_pos:
                self.values[pos[0]][pos[1]] = BaseBlockType.NORMAL
    elif self.values[x][y] == BaseBlockType.NORMAL:
        self.values[x][y] = BaseBlockType.DESTROYED
        return x, y
    else:
        raise ValueError("Wrong block state given")

    # Block was not removed
    return -1, -1

  def removeTargetBlock(self, x, y, prob=1.0, distort_mode="default"):
    """Remove the target block and return the special block effect."""
    removed_x, removed_y = -1, -1
    if self.rng.random() < prob:
      removed_x, removed_y = self._removeBlock(x, y, distort_mode)
    damaged_block_effect = self.special_block[2] if (removed_x, removed_y) == tuple(self.special_block[:2]) else 0
    return damaged_block_effect

  def applyEffect(self, effect, x, y):
    """Apply the card effect to the game board."""
    if not (0 <= x < MAX_BOARD_SIZE and 0 <= y < MAX_BOARD_SIZE):
        raise ValueError("Out of board")
    if self.values[x][y] == BaseBlockType.EMPTY or self.values[x][y] == BaseBlockType.DESTROYED:
        raise ValueError("No block")
    distort_applicapable = True if effect.__name__ in DistortApplicapableEffects else False
    if self.values[x][y] == BaseBlockType.DISTORTED and not distort_applicapable:
        raise ValueError("Can't target distorted block")
    
    effect(self, x, y)