import numpy as np
from transcendence_gym.constants import BaseBlockType, CardType, MAX_CARD_STRENGTH
from transcendence_gym.board import Board
from functools import partial

def applyLightning(board:Board, x, y, strength):
    damaged_block_effect = board.removeTargetBlock(x, y)
    relative_pos = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    p = 1.0 if strength > 0 else 0.50
    distort_mode = "ignore" if strength == 2 else "default"
    for pos in relative_pos:
        new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
        assert not (damaged_block_effect > 0 and new_block_effect > 0)
        damaged_block_effect = max(damaged_block_effect, new_block_effect)
    return damaged_block_effect

def applyRisingFire(board:Board, x, y, strength):
    damaged_block_effect = board.removeTargetBlock(x, y)
    relative_pos = [(-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)]
    p = 1.0 if strength > 0 else 0.50
    distort_mode = "ignore" if strength == 2 else "default"
    for pos in relative_pos:
        new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
        assert not (damaged_block_effect > 0 and new_block_effect > 0)
        damaged_block_effect = max(damaged_block_effect, new_block_effect)
    return damaged_block_effect

def applyShockwave(board:Board, x, y, strength):
    # 충격파: Removes blocks in a small radius around the target
    damaged_block_effect = board.removeTargetBlock(x, y)
    relative_pos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    p = 1.0 if strength > 0 else 0.75
    distort_mode = "ignore" if strength == 2 else "default"
    for pos in relative_pos:
        new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
        assert not (damaged_block_effect > 0 and new_block_effect > 0)
        damaged_block_effect = max(damaged_block_effect, new_block_effect)
    return damaged_block_effect

def applyTsunami(board:Board, x, y, strength):
    # 해일: Removes blocks in concentric circles, with the probability decreasing the further away from the target
    damaged_block_effect = board.removeTargetBlock(x, y)
    p = 0.85
    delta = 0.15
    depth = 1
    distort_mode = "ignore" if strength == 2 else "default"
    while p > 0:
        relative_pos = [(-depth, 0), (0, -depth), (0, depth), (depth, 0)]
        for pos in relative_pos:
            new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
            assert not (damaged_block_effect > 0 and new_block_effect > 0)
            damaged_block_effect = max(damaged_block_effect, new_block_effect)
        p -= delta
        depth += 1
    return damaged_block_effect

def applyMassiveExplosion(board:Board, x, y, strength):
    # 대폭발: Removes blocks in a diagonal pattern extending from the target
    damaged_block_effect = board.removeTargetBlock(x, y)
    p = 0.85
    delta = 0.15
    depth = 1
    distort_mode = "ignore" if strength == 2 else "default"
    while p > 0:
        relative_pos = [(-depth, -depth), (-depth, depth), (depth, -depth), (depth, depth)]
        for pos in relative_pos:
            new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
            assert not (damaged_block_effect > 0 and new_block_effect > 0)
            damaged_block_effect = max(damaged_block_effect, new_block_effect)
        p -= delta
        depth += 1
    return damaged_block_effect

def applyStorm(board:Board, x, y, strength):
    # 폭풍우: Removes blocks in vertical lines extending from the target
    damaged_block_effect = board.removeTargetBlock(x, y)
    p = 0.85
    delta = 0.15
    depth = 1
    distort_mode = "ignore" if strength == 2 else "default"
    while p > 0:
        relative_pos = [(-depth, 0), (depth, 0)]
        for pos in relative_pos:
            new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
            assert not (damaged_block_effect > 0 and new_block_effect > 0)
            damaged_block_effect = max(damaged_block_effect, new_block_effect)
        p -= delta
        depth += 1
    return damaged_block_effect

def applyThunderbolt(board:Board, x, y, strength):
    # 벼락: Damages up to a random number of blocks, or adds normal blocks
    damaged_block_effect = board.removeTargetBlock(x, y)
    rng = np.random.default_rng()
    n = rng.choice(range(-1, 3 + 2 * strength))
    if n < 0:
        positions = np.transpose(np.nonzero(board.values == BaseBlockType.DESTROYED))
        new_blocks_pos = rng.choice(positions, size=min(1, len(positions)), replace=False)
        for pos in new_blocks_pos:
            board[pos[0], pos[1]] = BaseBlockType.NORMAL
    else:
        positions = np.transpose(np.nonzero(board.values == BaseBlockType.NORMAL))
        new_blocks_pos = rng.choice(positions, size=min(n, len(positions)), replace=False)
        for pos in new_blocks_pos:
            new_block_effect = board.removeTargetBlock(pos[0], pos[1])
            assert not (damaged_block_effect > 0 and new_block_effect > 0)
            damaged_block_effect = max(damaged_block_effect, new_block_effect)
    return damaged_block_effect

def applyEarthquake(board:Board, x, y, strength):
    # 지진: Removes blocks in horizontal lines extending from the target
    damaged_block_effect = board.removeTargetBlock(x, y)
    p = 0.85
    delta = 0.15
    depth = 1
    distort_mode = "ignore" if strength == 2 else "default"
    while p > 0:
        relative_pos = [(0, -depth), (0, depth)]
        for pos in relative_pos:
            new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
            assert not (damaged_block_effect > 0 and new_block_effect > 0)
            damaged_block_effect = max(damaged_block_effect, new_block_effect)
        p -= delta
        depth += 1
    return damaged_block_effect

def applyPurification(board:Board, x, y, strength):
    # 정화: Can target distorted blocks directly and removes blocks horizontally adjacent to the target
    damaged_block_effect = board.removeTargetBlock(x, y, 1.0, "break")
    p = 0.5 if strength == 0 else 1.0
    distort_mode = "break"
    if strength == 0 or strength == 1:
        relative_pos = [(0, -1), (0, 1)]
        for pos in relative_pos:
            new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
            assert not (damaged_block_effect > 0 and new_block_effect > 0)
            damaged_block_effect = max(damaged_block_effect, new_block_effect)
    elif strength == 2:
        relative_pos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for pos in relative_pos:
            new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
            assert not (damaged_block_effect > 0 and new_block_effect > 0)
            damaged_block_effect = max(damaged_block_effect, new_block_effect)
    return damaged_block_effect

def applyTornado(board:Board, x, y, strength):
    # 용오름: Removes blocks diagonally adjacent to the target
    damaged_block_effect = board.removeTargetBlock(x, y)
    relative_pos = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    p = 1.0 if strength > 0 else 0.50
    distort_mode = "ignore" if strength == 2 else "default"
    for pos in relative_pos:
        new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
        assert not (damaged_block_effect > 0 and new_block_effect > 0)
        damaged_block_effect = max(damaged_block_effect, new_block_effect)
    return damaged_block_effect

def applyResonanceOfWorldTree(board:Board, x, y, strength):
    # 세계수의 공명: Can target distorted blocks directly and removes 2 blocks in the cardinal directions (up, down, left, right)
    damaged_block_effect = board.removeTargetBlock(x, y, 1.0, "break")
    p = 1.0
    distort_mode = "break"
    relative_pos = [(-2, 0), (-1, 0), (1, 0), (2, 0), (0, -2), (0, -1), (0, 1), (0, 2)]
    for pos in relative_pos:
        new_block_effect = board.removeTargetBlock(x + pos[0], y + pos[1], p, distort_mode)
        assert not (damaged_block_effect > 0 and new_block_effect > 0)
        damaged_block_effect = max(damaged_block_effect, new_block_effect)
    return damaged_block_effect

def getCardEffect(card_type:CardType, strength:int):
  # Dictionary mapping card numbers to their effect functions
  card_effects = {
      CardType.LIGHTNING: applyLightning,
      CardType.RISING_FIRE: applyRisingFire,
      CardType.SHOCKWAVE: applyShockwave,
      CardType.TSUNAMI: applyTsunami,
      CardType.MASSIVE_EXPLOSION: applyMassiveExplosion,
      CardType.STORM: applyStorm,
      CardType.THUNDERBOLT: applyThunderbolt,
      CardType.EARTHQUAKE: applyEarthquake,
      CardType.PURIFICATION: applyPurification,
      CardType.TORNADO: applyTornado,
      CardType.RESONANCE_OF_WORLD_TREE: applyResonanceOfWorldTree
  }
  assert strength <= MAX_CARD_STRENGTH
  try:
    effect_function = partial(card_effects[card_type], strength=strength)
    effect_function.__name__ = card_effects[card_type]
    return effect_function
  except KeyError:
    raise ValueError(f"Invalid card type: {card_type}. Must be one of {CardType}")