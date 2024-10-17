import numpy as np
import pytest
from unittest.mock import patch, Mock
from transcendence_gym.effects import (
    applyLightning, applyRisingFire,
    applyShockwave, applyTsunami, applyMassiveExplosion, applyStorm,
    applyThunderbolt, applyEarthquake, applyPurification, applyTornado,
    applyResonanceOfWorldTree
)
from transcendence_gym.board import Board
from transcendence_gym.card import Card

from transcendence_gym.constants import (
  BaseBlockType, MAX_BOARD_SIZE
)

@pytest.fixture
def board():
    board_test = Board(np.full((MAX_BOARD_SIZE, MAX_BOARD_SIZE), BaseBlockType.NORMAL))
    board_test._special_block = np.asarray((4, 4, 9))
    return board_test

def test_removeBlock(board:Board):
    with patch('numpy.random.default_rng') as mock_rng:
        # Test removing a normal block
        mock_rng.return_value.random.return_value = 0.5  # Below default prob of 1.0, always true
        x, y = board._removeBlock(0, 0)
        assert x == 0 and y == 0
        assert board[0, 0] == BaseBlockType.DESTROYED

        # Test removing an already destroyed block
        board[2, 2] = BaseBlockType.DESTROYED
        x, y = board._removeBlock(2, 2)
        assert x == -1 and y == -1
        assert board[2, 2] == BaseBlockType.DESTROYED

        # Test removing an empty block
        board[3, 3] = BaseBlockType.EMPTY
        x, y = board._removeBlock(3, 3)
        assert x == -1 and y == -1
        assert board[3, 3] == BaseBlockType.EMPTY

        # Test removing a distorted block with different distort modes
        board[3, 3] = BaseBlockType.DESTROYED
        board[5, 5] = BaseBlockType.DISTORTED
        
        # Default mode
        assert np.sum(board.values[board.values == BaseBlockType.DESTROYED]) == 3  # 3 blocks should be restored
        mock_rng.return_value.choice.return_value = np.array([(0, 0), (2, 2), (3, 3)])
        x, y = board._removeBlock(5, 5)
        assert x == -1 and y == -1
        assert board[5, 5] == BaseBlockType.DISTORTED  # Should not change
        assert np.sum(board.values == BaseBlockType.NORMAL) == 63  # all blocks are normal except the distorted one
        
        # Break mode
        x, y = board._removeBlock(5, 5, distort_mode="break")
        assert x == 5 and y == 5
        assert board[5, 5] == BaseBlockType.DESTROYED
        
        # Ignore mode
        board[5, 5] = BaseBlockType.DISTORTED
        x, y = board._removeBlock(5, 5, distort_mode="ignore")
        assert x == -1 and y == -1
        assert board[5, 5] == BaseBlockType.DISTORTED  # Should not change

        # Test out of bounds
        x, y = board._removeBlock(10, 10)
        assert x == -1 and y == -1

        # Test invalid distort mode
        with pytest.raises(ValueError):
            board._removeBlock(0, 0, distort_mode="invalid")

def test_applyLightning(board):
    with patch('numpy.random.default_rng') as mock_rng:
        # Test with RNG value 0.4
        mock_rng.return_value.random.return_value = 0.4
        board_temp = board.copy()
        effect = applyLightning(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 5
        assert effect == 0

        # Test with RNG value 0.8
        mock_rng.return_value.random.return_value = 0.8
        board_temp = board.copy()
        effect = applyLightning(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 1
        assert effect == 0

        # Test destroying special block
        board_temp = board.copy()
        effect = applyLightning(board_temp, 4, 4, 0)
        assert effect == 9

def test_applyRisingFire(board):
    with patch('numpy.random.default_rng') as mock_rng:
        # Test with RNG value 0.4
        mock_rng.return_value.random.return_value = 0.4
        board_temp = board.copy()
        effect = applyRisingFire(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 13  # Center + 12 adjacent
        assert effect == 9

        # Test with RNG value 0.8
        mock_rng.return_value.random.return_value = 0.8
        board_temp = board.copy()
        effect = applyRisingFire(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 1
        assert effect == 0

def test_applyShockwave(board):
    with patch('numpy.random.default_rng') as mock_rng:
        # Test with RNG value 0.4
        mock_rng.return_value.random.return_value = 0.4
        board_temp = board.copy()
        effect = applyShockwave(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 9  # Center + 8 adjacent
        assert effect == 9

        # Test with RNG value 0.8
        mock_rng.return_value.random.return_value = 0.8
        board_temp = board.copy()
        effect = applyShockwave(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 1
        assert effect == 0

def test_applyTsunami(board):
    with patch('numpy.random.default_rng') as mock_rng:
        # Test with RNG value 0.4
        mock_rng.return_value.random.return_value = 0.4
        board_temp = board.copy()
        effect = applyTsunami(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 13
        assert effect == 0

        # Test with RNG value 0.8
        mock_rng.return_value.random.return_value = 0.8
        board_temp = board.copy()
        effect = applyTsunami(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 5
        assert effect == 0

def test_applyMassiveExplosion(board):
    with patch('numpy.random.default_rng') as mock_rng:
        # Test with RNG value 0.4
        mock_rng.return_value.random.return_value = 0.4
        board_temp = board.copy()
        effect = applyMassiveExplosion(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 13
        assert effect == 9 

        # Test with RNG value 0.8
        mock_rng.return_value.random.return_value = 0.8
        board_temp = board.copy()
        effect = applyMassiveExplosion(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 5
        assert effect == 9

def test_applyStorm(board):
    with patch('numpy.random.default_rng') as mock_rng:
        # Test with RNG value 0.4
        mock_rng.return_value.random.return_value = 0.4
        board_temp = board.copy()
        effect = applyStorm(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 7
        assert effect == 0 

        # Test with RNG value 0.8
        mock_rng.return_value.random.return_value = 0.8
        board_temp = board.copy()
        effect = applyStorm(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 3
        assert effect == 0

def test_applyThunderbolt_add_block(board):
    with patch('numpy.random.default_rng') as mock_rng:
        mock_generator = mock_rng.return_value
        mock_generator.random.return_value = 0.4

        # New block is added on (5, 5)
        mock_generator.choice.side_effect = [
            -1,
            np.array([(5, 5)]),
        ]
        board_temp = board.copy()
        board_temp[5, 5] = BaseBlockType.DESTROYED
        effect = applyThunderbolt(board_temp, 3, 3, 0)
        assert board_temp[5, 5] == BaseBlockType.NORMAL
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 1
        assert effect == 0

def test_applyThunderbolt_destroy_block(board):
    with patch('numpy.random.default_rng') as mock_rng:
        mock_generator = mock_rng.return_value
        mock_generator.random.return_value = 0.4

        # Destroy two more blocks
        mock_generator.choice.side_effect = [
            2,
            np.array([(4, 4), (5, 5)]),
        ]
        board_temp = board.copy()
        effect = applyThunderbolt(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 3
        assert effect == 9


def test_applyEarthquake(board):
    with patch('numpy.random.default_rng') as mock_rng:
        # Test with RNG value 0.4
        mock_rng.return_value.random.return_value = 0.4
        board_temp = board.copy()
        effect = applyEarthquake(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 7
        assert effect == 0

        # Test with RNG value 0.8
        mock_rng.return_value.random.return_value = 0.8
        board_temp = board.copy()
        effect = applyEarthquake(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 3
        assert effect == 0

def test_applyPurification(board):
    with patch('numpy.random.default_rng') as mock_rng:
        board[3, 3] = BaseBlockType.DISTORTED
        
        # Test with RNG value 0.4
        mock_rng.return_value.random.return_value = 0.4
        board_temp = board.copy()
        effect = applyPurification(board_temp, 3, 3, 0)
        assert board_temp[3, 3] == BaseBlockType.DESTROYED
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 3  # Center + 2 adjacent
        assert effect == 0

        # Test with RNG value 0.8
        mock_rng.return_value.random.return_value = 0.8
        board_temp = board.copy()
        effect = applyPurification(board_temp, 3, 3, 0)
        assert board_temp[3, 3] == BaseBlockType.DESTROYED
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 1
        assert effect == 0

        # Test strength 2
        mock_rng.return_value.random.return_value = 0.8
        board_temp = board.copy()
        effect = applyPurification(board_temp, 3, 3, 2)
        assert board_temp[3, 3] == BaseBlockType.DESTROYED
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 5  # Center + 4 adjacent
        assert effect == 0

def test_applyTornado(board):
    with patch('numpy.random.default_rng') as mock_rng:
        # Test with RNG value 0.4
        mock_rng.return_value.random.return_value = 0.4
        board_temp = board.copy()
        effect = applyTornado(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 5  # Center + 4 diagonal
        assert effect == 9

        # Test with RNG value 0.8
        mock_rng.return_value.random.return_value = 0.8
        board_temp = board.copy()
        effect = applyTornado(board_temp, 3, 3, 0)
        assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 1
        assert effect == 0

def test_applyResonanceOfWorldTree(board):
    board_temp = board.copy()
    board_temp[3, 3] = BaseBlockType.DISTORTED
    effect = applyResonanceOfWorldTree(board_temp, 3, 3, 0)
    assert board_temp[3, 3] == BaseBlockType.DESTROYED
    assert np.sum(board_temp.values[board_temp.values == BaseBlockType.DESTROYED]) == 9
    assert effect == 0

def test_removeTargetBlock_success(board):
    with patch('numpy.random.default_rng') as mock_rng:
      # Test with RNG value 0.4
      mock_rng.return_value.random.return_value = 0.4
      board_temp = board.copy()
      board_temp._removeBlock = Mock(return_value=(5, 5))

      # Special block should match and return the special block effect
      result = board_temp.removeTargetBlock(5, 5, prob=0.5)
      assert result == 0
      board_temp._removeBlock.assert_called_once_with(5, 5, "default")

      # Test with RNG value 0.8
      mock_rng.return_value.random.return_value = 0.8
      board_temp = board.copy()
      board_temp._removeBlock = Mock(return_value=(5, 5))

      # Special block should match and return the special block effect
      result = board_temp.removeTargetBlock(5, 5, prob=0.5)
      assert result == 0
      assert not board_temp._removeBlock.called

def test_applyEffect_out_of_board(board):
    """# Effect should raise an error if x, y are out of bounds
    with pytest.raises(ValueError, match="Out of board"):
        board.applyEffect(lambda board, x, y: None, 11, 0)"""
    pass

def test_applyEffect_no_block(board):
    """# Effect should raise an error if the block is EMPTY or DESTROYED
    game_board.values[1][1] = BlockType.EMPTY
    with pytest.raises(ValueError, match="No block"):
        board.applyEffect(lambda board, x, y: None, 1, 1)"""
    pass

def test_applyEffect_distorted_block_invalid_effect(board):
    # Effect should raise an error if the block is DISTORTED and the effect is not applicable
    board_temp = board.copy()
    board_temp[1, 1] = BaseBlockType.DISTORTED
    with pytest.raises(ValueError, match="Can't target distorted block"):
        board_temp.applyEffect(lambda board, x, y: None, 1, 1)

def test_applyEffect_valid_effect(board):
    """# Effect should be applied without errors when valid
    board.values[1][1] = BlockType.DISTORTED
    def valid_effect(board, x, y):
        board.values[x][y] = BlockType.DESTROYED
    
    valid_effect.__name__ = "effect_name"  # Mark effect as applicable
    
    game_board.applyEffect(valid_effect, 1, 1)
    assert board.values[1][1] == BlockType.DESTROYED"""
    pass


### test about target on distort(applyEffect)