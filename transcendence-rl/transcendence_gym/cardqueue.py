from collections import deque
from transcendence_gym.constants import BASE_CARD_TYPE_SIZE, MAX_CARD_STRENGTH, HAND_SIZE, QUEUE_SIZE
from transcendence_gym.card import Card
from itertools import chain
class CardQueue(object):
  def __init__(self):
    self.hand = deque(maxlen=HAND_SIZE)
    self.queue = deque(maxlen=QUEUE_SIZE)
    while len(self.hand) < HAND_SIZE:
      self.hand.append(-1)
    while len(self.queue) < QUEUE_SIZE:
      self.queue.append(-1)
  
  def isValid(self):
    for card_num in chain(self.hand, self.queue):
      if card_num < 0:
        return False
    return True

  def mergeCard(self):
    assert self.hand[0] >= 0 and self.hand[1] >= 0
    while self.hand[0] % BASE_CARD_TYPE_SIZE == self.hand[1] % BASE_CARD_TYPE_SIZE \
          and max(self.hand[0], self.hand[1]) < BASE_CARD_TYPE_SIZE * MAX_CARD_STRENGTH:
        if self.hand[0] < self.hand[1]:
            self.hand[0], self.hand[1] = self.hand[1], self.hand[0]
        self.hand[0] += 10
        self.hand[1] = self.queue.pop()
        self.queue.appendleft(-1)

  def popCardFromHand(self, idx):
    assert 0 <= idx < HAND_SIZE
    assert self.isValid()
    # pop card and draw new card from queue
    card_num = self.hand[idx]
    self.hand[idx] = self.queue.pop()
    self.queue.appendleft(-1)
    # merge card if possible
    self.mergeCard()
    return Card(card_num)
  
  def setHand(self, idx, card_num):
    assert 0 <= idx < HAND_SIZE
    self.hand[idx] = card_num
  
  def setQueue(self, idx, card_num):
    assert 0 <= idx < QUEUE_SIZE
    self.queue[idx] = card_num

  # set valid card on appropriate position, priority: hand > queue
  def replaceCard(self, card_num):
    for idx in range(len(self.hand)-1, -1, -1):
      if self.hand[idx] == -1:
        self.hand[idx] = card_num
        if self.hand[0] > 0 and self.hand[1] > 0:
          self.mergeCard()
        return
    for idx in range(len(self.queue)-1, -1, -1):
      if self.queue[idx] == -1:
        self.queue[idx] = card_num
        return