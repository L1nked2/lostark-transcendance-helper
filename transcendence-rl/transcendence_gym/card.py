from transcendence_gym.constants import CardType, BASE_CARD_TYPE_SIZE, MAX_CARD_STRENGTH
from transcendence_gym.effects import getCardEffect

class Card(object):
    def __init__(self, card_num):
        self.card_num = card_num
        if card_num // BASE_CARD_TYPE_SIZE > MAX_CARD_STRENGTH:
            self.strength = 0
            self.type = card_num
        else:
          self.strength = card_num // BASE_CARD_TYPE_SIZE
          self.type = card_num % BASE_CARD_TYPE_SIZE

        if not self.type in CardType:
            raise ValueError(f"Invalid card type: {self.type}. Must be one of {CardType}")

        # Get the card effect
        self.effect = getCardEffect(self.type, self.strength)
        
    def getCardEffect(self):
        return self.effect


