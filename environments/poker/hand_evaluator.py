from typing import List, Tuple
import numpy as np
from collections import Counter

class Card:
    """Represents a playing card"""
    RANKS = '23456789TJQKA'
    SUITS = 'CDHS'  # Clubs, Diamonds, Hearts, Spades
    
    def __init__(self, card_str: str):
        """Initialize from string like '2H' for 2 of Hearts"""
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}")
        rank, suit = card_str[0], card_str[1]
        if rank not in self.RANKS or suit not in self.SUITS:
            raise ValueError(f"Invalid card: {card_str}")
        self.rank = rank
        self.suit = suit
        self.rank_idx = self.RANKS.index(rank)
        self.suit_idx = self.SUITS.index(suit)
        
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        return self.__str__()
    
    def to_tensor_index(self) -> int:
        """Convert card to index in 52-card tensor (0-51)"""
        return self.rank_idx * 4 + self.suit_idx

class HandEvaluator:
    """
    Advanced poker hand evaluator using lookup tables for fast evaluation.
    Supports both 5-card and 7-card hand evaluation.
    """
    
    # Hand rankings from highest to lowest
    HAND_RANKS = [
        'Straight Flush',
        'Four of a Kind',
        'Full House',
        'Flush',
        'Straight',
        'Three of a Kind',
        'Two Pair',
        'One Pair',
        'High Card'
    ]
    
    def __init__(self):
        """Initialize lookup tables for fast hand evaluation"""
        # Pre-compute straight patterns
        self.straights = []
        for i in range(10):  # Include A-5 straight
            straight = [r for r in range(i, i+5)]
            if i == 0:  # Special case: Ace-low straight
                straight = [12] + [0,1,2,3]  # Ace can be low
            self.straights.append(set(straight))
    
    def evaluate_hand(self, cards: List[Card]) -> Tuple[int, str, List[Card]]:
        """
        Evaluate a poker hand (5 or 7 cards).
        Returns (hand_strength, hand_name, best_five)
        """
        if len(cards) not in [5, 7]:
            raise ValueError("Must provide 5 or 7 cards")
            
        # Get all possible 5-card combinations if 7 cards provided
        if len(cards) == 7:
            best_score = -1
            best_hand = None
            best_name = None
            for i in range(21):  # 21 possible 5-card combinations
                combo = self._get_5_card_combo(cards, i)
                score, name, hand = self._evaluate_5_card_hand(combo)
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_hand = hand
            return best_score, best_name, best_hand
        else:
            return self._evaluate_5_card_hand(cards)
    
    def _get_5_card_combo(self, cards: List[Card], index: int) -> List[Card]:
        """Get specific 5-card combination from 7 cards"""
        # Pre-computed indices for all 21 possible 5-card combinations
        combos = [
            [0,1,2,3,4], [0,1,2,3,5], [0,1,2,3,6],
            [0,1,2,4,5], [0,1,2,4,6], [0,1,2,5,6],
            [0,1,3,4,5], [0,1,3,4,6], [0,1,3,5,6],
            [0,1,4,5,6], [0,2,3,4,5], [0,2,3,4,6],
            [0,2,3,5,6], [0,2,4,5,6], [0,3,4,5,6],
            [1,2,3,4,5], [1,2,3,4,6], [1,2,3,5,6],
            [1,2,4,5,6], [1,3,4,5,6], [2,3,4,5,6]
        ]
        return [cards[i] for i in combos[index]]
    
    def _evaluate_5_card_hand(self, cards: List[Card]) -> Tuple[int, str, List[Card]]:
        """
        Evaluate a 5-card poker hand.
        Returns (hand_strength, hand_name, cards)
        """
        # Sort cards by rank
        cards = sorted(cards, key=lambda x: x.rank_idx)
        
        # Check for flush
        is_flush = len(set(c.suit for c in cards)) == 1
        
        # Get rank counts
        ranks = [c.rank_idx for c in cards]
        rank_counts = Counter(ranks)
        
        # Check for straight
        unique_ranks = sorted(set(ranks))
        is_straight = False
        for straight in self.straights:
            if set(unique_ranks) == straight:
                is_straight = True
                break
                
        # Determine hand type and strength
        if is_straight and is_flush:
            return 8000 + max(ranks), "Straight Flush", cards
            
        if 4 in rank_counts.values():
            quads = [r for r,c in rank_counts.items() if c == 4][0]
            return 7000 + quads, "Four of a Kind", cards
            
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trips = [r for r,c in rank_counts.items() if c == 3][0]
            pair = [r for r,c in rank_counts.items() if c == 2][0]
            return 6000 + trips*13 + pair, "Full House", cards
            
        if is_flush:
            return 5000 + max(ranks), "Flush", cards
            
        if is_straight:
            return 4000 + max(ranks), "Straight", cards
            
        if 3 in rank_counts.values():
            trips = [r for r,c in rank_counts.items() if c == 3][0]
            kickers = sorted([r for r,c in rank_counts.items() if c == 1], reverse=True)
            return 3000 + trips*169 + kickers[0]*13 + kickers[1], "Three of a Kind", cards
            
        if list(rank_counts.values()).count(2) == 2:
            pairs = sorted([r for r,c in rank_counts.items() if c == 2], reverse=True)
            kicker = [r for r,c in rank_counts.items() if c == 1][0]
            return 2000 + pairs[0]*169 + pairs[1]*13 + kicker, "Two Pair", cards
            
        if 2 in rank_counts.values():
            pair = [r for r,c in rank_counts.items() if c == 2][0]
            kickers = sorted([r for r,c in rank_counts.items() if c == 1], reverse=True)
            return 1000 + pair*2197 + kickers[0]*169 + kickers[1]*13 + kickers[2], "One Pair", cards
            
        return sum(r * (13 ** i) for i,r in enumerate(reversed(ranks))), "High Card", cards
    
    def hand_to_tensor(self, cards: List[Card]) -> np.ndarray:
        """Convert hand to 52-dimensional binary tensor"""
        tensor = np.zeros(52, dtype=np.float32)
        for card in cards:
            tensor[card.to_tensor_index()] = 1
        return tensor
    
    def equity_vs_range(self, hand: List[Card], range_cards: List[List[Card]], board: List[Card] = None) -> float:
        """
        Calculate equity of hand vs a range of hands
        Args:
            hand: Hero's hole cards
            range_cards: List of possible villain hands
            board: Optional community cards (flop, turn, river)
        Returns:
            Equity as percentage (0-100)
        """
        if board is None:
            board = []
            
        total_wins = 0
        total_hands = 0
        
        # All cards in play
        used_cards = set(str(c) for c in hand + board)
        
        for villain_hand in range_cards:
            # Skip if villain hand uses any cards we have
            if any(str(c) in used_cards for c in villain_hand):
                continue
                
            villain_used = set(str(c) for c in villain_hand)
            remaining = [c for c in self.deck if str(c) not in used_cards and str(c) not in villain_used]
            
            if len(board) == 5:  # River
                hero_score = self.evaluate_hand(hand + board)[0]
                villain_score = self.evaluate_hand(villain_hand + board)[0]
                total_hands += 1
                if hero_score > villain_score:
                    total_wins += 1
                elif hero_score == villain_score:
                    total_wins += 0.5
                    
            else:  # Need to deal more cards
                needed = 5 - len(board)
                for future_board in self._get_future_boards(remaining, needed):
                    full_board = board + future_board
                    hero_score = self.evaluate_hand(hand + full_board)[0]
                    villain_score = self.evaluate_hand(villain_hand + full_board)[0]
                    total_hands += 1
                    if hero_score > villain_score:
                        total_wins += 1
                    elif hero_score == villain_score:
                        total_wins += 0.5
                        
        return (total_wins / total_hands) * 100 if total_hands > 0 else 0
    
    def _get_future_boards(self, remaining: List[Card], num_cards: int, partial: List[Card] = None) -> List[List[Card]]:
        """Generate all possible future board cards"""
        if partial is None:
            partial = []
            
        if num_cards == 0:
            return [partial]
            
        if not remaining:
            return []
            
        results = []
        for i, card in enumerate(remaining):
            new_partial = partial + [card]
            new_remaining = remaining[i+1:]
            results.extend(self._get_future_boards(new_remaining, num_cards-1, new_partial))
            
        return results
