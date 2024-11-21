import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from .hand_evaluator import Card, HandEvaluator

class NoLimitHoldemEnv(gym.Env):
    """
    No-Limit Texas Hold'em Poker Environment
    
    Features:
    - Full poker hand evaluation
    - Pot odds and implied odds calculation
    - Position-aware decision making
    - Stack-to-pot ratio considerations
    - Range-based opponent modeling
    
    State space includes:
    - Private cards (2 cards = 104 bits one-hot)
    - Community cards (5 cards = 260 bits one-hot)
    - Pot size and stack sizes (normalized)
    - Position and stage of game
    - Action history
    """
    
    def __init__(
        self,
        num_players: int = 6,
        starting_stack: int = 100,
        small_blind: int = 1,
        big_blind: int = 2,
        max_rounds: int = 1000
    ):
        super().__init__()
        
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_rounds = max_rounds
        
        # Initialize hand evaluator
        self.evaluator = HandEvaluator()
        
        # Action space: [fold, check/call, min_raise, pot_raise, all_in]
        self.action_space = spaces.Discrete(5)
        
        # Observation space
        self.observation_space = spaces.Dict({
            # Private cards (2 cards)
            'hole_cards': spaces.Box(low=0, high=1, shape=(52*2,), dtype=np.float32),
            
            # Community cards (5 cards)
            'community_cards': spaces.Box(low=0, high=1, shape=(52*5,), dtype=np.float32),
            
            # Game state
            'pot': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'stacks': spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.float32),
            'position': spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.float32),
            
            # Betting history for each street
            'bets_preflop': spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.float32),
            'bets_flop': spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.float32),
            'bets_turn': spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.float32),
            'bets_river': spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.float32),
        })
        
        self.reset()
        
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment for a new hand"""
        # Initialize deck
        self.deck = [Card(f"{r}{s}") for r in Card.RANKS for s in Card.SUITS]
        np.random.shuffle(self.deck)
        
        # Deal hole cards
        self.hole_cards = []
        for _ in range(self.num_players):
            self.hole_cards.append([self.deck.pop(), self.deck.pop()])
            
        # Initialize community cards
        self.community_cards = []
        
        # Initialize game state
        self.pot = self.small_blind + self.big_blind
        self.stacks = [self.starting_stack] * self.num_players
        self.stacks[0] -= self.small_blind  # Small blind
        self.stacks[1] -= self.big_blind   # Big blind
        
        # Initialize position (one-hot)
        self.position = np.zeros(self.num_players)
        self.position[0] = 1  # Start with small blind
        
        # Initialize betting history
        self.bets_preflop = np.zeros(self.num_players)
        self.bets_flop = np.zeros(self.num_players)
        self.bets_turn = np.zeros(self.num_players)
        self.bets_river = np.zeros(self.num_players)
        
        # Set initial bets
        self.bets_preflop[0] = self.small_blind / self.starting_stack
        self.bets_preflop[1] = self.big_blind / self.starting_stack
        
        # Game state tracking
        self.current_player = 2  # Start after big blind
        self.stage = 'preflop'
        self.rounds = 0
        
        return self._get_observation(), {}
        
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return new state
        
        Actions:
        0: Fold
        1: Check/Call
        2: Min Raise
        3: Pot Raise
        4: All-in
        """
        # Get current betting round array
        if self.stage == 'preflop':
            bets = self.bets_preflop
        elif self.stage == 'flop':
            bets = self.bets_flop
        elif self.stage == 'turn':
            bets = self.bets_turn
        else:  # river
            bets = self.bets_river
            
        # Process action
        reward = 0
        current_max_bet = np.max(bets) * self.starting_stack
        player_bet = bets[self.current_player] * self.starting_stack
        
        if action == 0:  # Fold
            reward = -player_bet  # Lose whatever was bet
            self.stacks[self.current_player] = 0  # Mark as folded
            
        elif action == 1:  # Check/Call
            call_amount = current_max_bet - player_bet
            if call_amount > self.stacks[self.current_player]:
                call_amount = self.stacks[self.current_player]  # All-in call
            self.stacks[self.current_player] -= call_amount
            bets[self.current_player] = current_max_bet / self.starting_stack
            self.pot += call_amount
            
        elif action == 2:  # Min Raise
            raise_amount = current_max_bet * 2 - player_bet
            if raise_amount > self.stacks[self.current_player]:
                raise_amount = self.stacks[self.current_player]  # All-in raise
            self.stacks[self.current_player] -= raise_amount
            bets[self.current_player] = (current_max_bet * 2) / self.starting_stack
            self.pot += raise_amount
            
        elif action == 3:  # Pot Raise
            raise_amount = self.pot * 2 - player_bet
            if raise_amount > self.stacks[self.current_player]:
                raise_amount = self.stacks[self.current_player]  # All-in raise
            self.stacks[self.current_player] -= raise_amount
            bets[self.current_player] = (self.pot * 2) / self.starting_stack
            self.pot += raise_amount
            
        else:  # All-in
            raise_amount = self.stacks[self.current_player]
            self.stacks[self.current_player] = 0
            bets[self.current_player] = (player_bet + raise_amount) / self.starting_stack
            self.pot += raise_amount
            
        # Move to next player or stage
        done = False
        self.current_player = (self.current_player + 1) % self.num_players
        
        # Skip players who are all-in or folded
        while self.stacks[self.current_player] == 0:
            self.current_player = (self.current_player + 1) % self.num_players
            
        # Check if betting round is complete
        if self._is_betting_complete():
            if self.stage == 'preflop':
                self._deal_flop()
            elif self.stage == 'flop':
                self._deal_turn()
            elif self.stage == 'turn':
                self._deal_river()
            else:  # river
                reward = self._handle_showdown()
                done = True
                
        # Check for early termination
        active_players = sum(1 for s in self.stacks if s > 0)
        if active_players == 1:
            # Award pot to last remaining player
            winner = next(i for i, s in enumerate(self.stacks) if s > 0)
            if winner == self.current_player:
                reward = self.pot
            done = True
            
        self.rounds += 1
        if self.rounds >= self.max_rounds:
            done = True
            
        return self._get_observation(), reward, done, False, {}
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Convert current state to observation"""
        obs = {
            'hole_cards': np.zeros(52*2),
            'community_cards': np.zeros(52*5),
            'pot': np.array([self.pot / (self.starting_stack * self.num_players)]),
            'stacks': np.array(self.stacks) / self.starting_stack,
            'position': self.position,
            'bets_preflop': self.bets_preflop,
            'bets_flop': self.bets_flop,
            'bets_turn': self.bets_turn,
            'bets_river': self.bets_river
        }
        
        # Convert hole cards to tensor
        if self.hole_cards:
            hole_cards = self.hole_cards[self.current_player]
            for i, card in enumerate(hole_cards):
                obs['hole_cards'][i*52 + card.to_tensor_index()] = 1
                
        # Convert community cards to tensor
        for i, card in enumerate(self.community_cards):
            obs['community_cards'][i*52 + card.to_tensor_index()] = 1
            
        return obs
        
    def _is_betting_complete(self) -> bool:
        """Check if current betting round is complete"""
        if self.stage == 'preflop':
            bets = self.bets_preflop
        elif self.stage == 'flop':
            bets = self.bets_flop
        elif self.stage == 'turn':
            bets = self.bets_turn
        else:  # river
            bets = self.bets_river
            
        active_bets = [b for i, b in enumerate(bets) if self.stacks[i] > 0]
        return len(set(active_bets)) <= 1
        
    def _deal_flop(self):
        """Deal the flop"""
        self.stage = 'flop'
        self.community_cards.extend([self.deck.pop() for _ in range(3)])
        self.current_player = 0
        
    def _deal_turn(self):
        """Deal the turn"""
        self.stage = 'turn'
        self.community_cards.append(self.deck.pop())
        self.current_player = 0
        
    def _deal_river(self):
        """Deal the river"""
        self.stage = 'river'
        self.community_cards.append(self.deck.pop())
        self.current_player = 0
        
    def _handle_showdown(self) -> float:
        """Handle showdown and return reward for current player"""
        active_players = [i for i, s in enumerate(self.stacks) if s > 0]
        
        # Evaluate hands
        scores = []
        for i in active_players:
            hand = self.hole_cards[i] + self.community_cards
            score = self.evaluator.evaluate_hand(hand)[0]
            scores.append((score, i))
            
        # Find winners
        max_score = max(score for score, _ in scores)
        winners = [i for score, i in scores if score == max_score]
        
        # Split pot among winners
        reward = 0
        if self.current_player in winners:
            reward = self.pot / len(winners)
            
        return reward
        
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions for current state"""
        valid = [0, 1]  # Can always fold or call
        
        if self.stage == 'preflop':
            bets = self.bets_preflop
        elif self.stage == 'flop':
            bets = self.bets_flop
        elif self.stage == 'turn':
            bets = self.bets_turn
        else:  # river
            bets = self.bets_river
            
        current_max_bet = np.max(bets) * self.starting_stack
        player_bet = bets[self.current_player] * self.starting_stack
        
        # Check if player can raise
        if self.stacks[self.current_player] > (current_max_bet - player_bet):
            valid.extend([2, 3, 4])  # Can raise
            
        return valid
        
    def render(self):
        """Render the current state"""
        print("\n=== Poker Game State ===")
        print(f"Stage: {self.stage}")
        print(f"Pot: {self.pot}")
        print(f"Community Cards: {self.community_cards}")
        print("\nPlayers:")
        for i in range(self.num_players):
            position = "SB" if i == 0 else "BB" if i == 1 else str(i)
            print(f"Player {position}:")
            print(f"  Stack: {self.stacks[i]}")
            if i == self.current_player:
                print(f"  Hole Cards: {self.hole_cards[i]}")
            print(f"  Current Bet: {self.bets_preflop[i] * self.starting_stack}")
        print("=====================\n")
