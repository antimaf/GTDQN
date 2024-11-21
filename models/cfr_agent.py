import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

class CFRAgent:
    def __init__(self, num_actions=100):
        self.num_actions = num_actions
        self.regret_sum = defaultdict(lambda: np.zeros(num_actions))
        self.strategy_sum = defaultdict(lambda: np.zeros(num_actions))
        self.iterations = 0
        
    def get_strategy(self, state_key: str) -> np.ndarray:
        regret = self.regret_sum[state_key]
        strategy = np.maximum(regret, 0)
        normalizing_sum = np.sum(strategy)
        
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.ones(self.num_actions) / self.num_actions
            
        return strategy
        
    def get_average_strategy(self, state_key: str) -> np.ndarray:
        strategy_sum = self.strategy_sum[state_key]
        normalizing_sum = np.sum(strategy_sum)
        
        if normalizing_sum > 0:
            return strategy_sum / normalizing_sum
        return np.ones(self.num_actions) / self.num_actions
        
    def select_action(self, state: np.ndarray) -> float:
        # Convert state to discrete key for lookup
        state_key = self._state_to_key(state)
        strategy = self.get_strategy(state_key)
        
        # Convert discrete action index to continuous bid
        action_idx = np.random.choice(self.num_actions, p=strategy)
        return self._idx_to_bid(action_idx, state[0])  # Scale by private value
        
    def update(self, state: np.ndarray, action: int, reward: float):
        state_key = self._state_to_key(state)
        strategy = self.get_strategy(state_key)
        
        # Update regret and strategy sums
        regret = np.zeros(self.num_actions)
        regret[action] = reward
        
        self.regret_sum[state_key] += regret - np.dot(strategy, regret)
        self.strategy_sum[state_key] += strategy
        self.iterations += 1
        
    def _state_to_key(self, state: np.ndarray) -> str:
        # Discretize continuous state for lookup
        private_value_idx = int(state[0] * 10)  # Discretize private value
        highest_bid_idx = int(state[1] * 10)   # Discretize highest bid
        return f"{private_value_idx}_{highest_bid_idx}"
        
    def _idx_to_bid(self, action_idx: int, private_value: float) -> float:
        # Convert discrete action index to continuous bid
        return (action_idx + 1) * private_value / self.num_actions
