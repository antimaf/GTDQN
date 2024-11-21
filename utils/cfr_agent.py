import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class InfoSet:
    """Information set for CFR algorithm"""
    def __init__(self, num_actions: int):
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.num_actions = num_actions
        
    def get_strategy(self, reach_prob: float) -> np.ndarray:
        """Get current strategy from regrets"""
        strategy = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(strategy)
        
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.ones(self.num_actions) / self.num_actions
            
        self.strategy_sum += reach_prob * strategy
        return strategy
        
    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy across all iterations"""
        strategy = self.strategy_sum.copy()
        normalizing_sum = np.sum(strategy)
        
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.ones(self.num_actions) / self.num_actions
            
        return strategy

class CFRAgent:
    """
    Counterfactual Regret Minimization agent for poker
    Implements vanilla CFR algorithm
    """
    def __init__(
        self,
        num_actions: int,
        num_iterations: int = 1000,
        exploration_prob: float = 0.05
    ):
        self.num_actions = num_actions
        self.num_iterations = num_iterations
        self.exploration_prob = exploration_prob
        self.info_sets: Dict[str, InfoSet] = {}
        
    def get_info_set(self, info_set_key: str) -> InfoSet:
        """Get or create information set"""
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InfoSet(self.num_actions)
        return self.info_sets[info_set_key]
        
    def _cfr(
        self,
        history: str,
        reach_probs: List[float],
        player: int,
        env
    ) -> np.ndarray:
        """
        Recursive CFR implementation
        
        Args:
            history: String representation of game history
            reach_probs: Reach probabilities for each player
            player: Current player
            env: Poker environment
            
        Returns:
            Expected value for each action
        """
        if env.is_terminal():
            return np.array([env.get_payoff(player)])
            
        info_set = self.get_info_set(self._get_info_set_key(history, env))
        strategy = info_set.get_strategy(reach_probs[player])
        
        # Get expected value for each action
        action_values = np.zeros(self.num_actions)
        
        for action in range(self.num_actions):
            new_history = history + str(action)
            new_reach_probs = reach_probs.copy()
            new_reach_probs[player] *= strategy[action]
            
            # Simulate action in environment
            old_state = env.get_state()
            env.step(action)
            action_values[action] = self._cfr(
                new_history,
                new_reach_probs,
                1 - player,  # Switch player
                env
            )[0]
            env.set_state(old_state)  # Restore state
            
        # Compute counterfactual value
        value = np.sum(strategy * action_values)
        
        # Accumulate counterfactual regret
        if player == 0:  # Update regrets only for player 0
            opponent_reach = reach_probs[1]
            for action in range(self.num_actions):
                info_set.regret_sum[action] += (
                    opponent_reach * (action_values[action] - value)
                )
                
        return np.array([value])
        
    def _get_info_set_key(self, history: str, env) -> str:
        """
        Create string key for information set
        Combines visible cards and betting history
        """
        state = env.get_state()
        visible_cards = env.get_visible_cards()
        return f"{visible_cards}|{history}"
        
    def train(self, env) -> None:
        """Train CFR agent for specified number of iterations"""
        util = 0
        
        for i in range(self.num_iterations):
            reach_probs = [1.0, 1.0]  # Initial reach probabilities
            util += self._cfr("", reach_probs, 0, env)[0]
            
            if (i + 1) % 100 == 0:
                print(f"Average game value after {i+1} iterations: {util/(i+1)}")
                
    def act(self, state: np.ndarray, env) -> Tuple[int, float]:
        """
        Select action using learned CFR strategy
        
        Args:
            state: Current game state
            env: Poker environment
            
        Returns:
            action: Selected action
            bet_size: Bet size if action is raise
        """
        info_set_key = self._get_info_set_key("", env)
        info_set = self.get_info_set(info_set_key)
        strategy = info_set.get_average_strategy()
        
        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_prob:
            action = np.random.randint(self.num_actions)
        else:
            action = np.random.choice(self.num_actions, p=strategy)
            
        # Determine bet size based on action
        bet_size = np.random.random() if action == 2 else 0  # 2 = raise
        
        return action, bet_size
