import numpy as np
from typing import List, Tuple, Dict
import math

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = None
        
    def ucb_score(self, exploration_weight=1.414):
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class MCTSAgent:
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
        
    def select_action(self, state: np.ndarray) -> float:
        """Select an action using MCTS."""
        root = MCTSNode(state)
        private_value = state[0]
        
        # Initialize untried actions as discretized bid values
        root.untried_actions = np.linspace(0, 1, 100).tolist()
        
        for _ in range(self.num_simulations):
            node = self._select(root)
            value = self._simulate(node)
            self._backpropagate(node, value)
            
        # Select best action based on visit count
        if root.children:
            best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        else:
            # If no children, make a reasonable bid
            current_bid = state[1]
            best_action = min(private_value, current_bid + 0.1)
            
        return float(best_action)
        
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand."""
        while node.untried_actions is None or len(node.untried_actions) == 0:
            if not node.children:
                return node
            node = max(node.children.values(), key=lambda n: n.ucb_score())
        return node
        
    def _simulate(self, node: MCTSNode) -> float:
        """Run random simulation from current state."""
        current_state = node.state.copy()
        private_value = current_state[0]
        highest_bid = current_state[1]
        
        # Make a random bid between current highest and private value
        min_bid = highest_bid + 0.01
        max_bid = min(1.0, private_value)
        
        if min_bid > max_bid:
            return 0  # Can't make valid bid
            
        action = np.random.uniform(min_bid, max_bid)
        
        # Calculate reward
        if action > highest_bid:
            reward = private_value - action if action <= private_value else -action
        else:
            reward = 0
            
        return reward
        
    def _backpropagate(self, node: MCTSNode, value: float):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
            
    def _step(self, state: np.ndarray, action: float) -> Tuple[float, np.ndarray, bool]:
        # Simplified bidding game dynamics for simulation
        private_value = state[0]
        highest_bid = state[1]
        
        if action > highest_bid:
            reward = private_value - action if action <= private_value else -action
            next_state = state.copy()
            next_state[1] = action
        else:
            reward = 0
            next_state = state.copy()
            
        return reward, next_state, True  # Single round simulation
