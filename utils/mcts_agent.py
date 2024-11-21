import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class MCTSNode:
    """Node in the MCTS tree"""
    state: np.ndarray
    parent: Optional['MCTSNode']
    action: Optional[int]
    children: Dict[int, 'MCTSNode']
    visits: int
    value: float
    prior_prob: float
    bet_size: float = 0.0
    
    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0
        
    def get_ucb_score(self, parent_visits: int, c_puct: float = 1.0) -> float:
        """
        Calculate Upper Confidence Bound score for node selection
        Uses PUCT (Predictor + UCT) formula from AlphaGo Zero
        """
        if self.visits == 0:
            return float('inf')
            
        # Exploitation term
        q_value = self.value / self.visits if self.visits > 0 else 0
        
        # Exploration term
        u_value = (c_puct * self.prior_prob * 
                  math.sqrt(parent_visits) / (1 + self.visits))
                  
        return q_value + u_value

class MCTSAgent:
    """
    Monte Carlo Tree Search agent for poker
    Implements PUCT (AlphaGo Zero style) selection
    """
    def __init__(
        self,
        num_actions: int,
        num_simulations: int = 100,
        max_depth: int = 50,
        c_puct: float = 1.0,
        gamma: float = 0.99,
        value_network = None  # Optional neural network for value estimation
    ):
        self.num_actions = num_actions
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.gamma = gamma
        self.value_network = value_network
        
    def _expand_node(
        self,
        node: MCTSNode,
        env,
        policy_probs: Optional[np.ndarray] = None
    ) -> None:
        """Expand a leaf node by adding all possible children"""
        if policy_probs is None:
            # Use uniform prior if no policy network
            policy_probs = np.ones(self.num_actions) / self.num_actions
            
        for action in range(self.num_actions):
            if action not in node.children:
                # Simulate action to get next state
                env_copy = deepcopy(env)
                next_state, _, done, _ = env_copy.step(action)
                
                if not done:
                    child = MCTSNode(
                        state=next_state,
                        parent=node,
                        action=action,
                        children={},
                        visits=0,
                        value=0.0,
                        prior_prob=policy_probs[action],
                        bet_size=np.random.random() if action == 2 else 0
                    )
                    node.children[action] = child
                    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child node using PUCT formula"""
        ucb_scores = [
            child.get_ucb_score(node.visits, self.c_puct)
            for child in node.children.values()
        ]
        best_action = list(node.children.keys())[np.argmax(ucb_scores)]
        return node.children[best_action]
        
    def _simulate(self, env, max_steps: int = 50) -> float:
        """
        Run random simulation from current state
        Returns cumulative discounted reward
        """
        if self.value_network is not None:
            # Use value network if available
            state = env.get_state()
            return self.value_network.predict(state)[0]
            
        # Random rollout
        done = False
        total_reward = 0
        step = 0
        discount = 1.0
        
        while not done and step < max_steps:
            action = np.random.randint(self.num_actions)
            _, reward, done, _ = env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            step += 1
            
        return total_reward
        
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Update statistics of all nodes in path"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
            value = -value  # Negate value for opponent's turn
            
    def search(self, root_state: np.ndarray, env) -> Dict[int, float]:
        """
        Perform MCTS search from root state
        Returns action probabilities based on visit counts
        """
        root = MCTSNode(
            state=root_state,
            parent=None,
            action=None,
            children={},
            visits=0,
            value=0.0,
            prior_prob=1.0
        )
        
        for _ in range(self.num_simulations):
            node = root
            env_sim = deepcopy(env)
            
            # Selection
            while node.is_expanded and not env_sim.is_terminal():
                node = self._select_child(node)
                env_sim.step(node.action)
                
            # Expansion
            if not env_sim.is_terminal():
                self._expand_node(node, env_sim)
                
            # Simulation
            value = self._simulate(env_sim)
            
            # Backpropagation
            self._backpropagate(node, value)
            
        # Calculate action probabilities from visit counts
        visits = np.array([
            child.visits for child in root.children.values()
        ])
        probs = visits / np.sum(visits)
        
        return {action: prob for action, prob in zip(root.children.keys(), probs)}
        
    def act(self, state: np.ndarray, env) -> Tuple[int, float]:
        """
        Select action using MCTS
        
        Args:
            state: Current game state
            env: Poker environment
            
        Returns:
            action: Selected action
            bet_size: Bet size if action is raise
        """
        action_probs = self.search(state, env)
        
        # Select action with highest visit count
        action = max(action_probs.items(), key=lambda x: x[1])[0]
        
        # Get bet size from corresponding child node
        root = MCTSNode(
            state=state,
            parent=None,
            action=None,
            children={},
            visits=0,
            value=0.0,
            prior_prob=1.0
        )
        self._expand_node(root, env)
        bet_size = root.children[action].bet_size
        
        return action, bet_size
