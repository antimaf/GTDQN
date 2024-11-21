import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional

class BiddingGame(gym.Env):
    """
    A multi-agent bidding game environment.
    
    Agents compete in auctions with private valuations.
    Each agent receives a private value and must decide how much to bid.
    The highest bidder wins and pays their bid amount.
    
    Strategic elements:
    - Value estimation
    - Bluffing
    - Opponent modeling
    """
    
    def __init__(
        self,
        num_agents: int = 3,
        max_value: float = 100.0,
        max_rounds: int = 50,
        min_increment: float = 1.0
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.max_value = max_value
        self.max_rounds = max_rounds
        self.min_increment = min_increment
        
        # Action space: bid amount (normalized)
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space: [private_value, current_highest_bid, own_cash, round]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode."""
        self.current_round = 0
        self.done = False
        
        # Generate private values for each agent
        self.private_values = np.random.uniform(0, self.max_value, self.num_agents)
        self.private_values = self.private_values / self.max_value  # Normalize
        
        # Initialize game state
        self.current_highest_bid = 0
        self.highest_bidder = None
        self.cash = np.ones(self.num_agents)  # Start with max cash
        self.current_agent = 0
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
        
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.done:
            raise RuntimeError("Episode is done, call reset() to start a new episode")
            
        # Normalize and validate bid
        bid = float(action)  # Convert action to float
        bid = np.clip(bid, 0, 1)
        
        # Check if bid is valid
        if bid <= self.current_highest_bid:
            reward = 0  # Invalid bid
        else:
            # Update highest bid
            self.current_highest_bid = bid
            self.highest_bidder = self.current_agent
            
            # Calculate reward
            if bid <= self.private_values[self.current_agent]:
                reward = self.private_values[self.current_agent] - bid
            else:
                reward = -bid
                
        # Update cash
        if self.highest_bidder == self.current_agent:
            self.cash[self.current_agent] -= bid
            
        # Move to next agent
        self.current_agent = (self.current_agent + 1) % self.num_agents
        
        # Check if round is complete
        if self.current_agent == 0:
            self.current_round += 1
            if self.current_round >= self.max_rounds:
                self.done = True
                
        # Get next observation
        obs = self._get_observation()
        
        return obs, reward, self.done, False, {}
        
    def _get_observation(self) -> np.ndarray:
        """Get the current observation for the active agent."""
        obs = np.array([
            self.private_values[self.current_agent],
            self.current_highest_bid,
            self.cash[self.current_agent],
            self.current_round / self.max_rounds
        ], dtype=np.float32)
        
        # Ensure observation is correct shape
        assert obs.shape == (4,), f"Observation shape is {obs.shape}, expected (4,)"
        return obs
        
    def render(self):
        """Render the current state of the environment."""
        print(f"\nRound {self.current_round + 1}/{self.max_rounds}")
        print(f"Current Agent: {self.current_agent}")
        print(f"Private Value: {self.private_values[self.current_agent]:.2f}")
        print(f"Highest Bid: {self.current_highest_bid:.2f}")
        print(f"Cash: {self.cash[self.current_agent]:.2f}")
        if self.highest_bidder is not None:
            print(f"Highest Bidder: Agent {self.highest_bidder}")
