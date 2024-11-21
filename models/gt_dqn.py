import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

class GTDQN(nn.Module):
    """
    Game Theory Deep Q-Network (GT-DQN) for the bidding game.
    Uses a simple feedforward network since states are vectors.
    """
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_size: int = 128,
        num_layers: int = 2
    ):
        super(GTDQN, self).__init__()
        
        # Build network layers
        self.layers = nn.ModuleList()
        current_size = state_dim
        
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.1)
            ))
            current_size = hidden_size
        
        # Output layer for Q-values
        self.output = nn.Linear(hidden_size, num_actions)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-values for each action"""
        x = state
        
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Compute Q-values
        q_values = self.output(x)
        
        return q_values
    
    def select_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Select action using epsilon-greedy strategy"""
        if not deterministic and np.random.random() < epsilon:
            # Random action
            action = torch.randint(0, self.output.out_features, (1,))
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(state)
                action = q_values.argmax(dim=1)
        
        return action

    def get_nash_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action based on Nash equilibrium strategy with epsilon-exploration.
        
        Args:
            state: Current state tensor
            epsilon: Exploration rate
            
        Returns:
            action: Selected action
            nash_probs: Nash equilibrium probabilities
        """
        with torch.no_grad():
            q_values = self.forward(state)
            
            if np.random.random() < epsilon:
                # Random exploration
                action = torch.randint(0, q_values.size(-1), (1,))
            else:
                # Nash equilibrium based action selection
                action = torch.multinomial(F.softmax(q_values, dim=1), 1)
                
        return action, F.softmax(q_values, dim=1)

    def select_action(self, state: np.ndarray) -> float:
        """Select an action using the current policy."""
        # Ensure state is a numpy array with correct shape
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (4,):
            raise ValueError(f"Expected state shape (4,), got {state.shape}")
            
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.forward(state_tensor)
            action_probs = F.softmax(q_values, dim=1)
            action_idx = torch.argmax(action_probs).item()
            
        # Convert discrete action to continuous bid
        action = action_idx / 100.0  # Scale to [0,1]
        return float(action)  # Return scalar float
