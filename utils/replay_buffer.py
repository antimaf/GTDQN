import numpy as np
import torch
from collections import deque, namedtuple
from typing import Tuple, List

# Define experience tuple structure
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'hidden_state'])

class ReplayBuffer:
    """
    Replay Buffer for storing and sampling experiences with LSTM hidden states.
    """
    def __init__(self, capacity: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Replay Buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            device: Device to store tensors on
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device
        
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None
    ):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            hidden_state: LSTM hidden state
        """
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        self.buffer.append(Experience(state, action, reward, next_state, done, hidden_state))
        
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of batched experiences
        """
        experiences = np.random.choice(self.buffer, batch_size, replace=False)
        
        # Unzip experiences into separate arrays
        states = torch.stack([e.state for e in experiences])
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float).to(self.device)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float).to(self.device)
        
        # Handle hidden states
        hidden_states = [e.hidden_state for e in experiences]
        if hidden_states[0] is not None:
            hidden_states = (
                torch.cat([h[0] for h in hidden_states], dim=1),
                torch.cat([h[1] for h in hidden_states], dim=1)
            )
        else:
            hidden_states = None
            
        return (states, actions, rewards, next_states, dones, hidden_states)
    
    def __len__(self) -> int:
        """Return current size of replay buffer."""
        return len(self.buffer)
