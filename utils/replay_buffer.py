import numpy as np
import torch
from collections import deque, namedtuple
from typing import Tuple, List, Dict

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
        state: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: bool,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None
    ):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state (dictionary of tensors)
            action: Action taken
            reward: Reward received
            next_state: Next state (dictionary of tensors)
            done: Whether episode ended
            hidden_state: LSTM hidden state
        """
        # Convert state dictionaries to device tensors if they aren't already
        if not isinstance(state[next(iter(state))], torch.Tensor):
            state = {k: torch.FloatTensor(v).to(self.device) for k, v in state.items()}
        if not isinstance(next_state[next(iter(next_state))], torch.Tensor):
            next_state = {k: torch.FloatTensor(v).to(self.device) for k, v in next_state.items()}
        
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
        
        # Get first state to determine keys
        first_state = experiences[0].state
        state_keys = first_state.keys()
        
        # Batch states by key
        states = {
            k: torch.stack([e.state[k] for e in experiences]) 
            for k in state_keys
        }
        next_states = {
            k: torch.stack([e.next_state[k] for e in experiences])
            for k in state_keys
        }
        
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float).to(self.device)
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
