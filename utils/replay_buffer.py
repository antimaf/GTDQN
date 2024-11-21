import torch
import numpy as np
from collections import deque, namedtuple
from typing import Dict, Tuple, List, Optional, Union

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'hidden_state'])

class ReplayBuffer:
    """Replay Buffer for storing and sampling experiences with LSTM hidden states."""
    
    def __init__(self, capacity: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        
    def push(
        self,
        state: Dict[str, Union[np.ndarray, torch.Tensor]],
        action: int,
        reward: float,
        next_state: Dict[str, Union[np.ndarray, torch.Tensor]],
        done: bool,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> None:
        """Store experience in replay buffer."""
        
        # Convert state dictionary values to tensors if they aren't already
        state_tensors = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                state_tensors[k] = torch.FloatTensor(v).to(self.device)
            elif isinstance(v, torch.Tensor):
                state_tensors[k] = v.to(self.device)
            else:
                state_tensors[k] = torch.FloatTensor([v]).to(self.device)
                
        next_state_tensors = {}
        for k, v in next_state.items():
            if isinstance(v, np.ndarray):
                next_state_tensors[k] = torch.FloatTensor(v).to(self.device)
            elif isinstance(v, torch.Tensor):
                next_state_tensors[k] = v.to(self.device)
            else:
                next_state_tensors[k] = torch.FloatTensor([v]).to(self.device)
        
        # Convert action, reward, and done to tensors
        action_tensor = torch.tensor([action], device=self.device, dtype=torch.long)
        reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
        done_tensor = torch.tensor([done], device=self.device, dtype=torch.float32)
        
        # Store the experience
        self.buffer.append(Experience(
            state=state_tensors,
            action=action_tensor,
            reward=reward_tensor,
            next_state=next_state_tensors,
            done=done_tensor,
            hidden_state=hidden_state
        ))
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        experiences = np.random.choice(self.buffer, batch_size, replace=False)
        
        # Get first state to determine keys
        first_state = experiences[0].state
        state_keys = first_state.keys()
        
        # Batch states by key
        states = {
            k: torch.stack([e.state[k] for e in experiences]).to(self.device)
            for k in state_keys
        }
        
        next_states = {
            k: torch.stack([e.next_state[k] for e in experiences]).to(self.device)
            for k in state_keys
        }
        
        # Stack other elements
        actions = torch.stack([e.action for e in experiences]).to(self.device)
        rewards = torch.stack([e.reward for e in experiences]).to(self.device)
        dones = torch.stack([e.done for e in experiences]).to(self.device)
        
        # Handle hidden states if present
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
        return len(self.buffer)
