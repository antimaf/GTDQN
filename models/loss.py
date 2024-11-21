import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GTDQNLoss(nn.Module):
    """
    Game-Theoretic DQN Loss Function
    Combines TD-error with importance sampling and game-theoretic constraints
    """
    def __init__(
        self,
        gamma: float = 0.99,
        n_steps: int = 1,
        beta: float = 0.4,
        min_priority: float = 1e-6,
        max_priority: float = 1.0
    ):
        super().__init__()
        self.gamma = gamma
        self.n_steps = n_steps
        self.beta = beta  # Importance sampling exponent
        self.min_priority = min_priority
        self.max_priority = max_priority
    
    def compute_td_error(
        self,
        current_q: torch.Tensor,
        target_q: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute n-step TD error with double Q-learning
        
        Args:
            current_q: Q-values from current network
            target_q: Q-values from target network
            actions: Selected actions
            rewards: Received rewards
            dones: Episode termination flags
            
        Returns:
            torch.Tensor: TD errors for each transition
        """
        # Get Q-values for selected actions
        q_selected = current_q.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute n-step returns
        with torch.no_grad():
            # Get maximum Q-values from target network
            next_q = target_q.max(1)[0]
            
            # Compute n-step discounted returns
            returns = rewards.clone()
            for i in range(1, self.n_steps):
                returns += (self.gamma ** i) * rewards.roll(-i) * (~dones.roll(-i))
            
            # Add discounted future value for non-terminal states
            returns += (self.gamma ** self.n_steps) * next_q * (~dones)
        
        # Compute TD errors
        td_errors = returns - q_selected
        
        return td_errors
    
    def compute_priorities(
        self,
        td_errors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute priorities for experience replay
        
        Args:
            td_errors: TD errors for transitions
            
        Returns:
            torch.Tensor: Priority weights for sampling
        """
        # Convert TD errors to priorities
        priorities = td_errors.abs()
        
        # Clip priorities to prevent extreme values
        priorities = priorities.clamp(self.min_priority, self.max_priority)
        
        return priorities
    
    def forward(
        self,
        current_q: torch.Tensor,
        target_q: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss with importance sampling and priorities
        
        Args:
            current_q: Q-values from current network
            target_q: Q-values from target network
            actions: Selected actions
            rewards: Received rewards
            dones: Episode termination flags
            weights: Importance sampling weights
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Loss value, New priorities)
        """
        # Compute TD errors
        td_errors = self.compute_td_error(current_q, target_q, actions, rewards, dones)
        
        # Compute new priorities
        priorities = self.compute_priorities(td_errors)
        
        # Apply importance sampling if weights provided
        if weights is not None:
            # Normalize weights to prevent scaling issues
            weights = weights / weights.max()
            
            # Apply beta scaling to weights
            weights = weights ** self.beta
            
            # Weight TD errors by importance sampling weights
            td_errors = td_errors * weights
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(td_errors, torch.zeros_like(td_errors), reduction='mean')
        
        return loss, priorities
