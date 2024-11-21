import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class PokerCNN(nn.Module):
    """
    CNN for processing card inputs (hole cards and community cards).
    Uses separate channels for different card properties and positions.
    """
    def __init__(
        self,
        input_channels: int,  # Number of card slots (2 hole + 5 community = 7)
        hidden_channels: List[int] = [64, 128, 256]
    ):
        super().__init__()
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        current_channels = input_channels
        
        for out_channels in hidden_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))
            current_channels = out_channels
            
        # Calculate output size
        self.output_size = hidden_channels[-1] * 4  # After pooling
        
class PokerDQN(nn.Module):
    """
    Deep Q-Network for No-Limit Hold'em Poker
    
    Architecture:
    1. Card Processing:
        - CNN for spatial patterns in cards
        - Separate processing for hole cards and community cards
        
    2. Game State Processing:
        - Dense layers for pot, stacks, position
        - LSTM for betting history
        
    3. Action Selection:
        - Combines card features and game state
        - Dueling architecture for better Q-value estimation
        - Nash equilibrium consideration through advantage mixing
    """
    def __init__(
        self,
        num_actions: int = 5,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Card processing networks
        self.hole_cards_cnn = PokerCNN(input_channels=2)
        self.community_cards_cnn = PokerCNN(input_channels=5)
        
        # LSTM for betting history
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=6,  # [pot, position, 4 betting rounds]
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Combine features
        combined_size = (
            self.hole_cards_cnn.output_size +
            self.community_cards_cnn.output_size +
            lstm_hidden_size
        )
        
        # Advantage stream (action-dependent)
        self.advantage_hidden = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.advantage = nn.Linear(256, num_actions)
        
        # Value stream (state-dependent)
        self.value_hidden = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.value = nn.Linear(256, 1)
        
        # Nash mixing network
        self.nash_hidden = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.nash_weights = nn.Linear(256, num_actions)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
                
    def _process_cards(self, hole_cards: torch.Tensor, community_cards: torch.Tensor) -> torch.Tensor:
        """Process hole cards and community cards through CNNs"""
        # Reshape card tensors to [batch, channels, height, width]
        hole_cards = hole_cards.view(-1, 2, 13, 4)  # 13 ranks, 4 suits
        community_cards = community_cards.view(-1, 5, 13, 4)
        
        # Process through CNNs
        hole_features = self.hole_cards_cnn(hole_cards)
        community_features = self.community_cards_cnn(community_cards)
        
        return torch.cat([hole_features, community_features], dim=1)
        
    def _process_history(
        self,
        pot: torch.Tensor,
        position: torch.Tensor,
        bets_preflop: torch.Tensor,
        bets_flop: torch.Tensor,
        bets_turn: torch.Tensor,
        bets_river: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process game state history through LSTM"""
        # Combine features
        batch_size = pot.size(0)
        features = torch.cat([
            pot,
            position,
            bets_preflop.unsqueeze(-1),
            bets_flop.unsqueeze(-1),
            bets_turn.unsqueeze(-1),
            bets_river.unsqueeze(-1)
        ], dim=-1)
        
        # Initialize hidden state if needed
        if hidden_state is None:
            h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm_hidden_size, device=pot.device)
            c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm_hidden_size, device=pot.device)
            hidden_state = (h0, c0)
            
        # Process through LSTM
        lstm_out, new_hidden = self.lstm(features.unsqueeze(1), hidden_state)
        return lstm_out.squeeze(1), new_hidden
        
    def forward(
        self,
        state: Dict[str, torch.Tensor],
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the network
        
        Returns:
            q_values: Action values
            nash_probs: Nash equilibrium probabilities
            new_hidden: New LSTM hidden state
        """
        # Process cards
        card_features = self._process_cards(state['hole_cards'], state['community_cards'])
        
        # Process game state history
        history_features, new_hidden = self._process_history(
            state['pot'],
            state['position'],
            state['bets_preflop'],
            state['bets_flop'],
            state['bets_turn'],
            state['bets_river'],
            hidden_state
        )
        
        # Combine all features
        combined = torch.cat([card_features, history_features], dim=1)
        
        # Dueling DQN
        advantage = self.advantage(self.advantage_hidden(combined))
        value = self.value(self.value_hidden(combined))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Nash equilibrium mixing
        nash_logits = self.nash_weights(self.nash_hidden(combined))
        nash_probs = F.softmax(nash_logits, dim=1)
        
        return q_values, nash_probs, new_hidden
        
    def select_action(
        self,
        state: Dict[str, torch.Tensor],
        epsilon: float = 0.0,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        valid_actions: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Select action using epsilon-greedy strategy with Nash equilibrium mixing
        
        Returns:
            action: Selected action
            new_hidden: New LSTM hidden state
        """
        if np.random.random() < epsilon:
            # Random action from valid actions
            if valid_actions:
                action = torch.tensor([np.random.choice(valid_actions)])
            else:
                action = torch.randint(0, 5, (1,))
            return action, hidden_state
            
        with torch.no_grad():
            q_values, nash_probs, new_hidden = self.forward(state, hidden_state)
            
            # Mask invalid actions with large negative values
            if valid_actions:
                mask = torch.ones_like(q_values) * float('-inf')
                mask[0, valid_actions] = 0
                q_values = q_values + mask
            
            # Mix Q-values with Nash probabilities
            mixed_values = 0.7 * q_values + 0.3 * nash_probs
            action = mixed_values.argmax(dim=1)
            
        return action, new_hidden
