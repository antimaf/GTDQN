import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class HistoryProcessor(nn.Module):
    """
    Process betting history using LSTM with attention mechanism
    """
    def __init__(self, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for processing betting history
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1
        )
        
        # Layer normalisation
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process betting history with LSTM and attention
        
        Args:
            x: Betting history [batch_size, seq_len, hidden_size]
            mask: Attention mask for padding
            
        Returns:
            torch.Tensor: Processed history representation
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Self-attention mechanism
        # Reshape for attention [seq_len, batch_size, hidden_size]
        lstm_out = lstm_out.transpose(0, 1)
        attended, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=mask,
            need_weights=False
        )
        
        # Residual connection and layer norm
        attended = self.layer_norm1(lstm_out + attended)
        
        # Position-wise FFN
        ffn_out = self.ffn(attended)
        
        # Final residual connection and layer norm
        output = self.layer_norm2(attended + ffn_out)
        
        # Reshape back to [batch_size, seq_len, hidden_size]
        output = output.transpose(0, 1)
        
        return output

class ActionEncoder(nn.Module):
    """
    Encode action sequence with positional information
    """
    def __init__(self, num_actions: int, hidden_size: int = 128):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size, max_len=50)
        
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode action sequence with positional information
        
        Args:
            actions: Action indices [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Encoded actions with positional information
        """
        # Embed actions
        embedded = self.action_embedding(actions)
        
        # Add positional encoding
        encoded = self.position_encoding(embedded)
        
        return encoded

class PositionalEncoding(nn.Module):
    """
    Inject positional information into sequence embeddings
    """
    def __init__(self, hidden_size: int, max_len: int = 50):
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        
        pe = torch.zeros(max_len, 1, hidden_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:x.size(0)]
