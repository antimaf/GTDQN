# Models

This directory contains neural network architectures and model definitions for the GT-DQN agent.

## Components

### DQN Models
- `dqn.py`: Base DQN architecture with both standard and dueling networks
- `lstm_dqn.py`: LSTM-based DQN for handling sequential poker decisions
- `nash_dqn.py`: DQN variant incorporating Nash equilibrium calculations

### Network Components
- `attention.py`: Self-attention mechanisms for card and action sequences
- `embeddings.py`: Card and action embedding layers
- `heads.py`: Various network heads (value, advantage, policy)

## Architecture Details

### Main Network Structure
```
Input -> Embeddings -> LSTM -> Attention -> Value/Policy Heads -> Output
```

1. **Input Processing**
   - Card embeddings for hole and community cards
   - Action history embeddings
   - Numerical features (pot, stacks, etc.)

2. **Sequential Processing**
   - LSTM layers for temporal dependencies
   - Self-attention for card combinations

3. **Output Heads**
   - Value estimation
   - Action probabilities
   - Nash equilibrium approximation

## Usage

```python
from models.nash_dqn import NashDQN

model = NashDQN(
    state_dim=512,
    action_dim=8,
    hidden_dim=256,
    n_layers=2
)

# Forward pass
q_values = model(state)
```

## Training Configuration

Default hyperparameters:
```python
{
    'lr': 0.0001,
    'batch_size': 256,
    'hidden_dim': 256,
    'n_layers': 2,
    'dropout': 0.1,
    'gamma': 0.99
}
```
