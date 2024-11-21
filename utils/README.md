# Utils

This directory contains utility functions and helper classes used throughout the GT-DQN project.

## Components

### Training Utilities
- `replay_buffer.py`: Experience replay buffer with LSTM state support
- `scheduler.py`: Learning rate and exploration rate schedulers
- `metrics.py`: Training metrics and logging utilities

### Poker Utilities
- `card_utils.py`: Card manipulation and hand evaluation functions
- `nash_utils.py`: Nash equilibrium calculation utilities

## Replay Buffer

The replay buffer is designed to handle complex poker states:
- Dictionary-based state representation
- LSTM hidden state management
- Efficient batch sampling
- GPU memory optimization

Example usage:
```python
from utils.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(
    capacity=100000,
    device="cuda"
)

# Store experience
buffer.push(
    state=current_state,
    action=action,
    reward=reward,
    next_state=next_state,
    done=done,
    hidden_state=lstm_hidden
)

# Sample batch
states, actions, rewards, next_states, dones, hidden_states = buffer.sample(32)
```

## Nash Equilibrium Utilities

Functions for computing and approximating Nash equilibria:
- Best response calculations
- Regret minimization
- Strategy profile updates

## Metrics and Logging

Training metrics tracked:
- Episode rewards
- Win rates
- Nash equilibrium distance
- Loss values
- Learning rates
- Exploration rates
