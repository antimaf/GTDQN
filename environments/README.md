# Environments

This directory contains the poker game environments used for training the GT-DQN agent.

## Files

- `poker_env.py`: Main poker environment implementing the OpenAI Gym interface
- `state_builder.py`: Handles state representation and encoding for the poker environment
- `action_space.py`: Defines the action space and valid actions for the poker game

## State Representation

The poker environment uses a dictionary-based state representation that includes:

- **Hole Cards**: Player's private cards (2 cards)
- **Community Cards**: Shared cards on the table (up to 5 cards)
- **Pot Size**: Current size of the pot
- **Stack Sizes**: Remaining chips for each player
- **Betting History**: Sequence of previous actions
- **Position**: Player positions (dealer, small blind, big blind)
- **Game Phase**: Current phase of the game (pre-flop, flop, turn, river)

## Action Space

The action space includes:
1. Fold
2. Check/Call
3. Bet/Raise with multiple bet sizes (0.5x, 1x, 2x, 3x pot)

## Usage

```python
from environments.poker_env import PokerEnv

# Create environment
env = PokerEnv(
    stack_size=1000,
    n_players=2,
    blinds=(1, 2)
)

# Reset environment
state = env.reset()

# Take action
next_state, reward, done, info = env.step(action)
```
