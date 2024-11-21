import torch
import numpy as np
from models.poker_dqn import PokerDQN
from environments.poker.poker_env import NoLimitHoldemEnv
from utils.metrics import PokerMetricsTracker
import os
import argparse
from datetime import datetime
import json
from typing import Dict, Any
import logging

def setup_logging(save_dir: str):
    """Setup logging configuration"""
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_save_dir(base_dir: str) -> str:
    """Create and return save directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_config(config: Dict[str, Any], save_dir: str):
    """Save training configuration"""
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def train_poker_ai(config: Dict[str, Any]):
    """Main training function for poker AI"""
    # Create save directory and setup logging
    save_dir = create_save_dir(config['save_dir'])
    logger = setup_logging(save_dir)
    save_config(config, save_dir)
    
    # Set device
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")
    
    # Initialize environment
    env = NoLimitHoldemEnv(
        num_players=config['num_players'],
        starting_stack=config['starting_stack'],
        small_blind=config['small_blind']
    )
    
    # Initialize networks
    policy_net = PokerDQN(
        num_actions=env.action_space.n,
        lstm_hidden_size=config['lstm_hidden_size'],
        lstm_num_layers=config['lstm_num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    target_net = PokerDQN(
        num_actions=env.action_space.n,
        lstm_hidden_size=config['lstm_hidden_size'],
        lstm_num_layers=config['lstm_num_layers'],
        dropout=config['dropout']
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        policy_net.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize metrics tracker
    metrics = PokerMetricsTracker(
        save_dir=save_dir,
        window_size=config['metrics_window']
    )
    
    # Training loop
    logger.info("Starting training...")
    epsilon = config['epsilon_start']
    
    for episode in range(config['num_episodes']):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        hidden_state = None
        episode_metrics = {}
        
        # Convert initial state to tensors
        state = {k: torch.FloatTensor(v).unsqueeze(0).to(device) 
                for k, v in state.items()}
        
        while not done:
            # Select action
            valid_actions = env.get_valid_actions()
            action, hidden_state = policy_net.select_action(
                state,
                epsilon=epsilon,
                hidden_state=hidden_state,
                valid_actions=valid_actions
            )
            
            # Execute action
            next_state, reward, done, _, info = env.step(action.item())
            episode_reward += reward
            
            # Convert next_state to tensors
            next_state = {k: torch.FloatTensor(v).unsqueeze(0).to(device) 
                         for k, v in next_state.items()}
            
            # Store transition and optimize
            if len(metrics.episode_rewards) > config['min_replay_size']:
                loss = policy_net.optimize_model(
                    state, action, reward, next_state, done,
                    target_net, optimizer,
                    gamma=config['gamma'],
                    device=device
                )
                episode_metrics['loss'] = loss
            
            # Move to next state
            state = next_state
            
        # Update target network
        if episode % config['target_update'] == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        # Decay epsilon
        epsilon = max(
            config['epsilon_end'],
            epsilon * config['epsilon_decay']
        )
        
        # Update metrics
        episode_metrics.update({
            'reward': episode_reward,
            'epsilon': epsilon,
            'won': info.get('won', False),
            'nash_distance': info.get('nash_distance', 0),
            'action_dist': info.get('action_distribution', {}),
            'bet_size': info.get('bet_size', 0),
            'bluff_freq': info.get('bluff_frequency', 0),
            'showdown_win': info.get('showdown_win', False),
            'stack_size': info.get('stack_size', 0),
            'hand_strength': info.get('hand_strength', 0)
        })
        metrics.update(episode_metrics)
        
        # Log progress
        if episode % config['log_freq'] == 0:
            stats = metrics.get_current_stats()
            logger.info(
                f"Episode {episode}/{config['num_episodes']} | "
                f"Avg Reward: {stats['avg_reward']:.2f} | "
                f"Win Rate: {stats['avg_win_rate']:.2f} | "
                f"Nash Distance: {stats['avg_nash_distance']:.4f} | "
                f"Epsilon: {epsilon:.4f}"
            )
            
        # Save checkpoints and plots
        if episode % config['save_freq'] == 0:
            # Save model
            checkpoint = {
                'episode': episode,
                'policy_state_dict': policy_net.state_dict(),
                'target_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon
            }
            torch.save(
                checkpoint,
                os.path.join(save_dir, f'checkpoint_{episode}.pt')
            )
            
            # Generate and save plots
            metrics.plot_training_curves()
            metrics.plot_hand_analysis()
            metrics.save_metrics()
            
    logger.info("Training complete!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Poker AI')
    
    # Environment settings
    parser.add_argument('--num_players', type=int, default=6)
    parser.add_argument('--starting_stack', type=int, default=10000)
    parser.add_argument('--small_blind', type=int, default=50)
    
    # Model settings
    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training settings
    parser.add_argument('--num_episodes', type=int, default=100000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--target_update', type=int, default=10)
    parser.add_argument('--min_replay_size', type=int, default=1000)
    
    # Logging and saving settings
    parser.add_argument('--save_dir', type=str, default='runs')
    parser.add_argument('--metrics_window', type=int, default=100)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    config = vars(args)
    
    train_poker_ai(config)
