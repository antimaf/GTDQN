import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import os
import json
from collections import deque
import torch

class PokerMetricsTracker:
    """Sophisticated metrics tracking for poker AI training"""
    
    def __init__(
        self,
        save_dir: str,
        window_size: int = 100,
        plot_style: str = 'darkgrid'
    ):
        self.save_dir = save_dir
        self.metrics_dir = os.path.join(save_dir, 'metrics')
        self.plots_dir = os.path.join(save_dir, 'plots')
        self.window_size = window_size
        
        # Create directories
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set plot style
        sns.set_style(plot_style)
        plt.style.use('seaborn')
        
        # Initialize metrics storage
        self.episode_rewards = []
        self.win_rates = []
        self.nash_distances = []
        self.loss_history = []
        self.epsilon_history = []
        self.action_distributions = []
        self.bet_sizes = []
        self.bluff_frequencies = []
        self.showdown_win_rates = []
        self.stack_sizes = []
        self.hand_strengths = []
        
        # Rolling statistics
        self.rolling_rewards = deque(maxlen=window_size)
        self.rolling_wins = deque(maxlen=window_size)
        self.rolling_nash = deque(maxlen=window_size)
        
        # Training start time
        self.start_time = datetime.now()
        
    def update(
        self,
        metrics: Dict[str, Any]
    ):
        """Update metrics with new episode data"""
        # Extract metrics
        self.episode_rewards.append(metrics.get('reward', 0))
        self.win_rates.append(metrics.get('won', False))
        self.nash_distances.append(metrics.get('nash_distance', 0))
        self.loss_history.append(metrics.get('loss', 0))
        self.epsilon_history.append(metrics.get('epsilon', 0))
        self.action_distributions.append(metrics.get('action_dist', {}))
        self.bet_sizes.append(metrics.get('bet_size', 0))
        self.bluff_frequencies.append(metrics.get('bluff_freq', 0))
        self.showdown_win_rates.append(metrics.get('showdown_win', False))
        self.stack_sizes.append(metrics.get('stack_size', 0))
        self.hand_strengths.append(metrics.get('hand_strength', 0))
        
        # Update rolling statistics
        self.rolling_rewards.append(metrics.get('reward', 0))
        self.rolling_wins.append(metrics.get('won', False))
        self.rolling_nash.append(metrics.get('nash_distance', 0))
        
    def plot_training_curves(self):
        """Generate comprehensive training curve plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        fig.suptitle('Poker AI Training Metrics', fontsize=16)
        
        # 1. Rewards
        ax = axes[0, 0]
        rewards_smooth = pd.Series(self.episode_rewards).rolling(self.window_size).mean()
        ax.plot(self.episode_rewards, alpha=0.3, color='blue', label='Raw')
        ax.plot(rewards_smooth, color='blue', label=f'{self.window_size}-ep Moving Avg')
        ax.set_title('Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        
        # 2. Win Rate
        ax = axes[0, 1]
        win_rate_smooth = pd.Series(self.win_rates).rolling(self.window_size).mean()
        ax.plot(win_rate_smooth, color='green')
        ax.set_title('Win Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        
        # 3. Nash Distance
        ax = axes[1, 0]
        nash_smooth = pd.Series(self.nash_distances).rolling(self.window_size).mean()
        ax.plot(nash_smooth, color='red')
        ax.set_title('Nash Equilibrium Distance')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Distance')
        
        # 4. Loss
        ax = axes[1, 1]
        loss_smooth = pd.Series(self.loss_history).rolling(self.window_size).mean()
        ax.plot(loss_smooth, color='purple')
        ax.set_title('Training Loss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        
        # 5. Action Distribution
        ax = axes[2, 0]
        action_counts = pd.DataFrame(self.action_distributions).sum()
        ax.bar(range(len(action_counts)), action_counts.values)
        ax.set_title('Action Distribution')
        ax.set_xlabel('Action')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(action_counts)))
        ax.set_xticklabels(['Fold', 'Call', 'Min Raise', 'Pot Raise', 'All-in'])
        
        # 6. Bluff Frequency
        ax = axes[2, 1]
        bluff_smooth = pd.Series(self.bluff_frequencies).rolling(self.window_size).mean()
        ax.plot(bluff_smooth, color='orange')
        ax.set_title('Bluff Frequency')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Frequency')
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(self.plots_dir, f'training_curves_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        
    def plot_hand_analysis(self):
        """Generate detailed hand strength and betting analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Poker Hand Analysis', fontsize=16)
        
        # 1. Hand Strength Distribution
        ax = axes[0, 0]
        sns.histplot(data=self.hand_strengths, bins=20, ax=ax)
        ax.set_title('Hand Strength Distribution')
        ax.set_xlabel('Hand Strength')
        ax.set_ylabel('Count')
        
        # 2. Bet Sizing vs Hand Strength
        ax = axes[0, 1]
        ax.scatter(self.hand_strengths, self.bet_sizes, alpha=0.5)
        ax.set_title('Bet Sizing vs Hand Strength')
        ax.set_xlabel('Hand Strength')
        ax.set_ylabel('Bet Size (BB)')
        
        # 3. Stack Size Evolution
        ax = axes[1, 0]
        stack_smooth = pd.Series(self.stack_sizes).rolling(self.window_size).mean()
        ax.plot(stack_smooth, color='green')
        ax.set_title('Stack Size Evolution')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Stack Size (BB)')
        
        # 4. Showdown Win Rate vs Hand Strength
        ax = axes[1, 1]
        showdown_df = pd.DataFrame({
            'hand_strength': self.hand_strengths,
            'showdown_win': self.showdown_win_rates
        })
        strength_bins = pd.qcut(showdown_df['hand_strength'], q=10)
        win_rates = showdown_df.groupby(strength_bins)['showdown_win'].mean()
        ax.bar(range(len(win_rates)), win_rates.values)
        ax.set_title('Showdown Win Rate by Hand Strength')
        ax.set_xlabel('Hand Strength Decile')
        ax.set_ylabel('Win Rate')
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(self.plots_dir, f'hand_analysis_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        
    def save_metrics(self):
        """Save metrics to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_data = {
            'episode_rewards': self.episode_rewards,
            'win_rates': self.win_rates,
            'nash_distances': self.nash_distances,
            'loss_history': self.loss_history,
            'epsilon_history': self.epsilon_history,
            'bluff_frequencies': self.bluff_frequencies,
            'showdown_win_rates': self.showdown_win_rates,
            'training_duration': str(datetime.now() - self.start_time),
            'final_stats': {
                'avg_reward': np.mean(self.rolling_rewards),
                'avg_win_rate': np.mean(self.rolling_wins),
                'avg_nash_distance': np.mean(self.rolling_nash)
            }
        }
        
        metrics_path = os.path.join(self.metrics_dir, f'metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
            
    def get_current_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        return {
            'avg_reward': np.mean(self.rolling_rewards) if self.rolling_rewards else 0,
            'avg_win_rate': np.mean(self.rolling_wins) if self.rolling_wins else 0,
            'avg_nash_distance': np.mean(self.rolling_nash) if self.rolling_nash else 0,
            'total_episodes': len(self.episode_rewards),
            'training_duration': str(datetime.now() - self.start_time)
        }
