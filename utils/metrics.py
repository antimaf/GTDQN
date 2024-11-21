import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import os
import json
from collections import deque
import torch

class NashConvergence:
    """Measure convergence to Nash equilibrium"""
    
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.reset()
    
    def reset(self):
        """Reset convergence metrics"""
        self.strategy_profile = torch.ones(self.num_actions) / self.num_actions
        self.exploitability = 0.0
        self.iterations = 0
    
    def update(self, current_strategy: torch.Tensor, best_response: torch.Tensor) -> float:
        """
        Update Nash convergence metrics
        
        Args:
            current_strategy: Current strategy profile
            best_response: Best response strategy
            
        Returns:
            float: Current exploitability measure
        """
        # Update average strategy
        self.iterations += 1
        alpha = 1.0 / self.iterations
        self.strategy_profile = (1 - alpha) * self.strategy_profile + alpha * current_strategy
        
        # Compute exploitability
        self.exploitability = torch.sum(best_response * current_strategy)
        
        return self.exploitability

class StackEfficiency:
    """Track effective stack utilization"""
    
    def __init__(self, initial_stack: float):
        self.initial_stack = initial_stack
        self.reset()
    
    def reset(self):
        """Reset efficiency metrics"""
        self.current_stack = self.initial_stack
        self.max_stack = self.initial_stack
        self.min_stack = self.initial_stack
        self.stack_history = []
    
    def update(self, stack_size: float) -> Tuple[float, float]:
        """
        Update stack efficiency metrics
        
        Args:
            stack_size: Current stack size
            
        Returns:
            Tuple[float, float]: (efficiency ratio, volatility)
        """
        self.current_stack = stack_size
        self.max_stack = max(self.max_stack, stack_size)
        self.min_stack = min(self.min_stack, stack_size)
        self.stack_history.append(stack_size)
        
        # Compute efficiency metrics
        efficiency_ratio = (self.current_stack - self.initial_stack) / self.initial_stack
        volatility = np.std(self.stack_history) / self.initial_stack if len(self.stack_history) > 1 else 0.0
        
        return efficiency_ratio, volatility

class BluffAnalyzer:
    """Analyze bluffing patterns and frequencies"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset bluff analysis metrics"""
        self.bluff_history = deque(maxlen=self.window_size)
        self.successful_bluffs = deque(maxlen=self.window_size)
        self.total_hands = 0
        self.total_bluffs = 0
        self.successful_bluff_count = 0
    
    def update(self, is_bluff: bool, won_hand: bool, hand_strength: float) -> Dict[str, float]:
        """
        Update bluff analysis metrics
        
        Args:
            is_bluff: Whether the hand was played as a bluff
            won_hand: Whether the hand was won
            hand_strength: Strength of the hand
            
        Returns:
            Dict[str, float]: Current bluff metrics
        """
        self.total_hands += 1
        
        if is_bluff:
            self.total_bluffs += 1
            self.bluff_history.append(hand_strength)
            if won_hand:
                self.successful_bluff_count += 1
                self.successful_bluffs.append(hand_strength)
        
        # Compute metrics
        bluff_frequency = self.total_bluffs / max(1, self.total_hands)
        success_rate = self.successful_bluff_count / max(1, self.total_bluffs)
        avg_bluff_strength = np.mean(list(self.bluff_history)) if self.bluff_history else 0.0
        
        return {
            'bluff_frequency': bluff_frequency,
            'success_rate': success_rate,
            'avg_bluff_strength': avg_bluff_strength
        }

class PokerMetricsTracker:
    """Comprehensive metrics tracking for poker AI training"""
    
    def __init__(
        self,
        save_dir: str,
        num_actions: int,
        initial_stack: float,
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
        
        # Initialize specialized metrics trackers
        self.nash_convergence = NashConvergence(num_actions)
        self.stack_efficiency = StackEfficiency(initial_stack)
        self.bluff_analyzer = BluffAnalyzer(window_size)
        
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
        self.bluff_metrics = []
        self.stack_metrics = []
        
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
        """Update all metrics with new episode data"""
        # Update specialized metrics
        nash_dist = self.nash_convergence.update(
            metrics.get('current_strategy', None),
            metrics.get('best_response', None)
        )
        
        stack_ratio, volatility = self.stack_efficiency.update(
            metrics.get('stack_size', self.stack_efficiency.initial_stack)
        )
        
        bluff_metrics = self.bluff_analyzer.update(
            metrics.get('is_bluff', False),
            metrics.get('won', False),
            metrics.get('hand_strength', 0.0)
        )
        
        # Store metrics
        self.episode_rewards.append(metrics.get('reward', 0))
        self.win_rates.append(metrics.get('won', False))
        self.nash_distances.append(nash_dist)
        self.loss_history.append(metrics.get('loss', 0))
        self.epsilon_history.append(metrics.get('epsilon', 0))
        self.action_distributions.append(metrics.get('action_dist', {}))
        self.bet_sizes.append(metrics.get('bet_size', 0))
        self.bluff_metrics.append(bluff_metrics)
        self.stack_metrics.append({
            'efficiency': stack_ratio,
            'volatility': volatility
        })
        
        # Update rolling statistics
        self.rolling_rewards.append(metrics.get('reward', 0))
        self.rolling_wins.append(metrics.get('won', False))
        self.rolling_nash.append(nash_dist)
        
    def plot_training_curves(self):
        """Generate comprehensive training curve plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        fig.suptitle('Poker AI Training Metrics', fontsize=16)
        
        # 1. Rewards and Win Rate
        self._plot_rewards_and_wins(axes[0, 0], axes[0, 1])
        
        # 2. Nash Convergence and Stack Efficiency
        self._plot_nash_and_stack(axes[1, 0], axes[1, 1])
        
        # 3. Bluff Analysis
        self._plot_bluff_metrics(axes[2, 0], axes[2, 1])
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'training_curves_{timestamp}.png'))
        plt.close()
    
    def _plot_rewards_and_wins(self, ax1: plt.Axes, ax2: plt.Axes):
        """Plot rewards and win rate metrics"""
        # Rewards
        rewards_smooth = pd.Series(self.episode_rewards).rolling(self.window_size).mean()
        ax1.plot(self.episode_rewards, alpha=0.3, color='blue', label='Raw')
        ax1.plot(rewards_smooth, color='blue', label=f'{self.window_size}-ep Moving Avg')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        
        # Win Rate
        win_rate_smooth = pd.Series(self.win_rates).rolling(self.window_size).mean()
        ax2.plot(win_rate_smooth, color='green')
        ax2.set_title('Win Rate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate')
    
    def _plot_nash_and_stack(self, ax1: plt.Axes, ax2: plt.Axes):
        """Plot Nash convergence and stack efficiency metrics"""
        # Nash Convergence
        nash_smooth = pd.Series(self.nash_distances).rolling(self.window_size).mean()
        ax1.plot(nash_smooth, color='red')
        ax1.set_title('Nash Convergence')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Exploitability')
        
        # Stack Efficiency
        efficiency = [m['efficiency'] for m in self.stack_metrics]
        volatility = [m['volatility'] for m in self.stack_metrics]
        ax2.plot(efficiency, color='purple', label='Efficiency')
        ax2.plot(volatility, color='orange', label='Volatility')
        ax2.set_title('Stack Efficiency')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Ratio')
        ax2.legend()
    
    def _plot_bluff_metrics(self, ax1: plt.Axes, ax2: plt.Axes):
        """Plot bluff analysis metrics"""
        # Bluff Frequency and Success Rate
        freq = [m['bluff_frequency'] for m in self.bluff_metrics]
        success = [m['success_rate'] for m in self.bluff_metrics]
        ax1.plot(freq, color='brown', label='Frequency')
        ax1.plot(success, color='gold', label='Success Rate')
        ax1.set_title('Bluff Analysis')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Rate')
        ax1.legend()
        
        # Average Bluff Hand Strength
        strength = [m['avg_bluff_strength'] for m in self.bluff_metrics]
        ax2.plot(strength, color='teal')
        ax2.set_title('Avg Bluff Hand Strength')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Strength')
