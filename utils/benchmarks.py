import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class EpisodeMetrics:
    """Stores metrics for a single training episode"""
    total_reward: float
    episode_length: int
    exploration_rate: float
    avg_q_value: float
    nash_distance: float  # Distance from Nash equilibrium
    bluff_frequency: float
    bet_sizes: List[float]
    win: bool
    profit: float
    mse_loss: float
    duration: float

class GTDQNBenchmarker:
    """Benchmarking system for GT-DQN poker AI"""
    
    def __init__(self, save_dir: str = "benchmarks"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training metrics
        self.episode_metrics: List[EpisodeMetrics] = []
        self.training_start_time = None
        self.epoch_times: List[float] = []
        
        # Performance metrics
        self.win_rates: Dict[str, List[float]] = defaultdict(list)
        self.expected_values: Dict[str, List[float]] = defaultdict(list)
        self.nash_convergence: List[float] = []
        
        # Strategy metrics
        self.bluff_success_rate: List[float] = []
        self.bet_sizing_distribution: List[List[float]] = []
        self.opponent_adaptation_score: List[float] = []
        
        # Computational metrics
        self.memory_usage: List[float] = []
        self.inference_times: List[float] = []
        
    def start_training_session(self):
        """Mark the start of a training session"""
        self.training_start_time = time.time()
        
    def log_episode(self, metrics: EpisodeMetrics):
        """Log metrics for a single episode"""
        self.episode_metrics.append(metrics)
        
    def log_win_rate(self, opponent_type: str, win_rate: float):
        """Log win rate against specific opponent type"""
        self.win_rates[opponent_type].append(win_rate)
        
    def log_expected_value(self, opponent_type: str, ev: float):
        """Log expected value against specific opponent type"""
        self.expected_values[opponent_type].append(ev)
        
    def log_nash_distance(self, distance: float):
        """Log distance from Nash equilibrium"""
        self.nash_convergence.append(distance)
        
    def log_computational_metrics(self, memory_mb: float, inference_time: float):
        """Log computational performance metrics"""
        self.memory_usage.append(memory_mb)
        self.inference_times.append(inference_time)
        
    def plot_learning_curve(self) -> None:
        """Plot the learning curve showing rewards over episodes"""
        rewards = [m.total_reward for m in self.episode_metrics]
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, label='Episode Reward')
        plt.title('Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.savefig(self.save_dir / 'learning_curve.png')
        plt.close()
        
    def plot_win_rates(self) -> None:
        """Plot win rates against different opponent types"""
        plt.figure(figsize=(10, 6))
        for opponent, rates in self.win_rates.items():
            plt.plot(rates, label=f'{opponent}')
        plt.title('Win Rates vs Different Opponents')
        plt.xlabel('Game Number')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.savefig(self.save_dir / 'win_rates.png')
        plt.close()
        
    def plot_nash_convergence(self) -> None:
        """Plot convergence to Nash equilibrium"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.nash_convergence)
        plt.title('Convergence to Nash Equilibrium')
        plt.xlabel('Episode')
        plt.ylabel('Distance from Nash Equilibrium')
        plt.savefig(self.save_dir / 'nash_convergence.png')
        plt.close()
        
    def plot_bet_distribution(self) -> None:
        """Plot distribution of bet sizes"""
        flat_bets = [bet for m in self.episode_metrics for bet in m.bet_sizes]
        plt.figure(figsize=(10, 6))
        sns.histplot(flat_bets, bins=50)
        plt.title('Bet Size Distribution')
        plt.xlabel('Bet Size (as fraction of pot)')
        plt.ylabel('Frequency')
        plt.savefig(self.save_dir / 'bet_distribution.png')
        plt.close()
        
    def generate_report(self) -> Dict:
        """Generate a comprehensive performance report"""
        if not self.episode_metrics:
            return {}
            
        recent_episodes = self.episode_metrics[-100:]  # Last 100 episodes
        
        report = {
            'training_metrics': {
                'total_episodes': len(self.episode_metrics),
                'avg_reward_last_100': np.mean([m.total_reward for m in recent_episodes]),
                'avg_episode_length': np.mean([m.episode_length for m in recent_episodes]),
                'avg_exploration_rate': np.mean([m.exploration_rate for m in recent_episodes]),
                'avg_q_value': np.mean([m.avg_q_value for m in recent_episodes]),
                'avg_mse_loss': np.mean([m.mse_loss for m in recent_episodes]),
            },
            'performance_metrics': {
                'win_rates': {k: np.mean(v[-100:]) for k, v in self.win_rates.items()},
                'expected_values': {k: np.mean(v[-100:]) for k, v in self.expected_values.items()},
                'nash_convergence': np.mean(self.nash_convergence[-100:]),
            },
            'strategy_metrics': {
                'bluff_frequency': np.mean([m.bluff_frequency for m in recent_episodes]),
                'avg_bet_size': np.mean([np.mean(m.bet_sizes) for m in recent_episodes if m.bet_sizes]),
            },
            'computational_metrics': {
                'avg_memory_usage_mb': np.mean(self.memory_usage[-100:]),
                'avg_inference_time_ms': np.mean(self.inference_times[-100:]) * 1000,
                'total_training_time': time.time() - self.training_start_time if self.training_start_time else 0,
            }
        }
        
        return report
    
    def save_plots(self) -> None:
        """Generate and save all benchmark plots"""
        self.plot_learning_curve()
        self.plot_win_rates()
        self.plot_nash_convergence()
        self.plot_bet_distribution()
        
    def save_report(self) -> None:
        """Save benchmark report to file"""
        report = self.generate_report()
        import json
        with open(self.save_dir / 'benchmark_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
class BaselineAgent:
    """Base class for baseline poker agents"""
    def __init__(self, strategy_type: str):
        self.strategy_type = strategy_type
        
    def act(self, state: np.ndarray) -> Tuple[int, float]:
        raise NotImplementedError
        
class RandomAgent(BaselineAgent):
    """Random baseline agent"""
    def __init__(self):
        super().__init__("random")
        
    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """Return random action and bet size"""
        action = np.random.randint(0, 3)  # fold, call, raise
        bet_size = np.random.random() if action == 2 else 0
        return action, bet_size
        
class RuleBasedAgent(BaselineAgent):
    """Simple rule-based poker agent"""
    def __init__(self, style: str = "tight_aggressive"):
        super().__init__(f"rule_based_{style}")
        self.style = style
        
    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Implement basic poker strategy based on hand strength
        Returns action and bet size
        """
        hand_strength = self._evaluate_hand_strength(state)
        
        if self.style == "tight_aggressive":
            if hand_strength > 0.7:  # Strong hand
                return 2, 1.0  # Raise max
            elif hand_strength > 0.4:  # Medium hand
                return 1, 0.0  # Call
            else:
                return 0, 0.0  # Fold
        else:  # loose_passive
            if hand_strength > 0.3:  # More willing to play hands
                return 1, 0.0  # Call
            else:
                return 0, 0.0  # Fold
                
    def _evaluate_hand_strength(self, state: np.ndarray) -> float:
        """Simplified hand strength evaluation"""
        # In practice, implement proper poker hand evaluation
        return np.random.random()  # Placeholder
