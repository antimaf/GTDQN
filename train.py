import torch
import numpy as np
from models.poker_dqn import PokerDQN
from utils.replay_buffer import ReplayBuffer
from environments.poker.poker_env import NoLimitHoldemEnv
import time
from typing import List, Tuple, Dict
import psutil
import os
import random
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PokerTrainer:
    """Trainer class for GT-DQN poker AI with Nash equilibrium mixing"""
    
    def __init__(
        self,
        num_actions: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "checkpoints"
    ):
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize networks
        self.policy_net = PokerDQN(
            num_actions=num_actions,
            lstm_hidden_size=128,
            lstm_num_layers=2,
            dropout=0.1
        ).to(device)
        
        self.target_net = PokerDQN(
            num_actions=num_actions,
            lstm_hidden_size=128,
            lstm_num_layers=2,
            dropout=0.1
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = Adam(self.policy_net.parameters(), lr=0.0001)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Training metrics
        self.episode_rewards = []
        self.win_rates = []
        self.nash_distances = []
        
    def optimize_model(
        self,
        batch_size: int,
        gamma: float
    ) -> float:
        """Perform one step of optimization"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
            
        # Sample transitions
        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors and move to device
        state_batch = {k: torch.cat([s[k] for s in batch.state]).to(self.device) 
                      for k in batch.state[0].keys()}
        action_batch = torch.tensor(batch.action).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = {k: torch.cat([s[k] for s in batch.next_state]).to(self.device)
                           for k in batch.next_state[0].keys()}
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values, _, _ = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_q_values, next_nash_probs, _ = self.target_net(next_state_batch)
            
            # Use Nash probabilities to compute expected future value
            next_state_values = (next_q_values * next_nash_probs).sum(dim=1).unsqueeze(1)
            
            # Mask for terminal states
            next_state_values = next_state_values * (1 - done_batch.unsqueeze(1))
            
        # Compute the expected Q values
        expected_state_action_values = reward_batch.unsqueeze(1) + (gamma * next_state_values)
        
        # Compute loss
        loss = smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
        
    def train(
        self,
        num_episodes: int = 10000,
        batch_size: int = 32,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update: int = 10,
        print_freq: int = 100,
        save_freq: int = 1000
    ):
        """Train the poker agent"""
        print("\nStarting poker training...")
        print(f"Device: {self.device}")
        print(f"Episodes: {num_episodes}")
        
        # Create environment
        env = NoLimitHoldemEnv(num_players=6)
        epsilon = epsilon_start
        
        # Training loop
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            hidden_state = None
            
            # Convert initial state to tensors
            state = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k,v in state.items()}
            
            while True:
                # Select action
                valid_actions = env.get_valid_actions()
                action, hidden_state = self.policy_net.select_action(
                    state,
                    epsilon=epsilon,
                    hidden_state=hidden_state,
                    valid_actions=valid_actions
                )
                
                # Execute action
                next_state, reward, done, _, _ = env.step(action.item())
                total_reward += reward
                
                # Convert next_state to tensors
                next_state = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) 
                            for k,v in next_state.items()}
                
                # Store transition
                self.replay_buffer.push(state, action.item(), reward, next_state, done)
                
                # Move to next state
                state = next_state
                
                # Optimize model
                loss = self.optimize_model(batch_size, gamma)
                
                if done:
                    break
                    
            # Update metrics
            self.episode_rewards.append(total_reward)
            
            # Update target network
            if episode % target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Print progress
            if episode % print_freq == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"\nEpisode {episode}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Epsilon: {epsilon:.3f}")
                print(f"Memory: {len(self.replay_buffer)}")
                
            # Save model
            if episode % save_freq == 0:
                self.save_checkpoint(episode)
                
        print("\nTraining complete!")
        
    def save_checkpoint(self, episode: int):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'win_rates': self.win_rates,
            'nash_distances': self.nash_distances
        }
        
        path = os.path.join(self.save_dir, f'checkpoint_{episode}.pt')
        torch.save(checkpoint, path)
        print(f"\nSaved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.episode_rewards = checkpoint['episode_rewards']
        self.win_rates = checkpoint['win_rates']
        self.nash_distances = checkpoint['nash_distances']
        
        print(f"\nLoaded checkpoint from {path}")
        return checkpoint['episode']
