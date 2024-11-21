import numpy as np
import torch
from environments.bidding_game import BiddingGame
from models.gt_dqn import GTDQN
from models.gt_layer import GameTheoreticLayer, SelfPlayManager
from models.mcts_agent import MCTSAgent
from models.cfr_agent import CFRAgent
import json
from pathlib import Path
import websockets
import asyncio
import threading
from typing import Dict, List, Tuple

class ExperimentRunner:
    def __init__(
        self,
        num_episodes: int = 1000,
        num_agents: int = 3,
        eval_frequency: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ):
        self.num_episodes = num_episodes
        self.num_agents = num_agents
        self.eval_frequency = eval_frequency
        self.batch_size = batch_size
        
        # Initialize environment
        self.env = BiddingGame(num_agents=num_agents)
        
        # Initialize GT-DQN agent
        self.gt_dqn = GTDQN(
            input_channels=4,  # [private_value, highest_bid, cash, round]
            num_actions=100,   # Discretized action space
            lstm_hidden_size=128,
            conv_channels=[32, 64, 64],
            fc_units=[512, 256]
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.gt_dqn.parameters(), lr=learning_rate)
        
        # Initialize baseline agents
        self.mcts_agent = MCTSAgent(num_simulations=100)
        self.cfr_agent = CFRAgent(num_actions=100)
        
        # Experience replay buffer
        self.replay_buffer = []
        self.max_buffer_size = 10000
        
        # Metrics storage
        self.metrics = {
            'gt_dqn_vs_mcts_winrate': [],
            'gt_dqn_vs_cfr_winrate': [],
            'gt_dqn_profits': [],
            'mcts_profits': [],
            'cfr_profits': [],
            'nash_convergence': [],
            'episodes': [],
            'losses': []
        }
        
        # Websocket for real-time updates
        self.clients = set()
        
    async def start_websocket(self):
        async with websockets.serve(self.handle_client, 'localhost', 8765):
            await asyncio.Future()
            
    async def handle_client(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            
    async def broadcast_metrics(self):
        if self.clients:
            message = json.dumps(self.metrics)
            await asyncio.gather(
                *[client.send(message) for client in self.clients]
            )
            
    def train(self, num_training_episodes: int = 5000):
        """Train the GT-DQN agent."""
        print("\n=== Training GT-DQN ===")
        print("Episode | Loss    | Avg Reward")
        print("-" * 35)
        
        for episode in range(num_training_episodes):
            # Run training episode
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            transitions = []
            
            while not done:
                # Select action and step environment
                action = self.gt_dqn.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                
                # Store transition
                transitions.append((state, action, reward, next_state, done))
                state = next_state
                
            # Store episode transitions
            self._store_transitions(transitions)
            
            # Train on replay buffer
            loss = self._train_on_buffer()
            
            # Print progress
            if episode % 100 == 0:
                print(f"{episode:7d} | {loss:7.4f} | {episode_reward:10.2f}")
                
        print("\n=== Training Complete ===")
        
    def run_experiment(self, num_eval_episodes: int = 100):
        """Evaluate the trained GT-DQN agent."""
        print("\n=== Evaluating GT-DQN ===")
        print("Episode | vs MCTS | vs CFR  | Avg Profit")
        print("-" * 45)
        
        gt_dqn_vs_mcts_wins = 0
        gt_dqn_vs_cfr_wins = 0
        total_profit = 0
        
        for episode in range(num_eval_episodes):
            # Evaluate vs MCTS
            gt_dqn_reward, mcts_reward, _ = self._run_episode(self.gt_dqn, self.mcts_agent, training=False)
            if gt_dqn_reward > mcts_reward:
                gt_dqn_vs_mcts_wins += 1
            
            # Evaluate vs CFR
            gt_dqn_reward2, cfr_reward, _ = self._run_episode(self.gt_dqn, self.cfr_agent, training=False)
            if gt_dqn_reward2 > cfr_reward:
                gt_dqn_vs_cfr_wins += 1
                
            # Update metrics
            avg_profit = (gt_dqn_reward + gt_dqn_reward2) / 2
            total_profit += avg_profit
            
            if episode % 10 == 0:
                mcts_wr = gt_dqn_vs_mcts_wins / (episode + 1)
                cfr_wr = gt_dqn_vs_cfr_wins / (episode + 1)
                avg_total_profit = total_profit / (episode + 1)
                print(f"{episode:7d} | {mcts_wr:6.2%} | {cfr_wr:6.2%} | {avg_total_profit:10.2f}")
        
        # Final results
        mcts_wr = gt_dqn_vs_mcts_wins / num_eval_episodes
        cfr_wr = gt_dqn_vs_cfr_wins / num_eval_episodes
        avg_total_profit = total_profit / num_eval_episodes
        
        print("\n=== Final Results ===")
        print(f"Win Rate vs MCTS: {mcts_wr:6.2%}")
        print(f"Win Rate vs CFR:  {cfr_wr:6.2%}")
        print(f"Average Profit:   {avg_total_profit:10.2f}")
        
        # Update metrics
        self.metrics['gt_dqn_vs_mcts_winrate'].append(mcts_wr)
        self.metrics['gt_dqn_vs_cfr_winrate'].append(cfr_wr)
        self.metrics['gt_dqn_profits'].append(avg_total_profit)
        
        return self.metrics
        
    def _run_episode(self, agent1, agent2, training=False) -> Tuple[float, float, List]:
        """Run a single episode between two agents."""
        state, _ = self.env.reset()
        done = False
        agent1_reward = 0
        agent2_reward = 0
        transitions = []
        
        while not done:
            # Agent 1 turn
            action1 = agent1.select_action(state)
            next_state, reward, done, truncated, info = self.env.step(action1)
            agent1_reward += reward
            
            if training:
                transitions.append((state, action1, reward, next_state, done))
            
            if not done:
                # Agent 2 turn
                action2 = agent2.select_action(next_state)
                state, reward, done, truncated, info = self.env.step(action2)
                agent2_reward += reward
            
            if done:
                break
                
            state = next_state
            
        return agent1_reward, agent2_reward, transitions
        
    def _store_transitions(self, transitions: List):
        """Store transitions in replay buffer."""
        self.replay_buffer.extend(transitions)
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]
            
    def _train_on_buffer(self) -> float:
        """Train GT-DQN on replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
            
        # Sample batch
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        # Create spatial states
        batch_size = states.size(0)
        spatial_states = torch.zeros(batch_size, 4, 8, 8)
        spatial_next_states = torch.zeros(batch_size, 4, 8, 8)
        
        for i in range(4):
            spatial_states[:, i, :, :] = states[:, i].view(-1, 1, 1).expand(-1, 8, 8)
            spatial_next_states[:, i, :, :] = next_states[:, i].view(-1, 1, 1).expand(-1, 8, 8)
        
        # Get current Q values
        q_values, _, _ = self.gt_dqn(spatial_states)
        action_indices = (actions * 100).long()
        current_q = q_values.gather(1, action_indices.unsqueeze(1)).squeeze()
        
        # Get target Q values
        with torch.no_grad():
            next_q_values, _, _ = self.gt_dqn(spatial_next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * max_next_q
        
        # Compute loss and update
        loss = torch.nn.functional.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def _update_metrics(self, episode: int, loss: float):
        # Evaluate performance
        gt_dqn_vs_mcts_wins = 0
        gt_dqn_vs_cfr_wins = 0
        num_eval = 100
        
        for _ in range(num_eval):
            gt_dqn_reward, mcts_reward, _ = self._run_episode(self.gt_dqn, self.mcts_agent, training=False)
            if gt_dqn_reward > mcts_reward:
                gt_dqn_vs_mcts_wins += 1
                
            gt_dqn_reward, cfr_reward, _ = self._run_episode(self.gt_dqn, self.cfr_agent, training=False)
            if gt_dqn_reward > cfr_reward:
                gt_dqn_vs_cfr_wins += 1
                
        # Update metrics
        self.metrics['episodes'].append(episode)
        self.metrics['gt_dqn_vs_mcts_winrate'].append(gt_dqn_vs_mcts_wins / num_eval)
        self.metrics['gt_dqn_vs_cfr_winrate'].append(gt_dqn_vs_cfr_wins / num_eval)
        self.metrics['losses'].append(loss)
        
        # Broadcast updates
        asyncio.get_event_loop().run_until_complete(self.broadcast_metrics())

if __name__ == "__main__":
    # Create experiment runner
    runner = ExperimentRunner(
        num_agents=3,
        batch_size=64,
        learning_rate=1e-4
    )
    
    # Train GT-DQN
    runner.train(num_training_episodes=5000)
    
    # Evaluate GT-DQN
    metrics = runner.run_experiment(num_eval_episodes=100)
    
    # Save final metrics
    with open('visualization/data/metrics.json', 'w') as f:
        json.dump(metrics, f)
