import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StrategyProfile:
    """Represents a strategy profile in the game"""
    action_probs: torch.Tensor  # Probability distribution over actions
    regrets: torch.Tensor      # Cumulative regrets for actions
    avg_strategy: torch.Tensor  # Average strategy over time
    visit_count: int           # Number of times this profile was visited

class GameTheoreticLayer(nn.Module):
    """
    Game Theoretic Layer for GT-DQN
    Implements regret minimization and strategy adaptation
    """
    def __init__(
        self,
        num_actions: int,
        hidden_size: int = 128,
        learning_rate: float = 0.01,
        regret_discount: float = 0.99
    ):
        super(GameTheoreticLayer, self).__init__()
        
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.regret_discount = regret_discount
        
        # Strategy networks
        self.strategy_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        # Regret networks
        self.regret_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        # Strategy profiles for different information sets
        self.strategy_profiles: Dict[str, StrategyProfile] = {}
        
    def _get_info_set_key(self, state: torch.Tensor) -> str:
        """Generate key for information set from state"""
        # In practice, implement proper info set abstraction
        return str(state.cpu().numpy().tobytes())
        
    def _initialize_profile(self, info_set_key: str) -> StrategyProfile:
        """Initialize a new strategy profile"""
        return StrategyProfile(
            action_probs=torch.ones(self.num_actions) / self.num_actions,
            regrets=torch.zeros(self.num_actions),
            avg_strategy=torch.ones(self.num_actions) / self.num_actions,
            visit_count=0
        )
        
    def get_strategy(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor,
        q_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mixed strategy using regret minimization
        
        Args:
            state: Current state tensor
            hidden: Hidden state from LSTM
            q_values: Q-values from DQN
            
        Returns:
            action_probs: Probability distribution over actions
            regret_adjusted_q: Regret-adjusted Q-values
        """
        info_set_key = self._get_info_set_key(state)
        
        if info_set_key not in self.strategy_profiles:
            self.strategy_profiles[info_set_key] = self._initialize_profile()
            
        profile = self.strategy_profiles[info_set_key]
        
        # Compute regrets from Q-values
        regrets = self.regret_net(hidden)
        max_q = q_values.max()
        current_regrets = max_q - q_values
        
        # Update cumulative regrets
        profile.regrets = (
            self.regret_discount * profile.regrets +
            (1 - self.regret_discount) * current_regrets
        )
        
        # Compute strategy using regret matching
        positive_regrets = F.relu(profile.regrets)
        regret_sum = positive_regrets.sum()
        
        if regret_sum > 0:
            action_probs = positive_regrets / regret_sum
        else:
            action_probs = torch.ones_like(positive_regrets) / self.num_actions
            
        # Update average strategy
        profile.visit_count += 1
        profile.avg_strategy = (
            profile.avg_strategy * (profile.visit_count - 1) +
            action_probs
        ) / profile.visit_count
        
        # Compute regret-adjusted Q-values
        strategy_weights = self.strategy_net(hidden)
        regret_adjusted_q = q_values + strategy_weights
        
        return action_probs, regret_adjusted_q
        
    def update_regrets(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        opponent_action: int
    ) -> None:
        """
        Update regrets based on action outcomes
        
        Args:
            state: State where action was taken
            action: Action taken
            reward: Reward received
            opponent_action: Action taken by opponent
        """
        info_set_key = self._get_info_set_key(state)
        profile = self.strategy_profiles.get(info_set_key)
        
        if profile is None:
            return
            
        # Compute counterfactual value
        cf_value = reward * (1 / profile.action_probs[action].item())
        
        # Update regrets for all actions
        for a in range(self.num_actions):
            if a == action:
                continue
            # Compute regret as difference between counterfactual values
            regret = max(0, cf_value - reward)
            profile.regrets[a] += regret
            
    def get_nash_distance(self, state: torch.Tensor) -> float:
        """
        Compute approximate distance from Nash equilibrium
        
        Args:
            state: Current state
            
        Returns:
            Distance from Nash equilibrium (0 = Nash equilibrium)
        """
        info_set_key = self._get_info_set_key(state)
        profile = self.strategy_profiles.get(info_set_key)
        
        if profile is None:
            return float('inf')
            
        # Compute KL divergence between current and average strategy
        kl_div = F.kl_div(
            profile.action_probs.log(),
            profile.avg_strategy,
            reduction='sum'
        )
        
        return kl_div.item()
        
class SelfPlayManager:
    """Manages self-play training for GT-DQN"""
    
    def __init__(
        self,
        num_agents: int,
        model_class,
        model_args: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.num_agents = num_agents
        self.device = device
        
        # Initialize population of agents
        self.agents = [
            model_class(**model_args).to(device)
            for _ in range(num_agents)
        ]
        
        # Track agent performance
        self.agent_stats = {
            i: {"wins": 0, "total_games": 0}
            for i in range(num_agents)
        }
        
    def select_opponents(
        self,
        current_agent: int,
        num_opponents: int = 1
    ) -> List[int]:
        """Select opponents for self-play"""
        available_agents = [
            i for i in range(self.num_agents)
            if i != current_agent
        ]
        return np.random.choice(
            available_agents,
            size=min(num_opponents, len(available_agents)),
            replace=False
        ).tolist()
        
    def update_stats(
        self,
        agent_id: int,
        won: bool
    ) -> None:
        """Update agent statistics"""
        stats = self.agent_stats[agent_id]
        stats["total_games"] += 1
        if won:
            stats["wins"] += 1
            
    def get_win_rate(self, agent_id: int) -> float:
        """Get win rate for an agent"""
        stats = self.agent_stats[agent_id]
        if stats["total_games"] == 0:
            return 0.0
        return stats["wins"] / stats["total_games"]
        
    def evolve_population(
        self,
        tournament_size: int = 2,
        elite_fraction: float = 0.1
    ) -> None:
        """
        Evolve population using tournament selection
        Keeps top performers and replaces others
        """
        # Sort agents by win rate
        sorted_agents = sorted(
            range(self.num_agents),
            key=lambda x: self.get_win_rate(x),
            reverse=True
        )
        
        # Keep elite agents
        num_elite = max(1, int(self.num_agents * elite_fraction))
        elite_agents = sorted_agents[:num_elite]
        
        # Replace worst performing agents
        for i in sorted_agents[num_elite:]:
            # Select random elite agent to copy
            elite_idx = np.random.choice(elite_agents)
            # Copy parameters with small mutations
            self.agents[i].load_state_dict(
                self.agents[elite_idx].state_dict()
            )
            self._mutate_agent(self.agents[i])
            
    def _mutate_agent(self, agent, mutation_std: float = 0.1) -> None:
        """Add random mutations to agent parameters"""
        with torch.no_grad():
            for param in agent.parameters():
                noise = torch.randn_like(param) * mutation_std
                param.add_(noise)
