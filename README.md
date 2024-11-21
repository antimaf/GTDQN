# GT-DQN: Game-Theoretic Deep Q-Networks for No-Limit Texas Hold'em

---

## Abstract

This project presents **GT-DQN**, a novel deep reinforcement learning framework that integrates game-theoretic principles with deep Q-learning for No-Limit Texas Hold'em poker. The approach combines **counterfactual regret minimisation** with **value-based learning** to develop optimal strategies in imperfect information environments.

---

## 1. Introduction

### 1.1 Background

No-Limit Texas Hold'em (NLTH) poker represents one of the most challenging domains in artificial intelligence due to its:
1. **Imperfect information nature**: Players cannot observe opponents' cards.
2. **Large state and action spaces**: Exponentially growing possibilities with each move.
3. **Need for mixed strategies**: Optimal play requires balancing aggressive and conservative actions.
4. **Complex opponent modelling requirements**: Players must predict and counter diverse strategies.

Recent breakthroughs in deep reinforcement learning (DRL) have shown promise in complex games such as Go and Chess, but traditional DRL algorithms struggle with poker’s unique challenges. **GT-DQN** aims to bridge the gap between deep Q-learning and game-theoretic principles by introducing novel mechanisms tailored to NLTH poker.

### 1.2 Key Contributions

1. Integration of **counterfactual regret minimisation (CFR)** with deep Q-learning.
2. Novel neural architecture for **poker state representation**.
3. Efficient **experience replay** mechanisms for imperfect information environments.
4. A **theoretically-grounded approach** to approximate Nash equilibria.

---

## 2. Technologies and Tools

### 2.1 Core Technologies

- **PyTorch (>=1.9.0)**: Primary deep learning framework used for implementing neural networks and GPU acceleration
- **Python (>=3.8)**: Main programming language
- **NumPy (>=1.19.5)**: Numerical computing library for efficient array operations
- **Gymnasium (>=0.26.0)**: Framework for reinforcement learning environments
- **TensorBoard (>=2.6.0)**: Visualisation tool for training metrics and model performance

### 2.2 Data Processing and Visualisation

- **Pandas (>=1.3.0)**: Data manipulation and analysis
- **Matplotlib (>=3.4.3)** and **Seaborn (>=0.11.2)**: Data visualisation and plotting
- **SciPy (>=1.7.0)**: Scientific computing tools for advanced mathematical operations

### 2.3 Development Tools

- **pytest (>=6.2.5)**: Testing framework for unit and integration tests
- **mypy (>=0.910)**: Static type checker for Python code
- **black (>=21.9b0)**: Code formatter for maintaining consistent code style
- **psutil (>=5.8.0)**: System monitoring and resource management

### 2.4 Key Features

1. **GPU Acceleration**: Utilises CUDA-enabled GPUs through PyTorch for faster training
2. **Distributed Training**: Supports multi-GPU training for larger models
3. **Real-time Monitoring**: Integration with TensorBoard for live training visualisation
4. **Type Safety**: Comprehensive type hints and static type checking
5. **Testing**: Automated test suite for model validation and regression testing

### 2.5 System Requirements

- **Python**: Version 3.8 or higher
- **CUDA**: Optional but recommended for GPU acceleration
- **RAM**: Minimum 16GB recommended for training
- **Storage**: At least 1GB for model checkpoints and training data
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

---

## 3. Theoretical Framework

### 3.1 Game-Theoretic Foundations

In an extensive-form game $\Gamma$, the following terms are defined:
- $\mathcal{H}$: Set of all game histories.
- $\mathcal{I}$: Set of information sets (indistinguishable histories).
- $\mathcal{A}(I)$: Legal actions at information set $I$.
- $\sigma_i$: Strategy for player $i$.
- $\pi^\sigma(h)$: Reach probability of history $h$ under strategy profile $\sigma$.

The **counterfactual value** for player $i$ at information set $I$ is given by:

$$
v_i^\sigma(I) = \sum_{h \in I} \pi_{-i}^\sigma(h) \sum_{z \in Z_h} \pi^\sigma(h, z) u_i(z)
$$

where:
- $Z_h$: Set of terminal histories reachable from $h$.
- $u_i(z)$: Utility for player $i$ at terminal history $z$.

### 3.2 Deep Q-Learning Integration

We modify the Q-learning update rule to incorporate mixed strategies:

$$
Q(s, a) \gets Q(s, a) + \alpha \left[ r + \gamma \sum_{a'} \pi(s', a') Q(s', a') - Q(s, a) \right]
$$

Here, the policy $\pi(s', a')$ is derived via **regret-matching**:

$$
\pi(s, a) = \frac{R^T_+(s, a)}{\sum_{a'} R^T_+(s, a')}
$$

Cumulative regret is updated as:

$$
R^{T+1}(s, a) = R^T(s, a) + \left(Q(s, a) - \sum_{a'} \pi^T(s, a') Q(s, a')\right)
$$

### 3.3 Information State Representation

The poker state $s_t$ at time $t$ is encoded as:

$$
s_t = \begin{bmatrix}
C_h & C_c & p_t & \mathbf{b}_t & \mathbf{v}_t & \mathbf{p}_t
\end{bmatrix}
$$

where:
- $C_h \in \{0,1\}^{2 \times 52}$: **Hole cards** encoding.
- $C_c \in \{0,1\}^{5 \times 52}$: **Community cards** encoding.
- $p_t \in \mathbb{R}$: Normalised **pot size**.
- $\mathbf{b}_t \in \mathbb{R}^{n}$: **Betting history** vector.
- $\mathbf{v}_t \in \mathbb{R}^{6}$: **Stack size** vector.
- $\mathbf{p}_t \in \{0,1\}^6$: **Position encoding**.

### 3.4 Value Decomposition

Following Wang et al. (2016), we decompose the action-value function as:

$$
Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\right)
$$

This **dueling architecture** separately estimates:
- **State value** $V(s)$.
- **Action advantage** $A(s, a)$.

---

## 4. Neural Architecture

### 4.1 Card Processing Networks

```python
class CardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.view(x.size(0), -1)


### 4.2 Sequential Processing
Betting history is processed using LSTM with attention:

```python
class HistoryProcessor(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for processing betting history
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1
        )
        
        # Layer normalisation
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, x, mask=None):
        """
        Process betting history with LSTM and attention
        
        Args:
            x (torch.Tensor): Betting history [batch_size, seq_len, hidden_size]
            mask (torch.Tensor): Attention mask for padding
            
        Returns:
            torch.Tensor: Processed history representation
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Self-attention mechanism
        # Reshape for attention [seq_len, batch_size, hidden_size]
        lstm_out = lstm_out.transpose(0, 1)
        attended, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=mask,
            need_weights=False
        )
        
        # Residual connection and layer norm
        attended = self.layer_norm1(lstm_out + attended)
        
        # Position-wise FFN
        ffn_out = self.ffn(attended)
        
        # Final residual connection and layer norm
        output = self.layer_norm2(attended + ffn_out)
        
        # Reshape back to [batch_size, seq_len, hidden_size]
        output = output.transpose(0, 1)
        
        return output

class ActionEncoder(nn.Module):
    def __init__(self, num_actions, hidden_size=128):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size, max_len=50)
        
    def forward(self, actions):
        """
        Encode action sequence with positional information
        
        Args:
            actions (torch.Tensor): Action indices [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Encoded actions with positional information
        """
        # Embed actions
        embedded = self.action_embedding(actions)
        
        # Add positional encoding
        encoded = self.position_encoding(embedded)
        
        return encoded

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=50):
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        
        pe = torch.zeros(max_len, 1, hidden_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:x.size(0)]

### 4.3 Value Networks
Implementation of the dueling architecture for value estimation:

```python
class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[512, 256]):
        super().__init__()
        
        # Common feature network
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(0.1)
        )
        
        # Value stream
        self.value_net = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[1], 1)
        )
        
        # Advantage stream
        self.advantage_net = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[1], action_dim)
        )
        
        # Game-theoretic layer
        self.gt_layer = GTLayer(hidden_sizes[1], action_dim)
        
        # Initialise weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialise network weights using He initialisation
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state, return_features=False):
        """
        Forward pass through the dueling network
        
        Args:
            state (torch.Tensor): Input state [batch_size, state_dim]
            return_features (bool): Whether to return intermediate features
            
        Returns:
            torch.Tensor: Q-values for each action
            torch.Tensor: Feature representations (if return_features=True)
        """
        # Extract features
        features = self.feature_net(state)
        
        # Compute value and advantage
        value = self.value_net(features)
        advantage = self.advantage_net(features)
        
        # Apply game-theoretic constraints
        gt_advantage = self.gt_layer(advantage)
        
        # Combine value and advantage (dueling architecture)
        q_values = value + gt_advantage - gt_advantage.mean(dim=1, keepdim=True)
        
        if return_features:
            return q_values, features
        return q_values

class GTLayer(nn.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        
        # Strategic reasoning module
        self.strategic_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        
        # Counterfactual value estimator
        self.counterfactual_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, advantage):
        """
        Apply game-theoretic constraints to advantage values
        
        Args:
            advantage (torch.Tensor): Raw advantage values [batch_size, action_dim]
            
        Returns:
            torch.Tensor: Constrained advantage values
        """
        # Strategic adjustment
        strategic_factor = self.strategic_net(advantage)
        
        # Counterfactual correction
        counterfactual_values = self.counterfactual_net(advantage)
        
        # Combine components
        adjusted_advantage = advantage * (1 + strategic_factor) + counterfactual_values
        
        return adjusted_advantage

## 5. Training Framework

### 5.1 Experience Collection
We use prioritised experience replay [Schaul et al., 2016] with importance sampling:

$$
p_i = |\delta_i|^\alpha \cdot \left(\frac{1}{N}\right)^\beta
$$

where:
- $\delta_i$: TD-error for transition $i$
- $\alpha$: Priority exponent
- $\beta$: Importance sampling exponent
- $N$: Buffer size

Implementation:
```python
class PrioritisedReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        
    def add(self, transition, error):
        priority = (abs(error) + 1e-5) ** self.alpha
        self.tree.add(priority, transition)
        self.max_priority = max(self.max_priority, priority)
```

### 5.2 Loss Function
The network is trained to minimise:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ w_i(r + \gamma \max_{a'} Q_{\theta'}(s', a') - Q_\theta(s, a))^2 \right]
$$

This loss function represents the weighted mean squared Bellman error where:
- $\theta$: Parameters of the current network
- $\theta'$: Parameters of the target network
- $\mathcal{D}$: Experience replay buffer
- $w_i$: Importance sampling weights
- $r$: Immediate reward
- $\gamma$: Discount factor
- $Q_\theta(s,a)$: Current network's Q-value estimate
- $\max_{a'} Q_{\theta'}(s', a')$: Target network's maximum Q-value for next state

The importance sampling weights $w_i$ are computed as:

$$
w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta
$$

where:
- $N$: Replay buffer size
- $P(i)$: Priority probability of transition $i$
- $\beta$: Importance sampling exponent (annealed from initial value to 1)

Implementation:
```python
class GTDQNLoss(nn.Module):
    def __init__(self, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, current_q, target_q, rewards, done, importance_weights):
        """
        Calculate the weighted TD loss for GT-DQN
        
        Args:
            current_q (torch.Tensor): Q-values from current network
            target_q (torch.Tensor): Q-values from target network
            rewards (torch.Tensor): Immediate rewards
            done (torch.Tensor): Binary tensor indicating terminal states
            importance_weights (torch.Tensor): Weights for prioritised replay
            
        Returns:
            torch.Tensor: Weighted TD loss
        """
        # Calculate TD target
        next_q = target_q.max(dim=1)[0].detach()
        td_target = rewards + self.gamma * next_q * (1 - done)
        
        # Calculate TD error
        td_error = td_target - current_q
        
        # Apply importance weights
        weighted_loss = importance_weights * td_error.pow(2)
        
        return weighted_loss.mean(), td_error.abs().detach()
```

### 5.3 Evaluation Metrics
The agent's performance is evaluated using several key metrics:

1. **Nash Convergence**: Measured as the average exploitability $\epsilon$ against best responses:
   $$
   \epsilon = \frac{1}{2}\sum_{i=1}^2 \max_{\sigma'_i} (u_i(\sigma'_i, \sigma_{-i}) - u_i(\sigma))
   $$

   where:
   - $\sigma'_i$: Best response strategy for player $i$
   - $\sigma_{-i}$: Fixed strategy of opponent
   - $u_i$: Utility function for player $i$

   Implementation:
   ```python
   def calculate_exploitability(agent, env):
       """
       Calculate the exploitability of an agent's strategy
       
       Args:
           agent: The GT-DQN agent
           env: Poker environment instance
           
       Returns:
           float: Average exploitability score
       """
       exploitability = 0
       for player in range(2):
           # Create best response agent
           br_agent = create_best_response(env, agent, player)
           
           # Simulate games
           total_utility = 0
           n_games = 1000
           
           for _ in range(n_games):
               state = env.reset()
               done = False
               while not done:
                   if env.current_player == player:
                       action = br_agent.act(state)
                   else:
                       action = agent.act(state)
                   state, reward, done, _ = env.step(action)
               total_utility += reward if player == 0 else -reward
           
           exploitability += max(0, total_utility / n_games)
       
       return exploitability / 2
   ```

2. **Expected Game Value**: Computed over N episodes:
   $$
   V = \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^T \gamma^t r_t^i
   $$

   where:
   - $N$: Number of episodes
   - $T$: Episode length
   - $\gamma$: Discount factor
   - $r_t^i$: Reward at time t in episode i

   Implementation:
   ```python
   def calculate_expected_value(agent, env, n_episodes=1000):
       """
       Calculate expected value over multiple episodes
       
       Args:
           agent: The GT-DQN agent
           env: Poker environment instance
           n_episodes: Number of episodes to evaluate
           
       Returns:
           float: Average episode return
       """
       total_return = 0
       
       for episode in range(n_episodes):
           state = env.reset()
           episode_return = 0
           done = False
           step = 0
           
           while not done:
               action = agent.act(state)
               next_state, reward, done, _ = env.step(action)
               episode_return += reward * (agent.gamma ** step)
               state = next_state
               step += 1
           
           total_return += episode_return
       
       return total_return / n_episodes
   ```

3. **Win Rate**: Against baseline agents (MCTS and CFR):
   $$
   WR_{agent} = \frac{\text{Games Won}}{\text{Total Games}} \times 100\%
   $$

   Implementation:
   ```python
   def calculate_win_rate(agent, opponent, env, n_games=1000):
       """
       Calculate win rate against a specific opponent
       
       Args:
           agent: The GT-DQN agent
           opponent: Opponent agent (e.g., MCTS or CFR agent)
           env: Poker environment instance
           n_games: Number of games to play
           
       Returns:
           float: Win rate percentage
       """
       wins = 0
       
       for game in range(n_games):
           state = env.reset()
           done = False
           
           while not done:
               if env.current_player == 0:
                   action = agent.act(state)
               else:
                   action = opponent.act(state)
               state, reward, done, _ = env.step(action)
           
           if reward > 0:  # Agent won
               wins += 1
       
       return (wins / n_games) * 100
   ```

4. **Average Stack Efficiency**: Measures effective stack utilisation:
   $$
   SE = \frac{1}{N}\sum_{i=1}^N \frac{\text{Final Stack}_i}{\text{Initial Stack}_i}
   $$

   where:
   - $N$: Number of episodes
   - $\text{Final Stack}_i$: Remaining chips after episode i
   - $\text{Initial Stack}_i$: Starting chips in episode i

   Implementation:
   ```python
   def calculate_stack_efficiency(agent, env, n_episodes=1000):
       """
       Calculate how efficiently the agent uses its stack
       
       Args:
           agent: The GT-DQN agent
           env: Poker environment instance
           n_episodes: Number of episodes to evaluate
           
       Returns:
           float: Average stack efficiency
       """
       total_efficiency = 0
       
       for _ in range(n_episodes):
           state = env.reset()
           initial_stack = env.player_stacks[0]
           done = False
           
           while not done:
               action = agent.act(state)
               state, _, done, _ = env.step(action)
           
           final_stack = env.player_stacks[0]
           episode_efficiency = final_stack / initial_stack
           total_efficiency += episode_efficiency
       
       return total_efficiency / n_episodes
   ```

5. **Bluff Frequency**: Ratio of successful bluffs to total bluff attempts:
   $$
   BF = \frac{\text{Successful Bluffs}}{\text{Total Bluff Attempts}}
   $$

   Implementation:
   ```python
   def calculate_bluff_frequency(agent, env, n_hands=1000):
       """
       Calculate the ratio of successful bluffs to total bluff attempts
       
       Args:
           agent: The GT-DQN agent
           env: Poker environment instance
           n_hands: Number of hands to evaluate
           
       Returns:
           float: Bluff success rate
       """
       total_bluffs = 0
       successful_bluffs = 0
       
       for _ in range(n_hands):
           state = env.reset()
           done = False
           bluffed_this_hand = False
           
           while not done:
               if env.current_player == 0:  # Agent's turn
                   action = agent.act(state)
                   # Check if action is a bluff
                   if is_bluff(env, action):
                       total_bluffs += 1
                       bluffed_this_hand = True
               
               state, reward, done, _ = env.step(action)
               
               if done and bluffed_this_hand and reward > 0:
                   successful_bluffs += 1
       
       return successful_bluffs / total_bluffs if total_bluffs > 0 else 0
   ```

### 5.4 Policy Extraction
Following [Heinrich & Silver, 2016], we use a softmax policy with temperature:

$$
\pi_\beta(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}
$$

where:
- $\tau$: Temperature parameter controlling exploration
- $Q(s,a)$: Action-value function
- $\pi_\beta$: Softmax policy

---

## 6. Implementation Details

### 6.1 Environment Configuration
```python
env_config = {
    'num_players': 6,
    'starting_stack': 10000,
    'small_blind': 50,
    'big_blind': 100,
    'max_rounds': 1000,
    'state_history_len': 10
}
```

### 6.2 Training Parameters
```python
training_params = {
    'num_episodes': 1e6,
    'batch_size': 128,
    'target_update': 1000,
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'alpha': 0.6,  # PER exponent
    'beta': 0.4,   # IS exponent
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995
}
```

### 6.3 Network Architecture
```python
network_config = {
    'card_channels': [64, 128, 256],
    'lstm_hidden': 128,
    'lstm_layers': 2,
    'advantage_hidden': [512, 256],
    'value_hidden': [512, 256],
    'dropout': 0.1
}
```

## References

[1] Brown, N., & Sandholm, T. (2018). Superhuman AI for heads-up no-limit poker: Libratus beats top professionals. Science, 359(6374), 418-424.

[2] Heinrich, J., & Silver, D. (2016). Deep reinforcement learning from self-play in imperfect-information games. arXiv preprint arXiv:1603.01121.

[3] Moravčík, M., Schmid, M., Burch, N., Lisý, V., Morrill, D., Bard, N., ... & Bowling, M. (2017). DeepStack: Expert-level artificial intelligence in heads-up no-limit poker. Science, 356(6337), 508-513.

[4] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritised experience replay. International Conference on Learning Representations.

[5] Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. International Conference on Machine Learning, 1995-2003.

[6] Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2008). Regret minimisation in games with incomplete information. Advances in Neural Information Processing Systems, 20.
