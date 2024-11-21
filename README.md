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
    def __init__(self, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        self.attention = MultiheadAttention(hidden_size, num_heads=4)
        
    def forward(self, x, mask=None):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=mask)
        return attn_out
```

### 4.3 Value Networks
Implementing the dueling architecture:
```python
class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.advantage = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        advantage = self.advantage(state)
        value = self.value(state)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
```

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

### 5.3 Evaluation Metrics
The agent's performance is evaluated using several key metrics:

1. **Nash Convergence**: Measured as the average exploitability $\epsilon$ against best responses:
   $$
   \epsilon = \frac{1}{2}\sum_{i=1}^2 \max_{\sigma'_i} (u_i(\sigma'_i, \sigma_{-i}) - u_i(\sigma))
   $$

2. **Expected Game Value**: Computed over N episodes:
   $$
   V = \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^T \gamma^t r_t^i
   $$

3. **Win Rate**: Against baseline agents (MCTS and CFR):
   $$
   WR_{agent} = \frac{\text{Games Won}}{\text{Total Games}} \times 100\%
   $$

4. **Average Stack Efficiency**: Measures effective stack utilisation:
   $$
   SE = \frac{1}{N}\sum_{i=1}^N \frac{\text{Final Stack}_i}{\text{Initial Stack}_i}
   $$

5. **Bluff Frequency**: Ratio of successful bluffs to total bluff attempts:
   $$
   BF = \frac{\text{Successful Bluffs}}{\text{Total Bluff Attempts}}
   $$

### 5.4 Policy Extraction
Following [Heinrich & Silver, 2016], we use a softmax policy with temperature:

$$
\pi_\beta(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}
$$

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
