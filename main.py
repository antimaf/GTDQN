import torch
import numpy as np
from environments.bidding_game import BiddingGame
from train import GTDQNTrainer
from models.gt_layer import GameTheoreticLayer, SelfPlayManager

def main():
    # Create environment
    env = BiddingGame(num_agents=3)
    
    # Get state shape from environment
    state_shape = env.observation_space.shape  # [private_value, current_highest_bid, own_cash, round]
    num_actions = 100  # Discretized action space
    
    # Create trainer
    trainer = GTDQNTrainer(
        state_shape=state_shape,
        num_actions=num_actions,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create game theoretic components
    gt_layer = GameTheoreticLayer(
        num_actions=num_actions,
        hidden_size=128
    )
    
    # Create self-play manager with GTDQN model
    model_args = {
        "state_dim": state_shape[0],  # 4-dimensional state
        "num_actions": num_actions,
        "hidden_size": 128,
        "num_layers": 2
    }
    
    self_play_manager = SelfPlayManager(
        num_agents=5,
        model_class=type(trainer.policy_net),
        model_args=model_args
    )
    
    # Train GT-DQN
    print("\n=== Starting GT-DQN Training ===")
    trainer.train_gt_dqn(
        env=env,
        agent=trainer.policy_net,
        gt_layer=gt_layer,
        self_play_manager=self_play_manager,
        num_episodes=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=1e-4,
        target_update=100,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=100000,
        min_memory_size=1000,
        update_frequency=4,
        self_play_frequency=50,
        nash_threshold=0.1
    )
    
    # Evaluate against baselines
    print("\n=== Evaluating GT-DQN ===")
    baselines = ["random", "tight_aggressive", "loose_passive", "cfr", "mcts"]
    
    for opponent in baselines:
        print(f"\nEvaluating against {opponent}...")
        win_rate, avg_profit = trainer.evaluate_against_baseline(
            env=env,
            opponent_type=opponent,
            num_games=100
        )
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Profit: {avg_profit:.2f}")

if __name__ == "__main__":
    main()
