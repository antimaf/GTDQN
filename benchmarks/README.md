# Benchmarks

This directory contains benchmark tools and results for evaluating the GT-DQN poker agent.

## Components

### Benchmark Tools
- `evaluate.py`: Main evaluation script
- `opponents/`: Implementation of baseline opponents
- `metrics/`: Performance metric calculations

## Evaluation Metrics

1. **Performance Metrics**
   - Win rate
   - BB/100 (big blinds per 100 hands)
   - AIVAT variance reduction
   - Nash equilibrium distance

2. **Computational Metrics**
   - Inference time
   - Memory usage
   - GPU utilization
   - Training convergence rate

## Baseline Opponents

1. **Rule-based**
   - Simple rules
   - Advanced rules with position play

2. **Traditional Algorithms**
   - CFR (Counterfactual Regret Minimization)
   - MCCFR (Monte Carlo CFR)

3. **Other Neural Approaches**
   - Basic DQN
   - A2C
   - PPO

## Running Benchmarks

```python
from benchmarks.evaluate import evaluate_agent

results = evaluate_agent(
    agent_path="checkpoints/latest.pth",
    n_hands=10000,
    opponent_type="cfr",
    stack_size=1000
)

# Print results
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"BB/100: {results['bb_100']:.2f}")
print(f"Nash Distance: {results['nash_distance']:.4f}")
```

## Results Format

Benchmark results are saved in JSON format:
```json
{
    "agent_version": "v1.0.0",
    "timestamp": "2024-01-20T10:00:00",
    "metrics": {
        "win_rate": 0.52,
        "bb_100": 5.23,
        "nash_distance": 0.0134
    },
    "computational": {
        "avg_inference_time": 0.023,
        "peak_memory_mb": 1256
    }
}
```
