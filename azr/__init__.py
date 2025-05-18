"""
Training components for the AZR system.

This package contains implementations of the training components:
- Policy updates for reinforcement learning
- Reward calculations for different tasks
"""

from azr.training.policy import update_policy
from azr.training.rewards import learnability_reward

__all__ = ['update_policy', 'learnability_reward']