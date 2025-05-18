def learnability_reward(solve_rewards):
    """
    Calculate the learnability reward for a task.
    Higher reward for tasks that are neither too easy nor too hard.
    
    Args:
        solve_rewards: List of rewards from solver attempts
        
    Returns:
        Learnability reward value
    """
    if not solve_rewards:
        return 0.0
    
    avg_solve_reward = sum(solve_rewards) / len(solve_rewards)
    
    # Tasks that are always solved (too easy) or never solved (too hard)
    # should get lower rewards than moderately difficult tasks
    if avg_solve_reward == 0.0 or avg_solve_reward == 1.0:
        return 0.0
    
    # Peak reward for tasks with 0.5 difficulty
    # This encourages tasks that are challenging but solvable
    return 1.0 - abs(0.5 - avg_solve_reward) * 2.0