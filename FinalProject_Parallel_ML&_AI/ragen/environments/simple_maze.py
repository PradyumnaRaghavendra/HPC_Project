"""
Super simple 5x5 maze to test if training works at all.

Task: Get from (0,0) to (4,4)
Actions: up, down, left, right
Reward: +1 for goal, -0.01 per step
"""
from typing import Dict, Tuple
from .base import MultiTurnEnvironment


class SimpleMazeEnvironment(MultiTurnEnvironment):
    """
    Simplest possible environment to test training:
    - 5x5 grid
    - Start: (0, 0)
    - Goal: (4, 4)
    - Actions: up/down/left/right
    - Success if reach goal in < 20 steps
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_turns = 20
        self.grid_size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.pos = self.start

    def reset(self, task_data: Dict) -> str:
        """Start new episode."""
        self.current_turn = 0
        self.history = []
        self.pos = self.start

        instruction = f"Get from {self.start} to {self.goal}"
        return instruction

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute action."""
        self.current_turn += 1

        # Parse action
        action = action.lower().strip()
        x, y = self.pos

        # Move
        if action == "up" and y > 0:
            y -= 1
        elif action == "down" and y < self.grid_size - 1:
            y += 1
        elif action == "left" and x > 0:
            x -= 1
        elif action == "right" and x < self.grid_size - 1:
            x += 1
        else:
            # Invalid move (hit wall or bad action)
            observation = f"Invalid move. You're at {self.pos}. Try: up, down, left, right"
            return observation, -0.1, False, {'success': False}

        self.pos = (x, y)

        # Check if reached goal
        if self.pos == self.goal:
            observation = f"Success! Reached {self.goal}"
            return observation, 1.0, True, {'success': True}

        # Check max turns
        if self.current_turn >= self.max_turns:
            observation = f"Out of moves. At {self.pos}, goal was {self.goal}"
            return observation, -0.1, True, {'success': False}

        # Continue
        observation = f"At {self.pos}. Goal: {self.goal}. Moves left: {self.max_turns - self.current_turn}"
        return observation, -0.01, False, {'success': False}

    def compute_reward(self, trajectory: list) -> float:
        """Sum rewards from trajectory."""
        if not trajectory:
            return 0.0
        return sum(step.get('reward', 0.0) for step in trajectory)

    def render_text(self, state: str) -> str:
        """Format observation for model."""
        prompt = f"""Task: {state}

You are in a 5x5 grid. Use these commands:
- up - Move up
- down - Move down
- left - Move left
- right - Move right

>"""
        return prompt
