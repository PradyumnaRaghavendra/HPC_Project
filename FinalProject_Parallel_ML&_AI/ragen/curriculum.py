"""
Curriculum Learning Manager for WebShop
Gradually increases task difficulty based on model performance
"""
from typing import Dict, List
from collections import deque


class CurriculumManager:
    """
    Manages curriculum learning progression for WebShop.

    Starts with easy tasks (10 products) and gradually increases
    difficulty (up to 100 products) based on success rate.
    """

    def __init__(self, config: Dict):
        """Initialize curriculum manager from config."""
        curriculum_config = config.get('curriculum', {})

        self.enabled = curriculum_config.get('enabled', False)

        if not self.enabled:
            # If disabled, just use the environment's configured product count (if available)
            self.current_products = config.get('environment', {}).get('num_products', None)
            self.stages = []
            self.current_stage = 0
            return

        # Curriculum stages with (product_count, success_threshold) pairs
        self.stages = curriculum_config.get('stages', [
            {'products': 10, 'threshold': 0.3},
            {'products': 25, 'threshold': 0.4},
            {'products': 50, 'threshold': 0.4},
            {'products': 100, 'threshold': 0.5},
        ])

        # Start at first stage
        self.current_stage = 0
        self.current_products = self.stages[0]['products']

        # Track recent success rates for progression decision
        self.check_every = curriculum_config.get('check_every', 10)
        self.recent_success_rates = deque(maxlen=3)  # Track last 3 evaluations

        print(f"\n🎓 CURRICULUM LEARNING ENABLED")
        print(f"   Stages: {[s['products'] for s in self.stages]} products")
        print(f"   Starting at: {self.current_products} products")
        print(f"   Check every: {self.check_every} steps\n")

    def get_current_products(self) -> int:
        """Get current product count for environment."""
        return self.current_products

    def record_success_rate(self, success_rate: float):
        """Record a new success rate from evaluation."""
        if not self.enabled:
            return

        self.recent_success_rates.append(success_rate)

    def should_increase_difficulty(self, step: int) -> bool:
        """
        Check if we should move to next stage.

        Criteria:
        - At least 3 evaluations completed
        - Average success rate meets current threshold
        - Not already at final stage
        """
        if not self.enabled:
            return False

        # Check every N steps
        if step % self.check_every != 0:
            return False

        # Need at least 3 evaluations to make decision
        if len(self.recent_success_rates) < 3:
            return False

        # Already at final stage?
        if self.current_stage >= len(self.stages) - 1:
            return False

        # Check if we meet threshold
        avg_success = sum(self.recent_success_rates) / len(self.recent_success_rates)
        current_threshold = self.stages[self.current_stage]['threshold']

        return avg_success >= current_threshold

    def increase_difficulty(self) -> int:
        """
        Move to next curriculum stage.

        Returns:
            New product count (for environment reinitialization)
        """
        if not self.enabled or self.current_stage >= len(self.stages) - 1:
            return self.current_products

        self.current_stage += 1
        old_products = self.current_products
        self.current_products = self.stages[self.current_stage]['products']

        print(f"\n🎓 CURRICULUM PROGRESSION")
        print(f"   {old_products} → {self.current_products} products")
        print(f"   Stage {self.current_stage + 1}/{len(self.stages)}")
        print(f"   Target success rate: {self.stages[self.current_stage]['threshold']:.1%}\n")

        # Reset success rate tracking for new stage
        self.recent_success_rates.clear()

        return self.current_products

    def get_progress_str(self) -> str:
        """Get human-readable progress string."""
        if not self.enabled:
            return f"Curriculum: Disabled ({self.current_products} products)"

        stage_num = self.current_stage + 1
        total_stages = len(self.stages)
        products = self.current_products

        if len(self.recent_success_rates) > 0:
            avg_success = sum(self.recent_success_rates) / len(self.recent_success_rates)
            return f"Curriculum: Stage {stage_num}/{total_stages} ({products} products, {avg_success:.1%} success)"
        else:
            return f"Curriculum: Stage {stage_num}/{total_stages} ({products} products)"
