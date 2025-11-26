"""
Action-based curriculum rewards
Teaches search → click → buy sequence progressively
"""

def get_curriculum_phase(step: int) -> str:
    """Determine which action to emphasize."""
    if step < 30:
        return "search"
    elif step < 60:
        return "click"
    else:
        return "buy"


def curriculum_reward_bonus(trajectory: dict, step: int) -> float:
    """
    Add curriculum-based bonus.
    
    Phase 1 (0-30): Extra reward for searching
    Phase 2 (30-60): Extra reward for clicking  
    Phase 3 (60+): Extra reward for buying
    """
    phase = get_curriculum_phase(step)
    actions = trajectory.get('actions', [])
    
    bonus = 0.0
    
    # Check what actions were taken
    has_search = any('search[' in str(a).lower() for a in actions)
    has_click = any('click[' in str(a).lower() for a in actions)
    has_buy = any('buy' in str(a).lower() for a in actions)
    
    if phase == "search":
        # Phase 1: Big bonus for ANY search
        if has_search:
            bonus += 0.30
    
    elif phase == "click":
        # Phase 2: Bonus for search→click
        if has_search:
            bonus += 0.15
        if has_click:
            bonus += 0.35  # BIG bonus!
        if has_search and has_click:
            bonus += 0.15  # Sequence bonus
    
    else:  # buy phase
        # Phase 3: Bonus for full sequence
        if has_search:
            bonus += 0.10
        if has_click:
            bonus += 0.20
        if has_buy:
            bonus += 0.40  # BIG bonus!
        if has_search and has_click and has_buy:
            bonus += 0.25  # Perfect!
    
    return bonus