"""
Expert demonstrations for SimpleMaze.
Just show the model the optimal path: right, right, right, right, down, down, down, down
"""


def get_maze_expert_demos():
    """Return expert demonstrations for maze navigation."""

    # Optimal path: (0,0) -> (4,4) is just go right 4 times, then down 4 times
    demos = []

    # Demo 1: Optimal path
    demos.append({
        'turns': [
            {
                'observation': 'Task: Get from (0, 0) to (4, 4)\n\nYou are in a 5x5 grid. Use these commands:\n- up - Move up\n- down - Move down\n- left - Move left\n- right - Move right\n\n>',
                'action': 'right',
                'reward': -0.01
            },
            {
                'observation': 'At (1, 0). Goal: (4, 4). Moves left: 19',
                'action': 'right',
                'reward': -0.01
            },
            {
                'observation': 'At (2, 0). Goal: (4, 4). Moves left: 18',
                'action': 'right',
                'reward': -0.01
            },
            {
                'observation': 'At (3, 0). Goal: (4, 4). Moves left: 17',
                'action': 'right',
                'reward': -0.01
            },
            {
                'observation': 'At (4, 0). Goal: (4, 4). Moves left: 16',
                'action': 'down',
                'reward': -0.01
            },
            {
                'observation': 'At (4, 1). Goal: (4, 4). Moves left: 15',
                'action': 'down',
                'reward': -0.01
            },
            {
                'observation': 'At (4, 2). Goal: (4, 4). Moves left: 14',
                'action': 'down',
                'reward': -0.01
            },
            {
                'observation': 'At (4, 3). Goal: (4, 4). Moves left: 13',
                'action': 'down',
                'reward': 1.0
            },
        ]
    })

    # Demo 2: Another optimal path (down first, then right)
    demos.append({
        'turns': [
            {
                'observation': 'Task: Get from (0, 0) to (4, 4)\n\nYou are in a 5x5 grid. Use these commands:\n- up - Move up\n- down - Move down\n- left - Move left\n- right - Move right\n\n>',
                'action': 'down',
                'reward': -0.01
            },
            {
                'observation': 'At (0, 1). Goal: (4, 4). Moves left: 19',
                'action': 'down',
                'reward': -0.01
            },
            {
                'observation': 'At (0, 2). Goal: (4, 4). Moves left: 18',
                'action': 'down',
                'reward': -0.01
            },
            {
                'observation': 'At (0, 3). Goal: (4, 4). Moves left: 17',
                'action': 'down',
                'reward': -0.01
            },
            {
                'observation': 'At (0, 4). Goal: (4, 4). Moves left: 16',
                'action': 'right',
                'reward': -0.01
            },
            {
                'observation': 'At (1, 4). Goal: (4, 4). Moves left: 15',
                'action': 'right',
                'reward': -0.01
            },
            {
                'observation': 'At (2, 4). Goal: (4, 4). Moves left: 14',
                'action': 'right',
                'reward': -0.01
            },
            {
                'observation': 'At (3, 4). Goal: (4, 4). Moves left: 13',
                'action': 'right',
                'reward': 1.0
            },
        ]
    })

    # Create more variations
    for _ in range(10):
        demos.append(demos[0].copy())

    return demos
