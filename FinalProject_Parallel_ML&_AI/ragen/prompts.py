"""
Pure WebShop action prompts - NO conversational elements
"""

def get_webshop_system_prompt():
    """System prompt that ONLY allows actions"""
    return """You are a WebShop agent. You MUST respond with ONLY ONE action.

VALID ACTIONS:
search[query] - Search for products
click[product_id] - Click a product (e.g., click[B07XYZ123])
buy now - Purchase current product
back - Go back

RULES:
1. Output ONLY the action, nothing else
2. No explanations, no reasoning, no text
3. One action per turn
4. If you output anything except a valid action, you FAIL

Example outputs:
search[laptop stand]
click[B07VNQN2V1]
buy now
back"""


def get_webshop_user_prompt(task, observation):
    """User prompt focused purely on action generation"""
    return f"""{task}

Current state: {observation}

Output your action now:"""


# NO CONVERSATION, NO "let me help you", NO explanations