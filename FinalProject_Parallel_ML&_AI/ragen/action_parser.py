"""
Intelligent action parser - extracts valid actions from messy outputs
CLEANED: Removes URL encoding and limits to 3 words max
"""
import re


def parse_action_from_output(raw_output: str, fallback_instruction: str = "") -> str:
    """
    Extract valid WebShop action from model output.
    
    NEW: Aggressive cleaning of URL encoding and length limiting!
    """
    if not raw_output:
        # Extract 2-3 keywords from task
        words = fallback_instruction.lower().split()
        keywords = [w for w in words[:5] if len(w) > 3 and w not in 
                   {'find', 'with', 'that', 'this', 'from', 'price', 'lower', 'than', 'dollars', 'color', 'size'}]
        query = ' '.join(keywords[:2]) if keywords else 'products'
        return f"search[{query}]"
    
    text = raw_output.strip()
    
    # Pattern 1: search[...] (CLEANED!)
    match = re.search(r'search\[([^\]]+)\]', text, re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        
        # AGGRESSIVE CLEANING
        # Remove ALL URL encoding
        query = query.replace('%2C', ' ').replace('%20', ' ').replace('%27', '')
        query = query.replace('%E2%80%A6', '').replace('%2Fsome_search_query', '')
        query = query.replace('+', ' ').replace('_', ' ')
        
        # Remove special characters
        query = re.sub(r'[^\w\s-]', ' ', query)
        
        # Remove extra spaces
        query = ' '.join(query.split())

        # FIXED: DO NOT TRUNCATE! Let model use full query
        # Old bug: was limiting to 3 words, making searches useless
        # Now: keep full query (up to reasonable length for WebShop)
        words = query.split()[:10]  # Max 10 words (plenty for any search)

        # Remove only critical stopwords that add no value
        stopwords = {'the', 'a', 'an', 'some', 'search', 'query'}
        words = [w for w in words if w.lower() not in stopwords]

        clean_query = ' '.join(words)

        return f"search[{clean_query}]" if clean_query else "search[products]"
    
    # Pattern 2: click[...]
    match = re.search(r'click\[([^\]]+)\]', text, re.IGNORECASE)
    if match:
        product_id = match.group(1).strip()
        # Look for valid product ID (B0 + 8-10 chars)
        id_match = re.search(r'(B0[A-Z0-9]{8,10})', product_id, re.IGNORECASE)
        if id_match:
            return f"click[{id_match.group(1).upper()}]"
    
    # Pattern 3: buy now
    if re.search(r'\bbuy\s*(now)?\b', text, re.IGNORECASE):
        return "buy now"
    
    # Pattern 4: back
    if re.search(r'\bback\b', text, re.IGNORECASE):
        return "back"
    
    # FALLBACK: Extract keywords from task (NOT generic!)
    words = fallback_instruction.lower().split()
    keywords = [w for w in words[:8] if len(w) > 3 and w not in 
               {'find', 'with', 'that', 'this', 'from', 'price', 'lower', 'than', 'dollars'}]
    
    query = ' '.join(keywords[:2]) if keywords else 'products'
    return f"search[{query}]"