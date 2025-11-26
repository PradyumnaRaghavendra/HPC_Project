"""
Intelligent action sanitizer for WebShop
Extracts meaningful actions from conversational text
"""
import re


def extract_keywords(text, max_words=5):
    """Extract meaningful keywords from text."""
    # Remove common words
    stopwords = {'find', 'me', 'the', 'a', 'an', 'for', 'with', 'that', 'this', 
                 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'from', 'of', 'by',
                 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'am', 'is', 'are',
                 'search', 'click', 'buy', 'task', 'please', 'can', 'will', 'would'}
    
    # Extract words
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    
    # Filter stopwords
    keywords = [w for w in words if w not in stopwords]
    
    # Return first max_words
    return keywords[:max_words]


def sanitize(text: str, fallback_query: str) -> str:
    """
    Convert any text to valid WebShop action.
    Aggressively extracts intent from conversational outputs.
    """
    if not text or not isinstance(text, str):
        # Use task as fallback
        keywords = extract_keywords(fallback_query, max_words=3)
        query = ' '.join(keywords) if keywords else 'products'
        return f"search[{query}]"
    
    text_clean = text.strip()
    text_lower = text_clean.lower()
    
    # ===== CHECK FOR VALID FORMAT FIRST =====
    
    # Already valid search
    if text_lower.startswith('search[') and ']' in text_lower:
        match = re.search(r'search\[([^\]]+)\]', text_lower)
        if match:
            query = match.group(1).strip()
            return f"search[{query[:80]}]"  # Truncate long queries
    
    # Already valid click
    if text_lower.startswith('click[') and ']' in text_lower:
        match = re.search(r'click\[([^\]]+)\]', text_lower)
        if match:
            product_id = match.group(1).strip().upper()
            return f"click[{product_id}]"
    
    # Buy command
    if text_lower in ['buy now', 'buy', 'purchase']:
        return 'buy now'
    
    # Back command
    if text_lower == 'back':
        return 'back'
    
    # ===== EXTRACT INTENT FROM CONVERSATIONAL TEXT =====
    
    # Check for buy intent
    if any(word in text_lower[:30] for word in ['buy', 'purchase', 'get this', 'add to cart']):
        return 'buy now'
    
    # Check for back intent
    if any(word in text_lower[:20] for word in ['go back', 'return', 'previous']):
        return 'back'
    
    # Check for click intent + extract product ID
    if any(word in text_lower for word in ['click', 'select', 'view', 'show']):
        # Try to find product ID (format: B0 + 8-10 alphanumeric)
        product_match = re.search(r'\b(B0[A-Z0-9]{8,10})\b', text_clean, re.IGNORECASE)
        if product_match:
            return f"click[{product_match.group(1).upper()}]"
        
        # No product ID found - just search instead
        keywords = extract_keywords(fallback_query, max_words=3)
        query = ' '.join(keywords) if keywords else 'products'
        return f"search[{query}]"
    
    # ===== DEFAULT: EXTRACT SEARCH QUERY =====
    
    # Try to find "search for X" pattern
    search_patterns = [
        r'search\s+for\s+([a-z0-9\s]+)',
        r'search\s+([a-z0-9\s]+)',
        r'find\s+([a-z0-9\s]+)',
        r'looking for\s+([a-z0-9\s]+)',
    ]
    
    for pattern in search_patterns:
        match = re.search(pattern, text_lower)
        if match:
            query = match.group(1).strip()[:80]
            if query and len(query) > 2:
                return f"search[{query}]"
    
    # No clear pattern - extract keywords from model output
    model_keywords = extract_keywords(text_clean, max_words=3)
    
    # If model output has good keywords, use them
    if model_keywords and len(model_keywords) >= 2:
        query = ' '.join(model_keywords)
        return f"search[{query}]"
    
    # Last resort: use task keywords
    task_keywords = extract_keywords(fallback_query, max_words=3)
    query = ' '.join(task_keywords) if task_keywords else 'products'
    return f"search[{query}]"