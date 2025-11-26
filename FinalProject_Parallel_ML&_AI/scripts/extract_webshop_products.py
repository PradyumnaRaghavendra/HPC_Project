"""
Extract 100 products from real WebShop data:
- 50 random products
- 50 hand-picked diverse products from different categories
"""

import json
import random
from pathlib import Path

# Load WebShop data
webshop_data_path = Path("../WebShop/data/items_shuffle_1000.json")

print("Loading WebShop data...")
with open(webshop_data_path) as f:
    all_products = json.load(f)

print(f"Total products available: {len(all_products)}")

# Extract 50 random products
random.seed(42)  # For reproducibility
random_indices = random.sample(range(len(all_products)), 50)
random_products = [all_products[i] for i in random_indices]

print(f"\nExtracted {len(random_products)} random products")

# Hand-pick 50 diverse products from different categories
# Look at categories available
categories = {}
for prod in all_products:
    cat = prod.get('category', 'unknown')
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(prod)

print(f"\nCategories found: {list(categories.keys())}")
print(f"Products per category: {[(cat, len(prods)) for cat, prods in categories.items()]}")

# Pick diverse products: ~10 from each category
diverse_products = []
target_per_category = 10

for cat, prods in categories.items():
    # Sample up to 10 products from this category
    sample_size = min(target_per_category, len(prods))
    samples = random.sample(prods, sample_size)
    diverse_products.extend(samples)
    print(f"  {cat}: selected {sample_size} products")

# Trim to exactly 50
diverse_products = diverse_products[:50]
print(f"\nTotal diverse products: {len(diverse_products)}")

# Combine and simplify product data
def simplify_product(prod, prod_id):
    """Simplify product to essential fields."""

    # Extract keywords from name and description
    name = prod.get('name', '').lower()
    small_desc = ' '.join(prod.get('small_description', [])).lower() if isinstance(prod.get('small_description'), list) else str(prod.get('small_description', '')).lower()

    # Basic keywords from name
    name_words = [w for w in name.split() if len(w) > 2 and w not in ['the', 'and', 'for', 'with']]

    # Category
    category = prod.get('category', 'general')

    # Price
    pricing = prod.get('pricing', '')

    return {
        'id': prod_id,
        'name': name,
        'keywords': name_words[:10],  # Top 10 keywords
        'category': category,
        'price': pricing,
        'asin': prod.get('asin', ''),
        'source': 'webshop'
    }

# Simplify all products
simplified_random = [simplify_product(p, i+1) for i, p in enumerate(random_products)]
simplified_diverse = [simplify_product(p, i+51) for i, p in enumerate(diverse_products)]

all_simplified = simplified_random + simplified_diverse

# Save to JSON
output_path = Path("data/medium_webshop_real_products.json")
output_path.parent.mkdir(exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(all_simplified, f, indent=2)

print(f"\n✓ Saved {len(all_simplified)} products to {output_path}")

# Show samples
print("\nSample random products:")
for prod in simplified_random[:3]:
    print(f"  {prod['id']}. {prod['name'][:60]}... ({prod['category']})")

print("\nSample diverse products:")
for prod in simplified_diverse[:3]:
    print(f"  {prod['id']}. {prod['name'][:60]}... ({prod['category']})")
