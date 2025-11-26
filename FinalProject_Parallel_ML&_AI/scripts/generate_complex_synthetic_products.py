"""
Generate 100 complex synthetic products for Medium200WebShop.
These are more complex than Stage 1 simple products:
- Longer product names with more attributes
- More keywords per product
- More nuanced categories and variations
- Similar products that are harder to discriminate
"""

import json
import random
from pathlib import Path

def generate_complex_products():
    """Generate 100 complex but realistic synthetic products."""

    products = []
    product_id = 101  # Start from 101 (after the 100 real products)

    # Complex Electronics (25 products)
    electronics_templates = [
        ("wireless noise-cancelling bluetooth headphones with active noise reduction",
         ["wireless", "noise", "cancelling", "bluetooth", "headphones", "active", "reduction", "audio"]),
        ("ultra-slim laptop computer with 16gb ram and ssd storage",
         ["ultra", "slim", "laptop", "computer", "16gb", "ram", "ssd", "storage"]),
        ("high-resolution 4k ultra hd gaming monitor with hdr support",
         ["high", "resolution", "4k", "ultra", "hd", "gaming", "monitor", "hdr"]),
        ("smart home security camera system with night vision and motion detection",
         ["smart", "home", "security", "camera", "night", "vision", "motion", "detection"]),
        ("portable external hard drive with 2tb storage capacity and usb-c",
         ["portable", "external", "hard", "drive", "2tb", "storage", "usb", "c"]),
    ]

    colors = ["black", "white", "silver", "blue", "red"]
    sizes = ["small", "medium", "large", "compact", "professional"]

    for i in range(25):
        template_name, template_keywords = random.choice(electronics_templates)
        color = random.choice(colors)
        size = random.choice(sizes)

        name = f"{color} {size} {template_name}"
        keywords = [color, size] + template_keywords

        products.append({
            'id': product_id,
            'name': name,
            'keywords': keywords[:12],
            'category': 'electronics',
            'price': f"${random.randint(50, 500)}.{random.randint(0, 99):02d}",
            'source': 'synthetic_complex'
        })
        product_id += 1

    # Complex Clothing & Fashion (25 products)
    clothing_templates = [
        ("premium cotton long-sleeve button-down dress shirt with collar",
         ["premium", "cotton", "long", "sleeve", "button", "down", "dress", "shirt", "collar"]),
        ("athletic performance moisture-wicking running shorts with pockets",
         ["athletic", "performance", "moisture", "wicking", "running", "shorts", "pockets"]),
        ("heavyweight fleece pullover hoodie sweatshirt with drawstring",
         ["heavyweight", "fleece", "pullover", "hoodie", "sweatshirt", "drawstring"]),
        ("stretch denim slim-fit jeans pants with five pockets",
         ["stretch", "denim", "slim", "fit", "jeans", "pants", "five", "pockets"]),
        ("waterproof insulated winter parka jacket with removable hood",
         ["waterproof", "insulated", "winter", "parka", "jacket", "removable", "hood"]),
    ]

    colors = ["navy", "charcoal", "olive", "burgundy", "khaki"]

    for i in range(25):
        template_name, template_keywords = random.choice(clothing_templates)
        color = random.choice(colors)

        name = f"{color} {template_name}"
        keywords = [color] + template_keywords

        products.append({
            'id': product_id,
            'name': name,
            'keywords': keywords[:12],
            'category': 'clothing',
            'price': f"${random.randint(20, 150)}.{random.randint(0, 99):02d}",
            'source': 'synthetic_complex'
        })
        product_id += 1

    # Complex Home & Kitchen (25 products)
    home_templates = [
        ("stainless steel non-stick ceramic coating frying pan with lid",
         ["stainless", "steel", "non", "stick", "ceramic", "coating", "frying", "pan", "lid"]),
        ("electric programmable slow cooker with digital timer and temperature control",
         ["electric", "programmable", "slow", "cooker", "digital", "timer", "temperature", "control"]),
        ("high-thread-count egyptian cotton bed sheet set with deep pockets",
         ["high", "thread", "count", "egyptian", "cotton", "bed", "sheet", "deep", "pockets"]),
        ("adjustable ergonomic lumbar support office chair with armrests",
         ["adjustable", "ergonomic", "lumbar", "support", "office", "chair", "armrests"]),
        ("memory foam contour pillow with cooling gel and breathable cover",
         ["memory", "foam", "contour", "pillow", "cooling", "gel", "breathable", "cover"]),
    ]

    for i in range(25):
        template_name, template_keywords = random.choice(home_templates)

        name = template_name
        keywords = template_keywords

        products.append({
            'id': product_id,
            'name': name,
            'keywords': keywords[:12],
            'category': 'home_kitchen',
            'price': f"${random.randint(30, 200)}.{random.randint(0, 99):02d}",
            'source': 'synthetic_complex'
        })
        product_id += 1

    # Complex Sports & Fitness (15 products)
    sports_templates = [
        ("adjustable weight dumbbell set with non-slip grip handles",
         ["adjustable", "weight", "dumbbell", "set", "non", "slip", "grip", "handles"]),
        ("high-density foam roller for deep tissue massage and recovery",
         ["high", "density", "foam", "roller", "deep", "tissue", "massage", "recovery"]),
        ("professional yoga mat with alignment markers and carrying strap",
         ["professional", "yoga", "mat", "alignment", "markers", "carrying", "strap"]),
        ("lightweight camping tent with waterproof rainfly and mesh windows",
         ["lightweight", "camping", "tent", "waterproof", "rainfly", "mesh", "windows"]),
        ("inflatable stand-up paddle board with adjustable paddle and pump",
         ["inflatable", "stand", "up", "paddle", "board", "adjustable", "pump"]),
    ]

    for i in range(15):
        template_name, template_keywords = random.choice(sports_templates)

        name = template_name
        keywords = template_keywords

        products.append({
            'id': product_id,
            'name': name,
            'keywords': keywords[:12],
            'category': 'sports_outdoors',
            'price': f"${random.randint(40, 300)}.{random.randint(0, 99):02d}",
            'source': 'synthetic_complex'
        })
        product_id += 1

    # Complex Beauty & Personal Care (10 products)
    beauty_templates = [
        ("organic natural moisturizing face cream with hyaluronic acid and vitamins",
         ["organic", "natural", "moisturizing", "face", "cream", "hyaluronic", "acid", "vitamins"]),
        ("professional ceramic ionic hair dryer with multiple heat settings",
         ["professional", "ceramic", "ionic", "hair", "dryer", "multiple", "heat", "settings"]),
        ("electric rechargeable toothbrush with pressure sensor and timer",
         ["electric", "rechargeable", "toothbrush", "pressure", "sensor", "timer"]),
        ("vitamin c brightening serum with anti-aging antioxidants",
         ["vitamin", "c", "brightening", "serum", "anti", "aging", "antioxidants"]),
        ("professional makeup brush set with synthetic bristles and case",
         ["professional", "makeup", "brush", "set", "synthetic", "bristles", "case"]),
    ]

    for i in range(10):
        template_name, template_keywords = random.choice(beauty_templates)

        name = template_name
        keywords = template_keywords

        products.append({
            'id': product_id,
            'name': name,
            'keywords': keywords[:12],
            'category': 'beauty',
            'price': f"${random.randint(15, 80)}.{random.randint(0, 99):02d}",
            'source': 'synthetic_complex'
        })
        product_id += 1

    return products


# Generate and save
products = generate_complex_products()

output_path = Path("data/medium_webshop_synthetic_products.json")
output_path.parent.mkdir(exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(products, f, indent=2)

print(f"✓ Generated {len(products)} complex synthetic products")
print(f"✓ Saved to {output_path}")

# Show samples
print("\nSample products by category:")
categories = {}
for prod in products:
    cat = prod['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(prod)

for cat, prods in categories.items():
    print(f"\n{cat.upper()} ({len(prods)} products):")
    for prod in prods[:2]:
        print(f"  {prod['id']}. {prod['name'][:70]}...")
