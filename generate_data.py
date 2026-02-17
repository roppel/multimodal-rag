"""
Download real product images from Unsplash
"""
import requests
import json
import os

# Image URLs and metadata
PRODUCTS = [
    {
        "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=800",
        "filename": "red_running_shoes.jpg",
        "name": "Running Shoes",
        "color": "red",
        "description": "Red athletic running shoes, lightweight mesh material, excellent for marathon training",
        "category": "footwear",
        "price": 120
    },
    {
        "url": "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=800",
        "filename": "white_sneakers.jpg",
        "name": "Sneakers",
        "color": "white",
        "description": "Classic white leather sneakers, versatile casual style",
        "category": "footwear",
        "price": 85
    },
    {
        "url": "https://images.unsplash.com/photo-1605812860427-4024433a70fd?w=800",
        "filename": "brown_boots.jpg",
        "name": "Hiking Boots",
        "color": "brown",
        "description": "Durable waterproof hiking boots with ankle support",
        "category": "footwear",
        "price": 150
    },
    {
        "url": "https://images.unsplash.com/photo-1580480055273-228ff5388ef8?w=800",
        "filename": "black_office_chair.jpg",
        "name": "Office Chair",
        "color": "black",
        "description": "Ergonomic office chair with lumbar support and adjustable height",
        "category": "furniture",
        "price": 299
    },
    {
        "url": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=800",
        "filename": "blue_armchair.jpg",
        "name": "Armchair",
        "color": "blue",
        "description": "Modern blue velvet armchair, mid-century design, comfortable padding",
        "category": "furniture",
        "price": 450
    },
    {
        "url": "https://images.unsplash.com/photo-1533090481720-856c6e3c1fdc?w=800",
        "filename": "wooden_table.jpg",
        "name": "Dining Table",
        "color": "wood",
        "description": "Solid oak dining table seats 6, rustic farmhouse style",
        "category": "furniture",
        "price": 799
    },
    {
        "url": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=800",
        "filename": "silver_laptop.jpg",
        "name": "Laptop",
        "color": "silver",
        "description": "15-inch laptop, 16GB RAM, fast SSD, perfect for software development",
        "category": "electronics",
        "price": 1299
    },
    {
        "url": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=800",
        "filename": "smartphone.jpg",
        "name": "Smartphone",
        "color": "black",
        "description": "Latest smartphone with 5G, excellent camera system",
        "category": "electronics",
        "price": 899
    },
    {
        "url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=800",
        "filename": "white_headphones.jpg",
        "name": "Headphones",
        "color": "white",
        "description": "Noise-canceling wireless headphones, 30-hour battery life",
        "category": "electronics",
        "price": 249
    },
    {
        "url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=800",
        "filename": "gray_backpack.jpg",
        "name": "Backpack",
        "color": "gray",
        "description": "Durable travel backpack with laptop compartment, water-resistant",
        "category": "accessories",
        "price": 89
    }
]

def download_images():
    """Download all images from Unsplash"""
    # Create images directory
    os.makedirs("images", exist_ok=True)

    print("Downloading real product images from Unsplash...")
    print("="*60)

    descriptions = {}

    for i, product in enumerate(PRODUCTS, 1):
        print(f"\n[{i}/{len(PRODUCTS)}] Downloading {product['filename']}...")

        try:
            # Download image
            response = requests.get(product['url'], stream=True)
            response.raise_for_status()

            # Save image
            filepath = os.path.join("images", product['filename'])
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✓ Saved to {filepath}")

            # Add to descriptions
            descriptions[product['filename']] = {
                "name": product['name'],
                "color": product['color'],
                "description": product['description'],
                "category": product['category'],
                "price": product['price']
            }

        except Exception as e:
            print(f"✗ Failed to download {product['filename']}: {e}")

    # Save descriptions.json
    os.makedirs("data", exist_ok=True)
    with open("data/descriptions.json", "w") as f:
        json.dump(descriptions, f, indent=2)

    print("\n" + "="*60)
    print(f"✓ Downloaded {len(descriptions)} images")
    print("✓ Created data/descriptions.json")
    print("\nReady to run: python main.py")

if __name__ == "__main__":
    download_images()