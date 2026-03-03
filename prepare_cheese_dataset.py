"""
Sample 100 cheese products and download images from Amazon URLs
Prepare data for multi-modal RAG system
"""
import json
import random
import requests
import os
from pathlib import Path
import time

def load_cheese_products():
    """Load the filtered cheese products"""
    with open('data/cheese_products.json', 'r') as f:
        products = json.load(f)
    return products

def sample_products(products, n=100):
    """Sample n products with good metadata"""
    # Filter for products with price, title, image
    good_products = [
        p for p in products 
        if p.get('price') and p.get('title') and p.get('imUrl')
    ]
    
    print(f"Found {len(good_products)} products with complete metadata")
    
    # Sample
    sampled = random.sample(good_products, min(n, len(good_products)))
    
    return sampled

def download_image(url, filename):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download {url}: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def prepare_cheese_dataset(products, output_dir='cheese_data'):
    """Download images and create metadata file"""
    # Create directories
    images_dir = Path(output_dir) / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    successful_products = []
    
    print(f"\nDownloading images for {len(products)} products...")
    
    for i, product in enumerate(products, 1):
        asin = product['asin']
        title = product['title']
        image_url = product['imUrl']
        
        # Create filename
        filename = f"{asin}.jpg"
        filepath = images_dir / filename
        
        print(f"[{i}/{len(products)}] Downloading: {title[:50]}...")
        
        # Download image
        if download_image(image_url, filepath):
            # Add to successful products
            successful_products.append({
                'filename': filename,
                'asin': asin,
                'title': title,
                'price': product.get('price'),
                'brand': product.get('brand', 'Unknown'),
                'categories': product.get('categories', []),
                'image_url': image_url
            })
            
            # Be nice to Amazon's servers
            time.sleep(0.5)
        else:
            print(f"  ⚠️  Failed, skipping...")
    
    print(f"\n✓ Successfully downloaded {len(successful_products)} images")
    
    # Save metadata
    metadata_file = Path(output_dir) / 'descriptions.json'
    with open(metadata_file, 'w') as f:
        json.dump(successful_products, f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_file}")
    
    return successful_products

def analyze_dataset(products):
    """Show what we got"""
    print("\n" + "="*70)
    print("CHEESE RAG DATASET - READY")
    print("="*70)
    
    print(f"\nTotal products: {len(products)}")
    
    # Sample
    print("\nSample products:")
    for i, p in enumerate(random.sample(products, min(10, len(products))), 1):
        print(f"{i}. {p['title'][:55]}... (${p['price']})")
    
    # Price stats
    prices = [p['price'] for p in products if p['price']]
    if prices:
        print(f"\nPrice range: ${min(prices):.2f} - ${max(prices):.2f}")
        print(f"Average: ${sum(prices)/len(prices):.2f}")
    
    # Brands
    brands = [p.get('brand') for p in products if p.get('brand') and p.get('brand') != 'Unknown']
    if brands:
        top_brands = {}
        for brand in brands:
            top_brands[brand] = top_brands.get(brand, 0) + 1
        
        print(f"\nTop brands:")
        for brand, count in sorted(top_brands.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {brand}: {count} products")

def main():
    print("Loading cheese products...")
    products = load_cheese_products()
    
    print(f"Loaded {len(products)} cheese products")
    
    # Sample 1000
    sampled = sample_products(products, n=1000)
    print(f"Sampled {len(sampled)} products")
    
    # Download images and prepare dataset
    final_products = prepare_cheese_dataset(sampled)
    
    # Analyze
    analyze_dataset(final_products)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("1. Run GPT-4 Vision on images (creates visual descriptions)")
    print("2. Create test queries based on Cheese Express patterns")
    print("3. Run evaluation")
    print("="*70)

if __name__ == "__main__":
    main()
