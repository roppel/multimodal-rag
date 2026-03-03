"""
Cheese RAG System - WITH GPT Categorization
Run this to index with better categorization
"""
from main import MultiModalRAG
import json
import os
from pathlib import Path
from openai import OpenAI

class CheeseRAGWithGPT(MultiModalRAG):
    """Extended RAG for cheese e-commerce data with GPT categorization"""

    def __init__(self, data_dir='cheese_data'):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.metadata_file = self.data_dir / 'descriptions.json'
        self.gpt_client = OpenAI()

    def load_cheese_products(self):
        """Load cheese product metadata"""
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Cheese metadata not found at {self.metadata_file}\n"
                "Run prepare_cheese_dataset.py first!"
            )

        with open(self.metadata_file, 'r') as f:
            products = json.load(f)

        print(f"Loaded {len(products)} cheese products")
        return products

    def categorize_with_gpt(self, title, brand):
        """Use GPT to categorize a cheese product"""

        prompt = f"""Categorize this cheese product:

Title: {title}
Brand: {brand}

Provide categorization in JSON format:

{{
  "category": "one of: cheese, gift_basket, accessory, subscription",
  "cheese_type": "one of: cheddar, manchego, stilton, brie, gouda, mozzarella, parmesan, swiss, blue, feta, string, other",
  "region": "one of: french, italian, english, spanish, swiss, american, unknown"
}}

Rules:
- category: gift_basket if it's a gift set/basket/sampler/collection, accessory if it's a board/knife/tool, subscription if it's a monthly club, otherwise cheese
- cheese_type: identify the primary cheese type, or 'other' if not in the list or if it's a mix
- region: identify origin/region, or 'unknown' if unclear

Return ONLY the JSON, no other text."""

        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100
            )

            result = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
                result = result.strip()

            return json.loads(result)

        except Exception as e:
            print(f"  ⚠️  GPT categorization failed: {e}")
            return None

    def index_cheese_data_with_gpt(self):
        """Index cheese products using GPT categorization"""
        products = self.load_cheese_products()

        print(f"\nIndexing {len(products)} cheese products with GPT categorization...")
        print("Cost: ~$2 total ($1.50 vision + $0.50 categorization)")

        indexed_count = 0
        categorized_products = []

        for i, product in enumerate(products, 1):
            filename = product['filename']
            image_path = self.images_dir / filename

            if not image_path.exists():
                print(f"⚠️  Image not found: {filename}, skipping...")
                continue

            # Get GPT categorization
            categories = self.categorize_with_gpt(product['title'], product.get('brand', 'Unknown'))

            if not categories:
                print(f"⚠️  Skipping {filename} - categorization failed")
                continue

            # Create text description
            text_desc = f"{product['title']} - ${product.get('price', 'N/A')}"
            if product.get('brand') and product['brand'] != 'Unknown':
                text_desc += f" by {product['brand']}"

            # Build metadata
            metadata = {
                'filename': filename,
                'asin': product['asin'],
                'title': product['title'],
                'price': product.get('price'),
                'brand': product.get('brand', 'Unknown'),
                'category': categories['category'],
                'cheese_type': categories['cheese_type'],
                'region': categories['region']
            }

            try:
                # Get visual description from GPT-4 Vision
                visual_desc = self.get_image_description_from_claude(str(image_path))

                # Create embeddings
                text_embedding = self.text_embedder.encode(text_desc).tolist()
                visual_embedding = self.text_embedder.encode(visual_desc).tolist()

                # Store in ChromaDB
                self.text_collection.add(
                    documents=[text_desc],
                    embeddings=[text_embedding],
                    metadatas=[metadata],
                    ids=[f"text_{filename}"]
                )

                self.image_collection.add(
                    documents=[visual_desc],
                    embeddings=[visual_embedding],
                    metadatas=[metadata],
                    ids=[f"image_{filename}"]
                )

                # Save for eval
                categorized_product = product.copy()
                categorized_product.update({
                    'category': categories['category'],
                    'cheese_type': categories['cheese_type'],
                    'region': categories['region']
                })
                categorized_products.append(categorized_product)

                indexed_count += 1

                if i % 50 == 0:
                    print(f"  Processed {i}/{len(products)} products...")

            except Exception as e:
                print(f"⚠️  Error processing {filename}: {e}")
                continue

        # Save categorized products
        categorized_file = self.data_dir / 'categorized_gpt_indexed.json'
        with open(categorized_file, 'w') as f:
            json.dump(categorized_products, f, indent=2)

        print(f"\n✓ Successfully indexed {indexed_count} cheese products")
        print(f"  - Text descriptions: {indexed_count}")
        print(f"  - Visual descriptions: {indexed_count}")
        print(f"  - Saved to: {categorized_file}")

def main():
    print("="*70)
    print("CHEESE RAG - GPT CATEGORIZATION")
    print("="*70)

    rag = CheeseRAGWithGPT()
    rag.index_cheese_data_with_gpt()

    print("\n" + "="*70)
    print("DONE! Now run: python eval_cheese_final.py")
    print("="*70)

if __name__ == "__main__":
    main()