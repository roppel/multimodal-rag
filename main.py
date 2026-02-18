"""
Multi-Modal RAG System
Searches over both images and text descriptions using vision models and embeddings
"""
from openai import OpenAI
import json
import os
import base64
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class MultiModalRAG:
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI()

        # Initialize text embedding model
        print("Loading embedding model...")
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=False
        ))

        # Create collections for text and images
        self.text_collection = self.chroma_client.create_collection(
            name="product_descriptions"
        )
        self.image_collection = self.chroma_client.create_collection(
            name="product_images"
        )

        print("MultiModal RAG initialized!")

    def load_image_as_base64(self, image_path):
        """Convert image to base64 for Claude"""
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")

    def get_image_description_from_claude(self, image_path):
        """Use GPT-4 Vision to describe the image"""
        image_data = self.load_image_as_base64(image_path)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe this product image in detail. Focus on: color, type of item, key visual features, style. Be concise but specific."
                        }
                    ],
                }
            ],
        )

        return response.choices[0].message.content

    def index_data(self):
        """Index both images and text descriptions"""
        # Load descriptions
        with open("data/descriptions.json", "r") as f:
            descriptions = json.load(f)

        print(f"\nIndexing {len(descriptions)} products...")

        text_docs = []
        text_embeddings = []
        text_ids = []
        text_metadatas = []

        image_docs = []
        image_embeddings = []
        image_ids = []
        image_metadatas = []

        for i, (filename, info) in enumerate(descriptions.items()):
            image_path = f"images/{filename}"

            # Index text description
            text_content = f"{info['name']} - {info['description']} - {info['color']} - {info['category']}"
            text_embedding = self.text_embedder.encode(text_content).tolist()

            text_docs.append(text_content)
            text_embeddings.append(text_embedding)
            text_ids.append(f"text_{i}")
            text_metadatas.append({
                "filename": filename,
                "name": info['name'],
                "color": info['color'],
                "category": info['category'],
                "price": info['price']
            })

            # Index image using Claude's vision
            print(f"Processing image {i+1}/{len(descriptions)}: {filename}")
            image_description = self.get_image_description_from_claude(image_path)
            image_embedding = self.text_embedder.encode(image_description).tolist()

            image_docs.append(image_description)
            image_embeddings.append(image_embedding)
            image_ids.append(f"image_{i}")
            image_metadatas.append({
                "filename": filename,
                "image_path": image_path,
                "visual_description": image_description
            })

        # Add to ChromaDB
        self.text_collection.add(
            documents=text_docs,
            embeddings=text_embeddings,
            ids=text_ids,
            metadatas=text_metadatas
        )

        self.image_collection.add(
            documents=image_docs,
            embeddings=image_embeddings,
            ids=image_ids,
            metadatas=image_metadatas
        )

        print(f"\n✓ Indexed {len(text_docs)} text descriptions")
        print(f"✓ Indexed {len(image_docs)} images with visual descriptions")

    def search(self, query, n_results=3, search_images=True, search_text=True):
        """Search across images and/or text"""
        query_embedding = self.text_embedder.encode(query).tolist()

        results = {
            "query": query,
            "text_results": [],
            "image_results": []
        }

        # Search text descriptions
        if search_text:
            text_results = self.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            for i in range(len(text_results['ids'][0])):
                results["text_results"].append({
                    "content": text_results['documents'][0][i],
                    "metadata": text_results['metadatas'][0][i],
                    "distance": text_results['distances'][0][i]
                })

        # Search images
        if search_images:
            image_results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            for i in range(len(image_results['ids'][0])):
                results["image_results"].append({
                    "visual_description": image_results['documents'][0][i],
                    "metadata": image_results['metadatas'][0][i],
                    "distance": image_results['distances'][0][i]
                })

        return results

    def hybrid_search(self, query, n_results=3, filters=None, search_images=True, search_text=True):
        """
        Hybrid search: combine metadata filtering with semantic search

        Args:
            query: Search query string
            n_results: Number of results to return
            filters: Dict of metadata filters, e.g., {"color": "red", "category": "footwear"}
            search_images: Whether to search image descriptions
            search_text: Whether to search text descriptions
        """
        query_embedding = self.text_embedder.encode(query).tolist()

        results = {
            "query": query,
            "filters": filters,
            "text_results": [],
            "image_results": []
        }

        # Build where clause for ChromaDB
        where_clause = None
        if filters:
            where_conditions = []
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Handle range queries like {"$lt": 300}
                    where_conditions.append({key: value})
                else:
                    # Handle exact matches
                    where_conditions.append({key: {"$eq": value}})

            # Combine conditions with AND
            if len(where_conditions) == 1:
                where_clause = where_conditions[0]
            else:
                where_clause = {"$and": where_conditions}

        # Search text descriptions with filters
        if search_text:
            try:
                text_results = self.text_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results * 2,  # Get more results to account for filtering
                    where=where_clause
                )

                for i in range(min(n_results, len(text_results['ids'][0]))):
                    results["text_results"].append({
                        "content": text_results['documents'][0][i],
                        "metadata": text_results['metadatas'][0][i],
                        "distance": text_results['distances'][0][i]
                    })
            except Exception as e:
                print(f"Text search error: {e}")

        # Search images with filters
        if search_images:
            try:
                image_results = self.image_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results * 2,
                    where=where_clause
                )

                for i in range(min(n_results, len(image_results['ids'][0]))):
                    results["image_results"].append({
                        "visual_description": image_results['documents'][0][i],
                        "metadata": image_results['metadatas'][0][i],
                        "distance": image_results['distances'][0][i]
                    })
            except Exception as e:
                print(f"Image search error: {e}")

        return results

    def answer_question(self, query):
        """Use RAG to answer a question about products"""
        # Get relevant context
        search_results = self.search(query, n_results=3)

        # Build context for Claude
        context_parts = []

        context_parts.append("TEXT DESCRIPTIONS:")
        for result in search_results['text_results']:
            context_parts.append(f"- {result['content']}")

        context_parts.append("\nVISUAL DESCRIPTIONS:")
        for result in search_results['image_results']:
            context_parts.append(f"- {result['visual_description']} (from {result['metadata']['filename']})")

        context = "\n".join(context_parts)

        # Ask GPT-4 to answer using the context
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": f"""Based on this product catalog information:

{context}

Answer this question: {query}

Be specific and reference the products you're talking about."""
                }
            ],
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": search_results
        }


def main():
    """Demo the multi-modal RAG system"""
    print("="*60)
    print("Multi-Modal RAG System Demo")
    print("="*60)

    # Initialize system
    rag = MultiModalRAG()

    # Index the data
    rag.index_data()

    print("\n" + "="*60)
    print("System ready! Testing queries...")
    print("="*60)

    # Test queries
    queries = [
        "Show me red products",
        "What furniture do you have?",
        "I need something for running",
        "What electronics are available?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        result = rag.answer_question(query)
        print(f"\nAnswer: {result['answer']}")

        print("\nTop matches:")
        print("Text:")
        for r in result['sources']['text_results'][:2]:
            print(f"  - {r['metadata']['name']} ({r['metadata']['color']})")
        print("Images:")
        for r in result['sources']['image_results'][:2]:
            print(f"  - {r['metadata']['filename']}")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()