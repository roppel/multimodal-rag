from main import MultiModalRAG

# Load the already-indexed system
rag = MultiModalRAG()
rag.index_data()

# Test queries that require actual visual understanding
test_queries = [
    "Show me products with laces",  # Only visible in images
    "What has a mesh pattern?",     # Texture, not in text
    "Show me items with straps",    # Visual feature
    "What products are shiny?",     # Surface property
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = rag.search(query, n_results=2)

    print("Visual matches:")
    for r in results['image_results']:
        print(f"  - {r['metadata']['filename']}")
        print(f"    Visual: {r['visual_description'][:100]}...")