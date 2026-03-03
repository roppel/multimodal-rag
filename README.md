# Multi-Modal RAG for E-Commerce Search

A production-oriented multi-modal RAG system for searching cheese products, demonstrating hybrid search architecture and systematic evaluation at scale.

## Overview

Built a multi-modal search system for 580 cheese products from Amazon, combining GPT-4 Vision for image analysis with semantic text search and metadata filtering. The project demonstrates scaling challenges, evaluation methodology considerations, and production ML patterns.

**Domain Context:** Leveraged experience from Cheese Express (e-commerce cheese retailer) to design realistic test queries and understand customer search patterns.

---

## Key Features

- **Multi-modal search**: Combines text descriptions with GPT-4 Vision analysis of product images
- **Hybrid search architecture**: Semantic embeddings + metadata filtering (category, price, region, cheese type)
- **Automatic filter extraction**: LLM parses natural language queries into structured filters
- **GPT-based categorization**: Categorizes products by type, region, and category with higher accuracy than keyword matching
- **Persistent vector storage**: ChromaDB with disk persistence for production-like deployment
- **Systematic evaluation**: F1, precision, and recall metrics across multiple query types

---

## Architecture

### Search Pipeline

```
User Query: "French cheese under $50"
    ↓
1. Filter Extraction (GPT-4o-mini)
   → {region: "french", price: {$lt: 50}}
    ↓
2. Hybrid Search
   a. Metadata Filter: region="french" AND price<$50
   b. Semantic Search: embed(query) → find similar products
   c. Combine: Top K results from filtered set
    ↓
3. Results: Ranked products matching both filters and semantic similarity
```

### Components

- **Text Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vision Analysis**: GPT-4 Vision for product image descriptions
- **Vector DB**: ChromaDB with persistent storage
- **Categorization**: GPT-4o-mini for product classification
- **Filter Extraction**: GPT-4o-mini for query parsing

---

## Project Evolution

### Phase 1: Baseline (10 Products)
- Manual product descriptions
- Pure semantic search
- **Result: 59% F1**
- **Learning**: Works well at small scale, but limited by manual data curation

### Phase 2: Scaling Challenge (580 Products)
- Real Amazon product data
- Keyword-based categorization
- **Result: 17% F1**
- **Learning**: Categorization quality is critical; keyword matching fails on diverse product titles

### Phase 3: GPT Categorization (Final)
- GPT-based product categorization
- Multi-modal image analysis
- **Result: 13.5% F1 (53% precision, 16% recall)**
- **Learning**: Low recall is expected when showing top-5 from 580 products; metrics must match use case

---

## Results & Analysis

### Final Performance
```
Metric                    Value
─────────────────────────────────
Precision@5              52.9%
Recall                   16.2%
F1 Score                 13.5%

Query Type Performance:
- Gift baskets:          15% F1 (100% precision, 8% recall)
- Specific cheese:       6-50% F1 (varies by availability)
- Regional search:       0-35% F1 (depends on dataset composition)
- Price filtering:       2% F1 (high precision, very low recall)
```

### Key Findings

**GPT Categorization Impact:**
- Found 5-10x more relevant products than keyword matching
- 154 cheddars identified (vs 27 with keywords)
- 61 gift baskets (vs 2-8 with keywords)
- 76 accessories (vs 0 with keywords)

**Precision vs Recall Tradeoff:**
- Returning 5 results from 154 cheddars yields 3% recall (unavoidable)
- Real e-commerce faces same challenge: Amazon doesn't show all 10,000 products
- Precision@5 (53%) more meaningful metric than total recall

**What Worked:**
- ✓ Hybrid search (filters + semantic) prevents irrelevant results
- ✓ GPT categorization dramatically improved product classification
- ✓ Multi-modal (text + vision) provided richer search context

**What Didn't:**
- ✗ F1 metric designed for small datasets doesn't translate to large catalogs
- ✗ Semantic search alone insufficient without metadata filters
- ✗ Low recall is structural (5 results / 580 products) not fixable

---

## Technical Implementation

### Data Processing
1. Downloaded 580 cheese products from Amazon
2. GPT-4 Vision analysis of all product images (~$1.50)
3. GPT-4o-mini categorization (~$0.50)
4. Created embeddings for text + visual descriptions

### Search Implementation
```python
# Hybrid search: filters + semantic similarity
results = rag.smart_search(
    query="aged cheddar under $50",
    n_results=5
)
# Auto-extracts: {cheese_type: "cheddar", price: {$lt: 50}}
# Filters products, then semantic search within results
```

### Evaluation Methodology
- 14 test queries covering gift baskets, specific cheeses, regional searches, price filtering
- Measured precision, recall, F1 on top-5 results
- Compared keyword vs GPT categorization approaches

---

## What I Learned

### Scaling Challenges
- Small datasets (10 products) hide problems that emerge at scale (580 products)
- Categorization quality becomes critical with diverse, messy real-world data
- Amazon product titles are inconsistent, requiring smarter classification than keywords

### Evaluation Design
- Traditional F1 metrics don't match large catalog e-commerce use cases
- Precision@K more meaningful than total recall when showing limited results
- Evaluation methodology must align with actual user experience

### Production ML Patterns
- Hybrid search (exact filters + semantic) beats pure ML for structured data
- GPT categorization cost (~$0.001/product) worthwhile for quality improvement
- Persistent vector storage essential for production deployment

### Real-World Tradeoffs
- Perfect recall impossible when catalog >> results shown
- User experience optimizes for precision in top results, not finding everything
- Metrics should reflect business goals, not academic benchmarks

---

## Tech Stack

- **Language Models**: OpenAI GPT-4 Vision, GPT-4o-mini
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2, 384 dimensions)
- **Vector Database**: ChromaDB (persistent storage)
- **Language**: Python 3.9
- **Dataset**: 580 Amazon cheese products with images

---

## Setup & Usage

### Installation
```bash
pip install -r requirements.txt
```

### Index Products (One-time)
```bash
# Download and index 580 products (~25 min, ~$2 in API costs)
python cheese_rag_gpt.py
```

### Run Evaluation
```bash
# Test system performance
python eval_cheese_final.py
```

### Example Usage
```python
from cheese_rag_gpt import CheeseRAGWithGPT

rag = CheeseRAGWithGPT()

# Search with automatic filter extraction
results = rag.smart_search("French cheese under $50", n_results=5)

# Returns:
# {
#   'query': 'French cheese under $50',
#   'filters': {'region': 'french', 'price': {'$lt': 50}},
#   'text_results': [...],
#   'image_results': [...]
# }
```

---

## Project Structure

```
multimodal-rag/
├── main.py                          # Core MultiModalRAG base class
├── cheese_rag_gpt.py               # Indexing with GPT categorization
├── eval_cheese_final.py            # Evaluation script
├── README.md                        # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes chroma_db/, venv/, etc.
│
├── cheese_data/
│   ├── descriptions.json           # Original 580 product metadata
│   ├── categorized_gpt_indexed.json # GPT-categorized products
│   └── images/                     # 580 product images (not in git)
│
└── chroma_db/                      # Persistent vector storage (not in git)
```

---

## Future Improvements

If continuing this project, potential enhancements:

1. **Reranking**: Two-stage retrieval (retrieve 20, rerank to top 5) could improve precision
2. **Better embeddings**: Larger models (768-dim) or domain-specific embeddings
3. **Query expansion**: "aged" → ["aged", "mature", "sharp", "extra sharp"]
4. **Manual labeling**: Verify/correct categorization for 100-200 key products
5. **Different metrics**: NDCG, MRR better suited for ranking evaluation
6. **Larger dataset**: 1000+ products to surface additional scaling challenges

---

## Cost Analysis

### Development Costs
- GPT-4 Vision (580 images): ~$1.50
- GPT-4o-mini categorization (580 products): ~$0.50
- Filter extraction (testing): ~$0.20
- **Total: ~$2.20**

### Production Estimate (hypothetical)
- Monthly searches: 10,000
- Filter extraction: $0.20/1000 queries = $2/month
- Vector DB: ChromaDB (self-hosted, free) or Pinecone ($0-70/month)
- **Estimated: $2-72/month depending on scale**

---

## Lessons for Production

1. **Hybrid > Pure ML**: Combining exact filters with semantic search prevents bad results
2. **Categorization Quality Matters**: Invest in good product classification upfront
3. **Metrics Must Match Use Case**: F1 designed for complete retrieval doesn't fit top-K search
4. **Real Data is Messy**: Amazon titles inconsistent; robust categorization essential
5. **User Experience ≠ Perfect Recall**: Showing 5 great results > 50 mediocre ones

---

## Acknowledgments

- Dataset: Amazon product metadata (UCSD)
- Domain expertise: Cheese Express e-commerce experience
- Inspiration: Friend's production RAG system for code analysis

---

## License

MIT License - free to use, modify, and distribute.