# Multi-Modal RAG with Intelligent Search

A production-ready Retrieval-Augmented Generation system that searches across images and text with automatic query understanding and hybrid search architecture.

## Overview

This project demonstrates advanced RAG patterns through iterative improvement:
1. **Baseline**: Semantic search only (59% F1)
2. **Hybrid Search**: Added metadata filtering (86% F1, +27 points)
3. **Smart Search**: Automatic filter extraction (83% F1, end-to-end)

The system uses GPT-4 Vision to understand images, combines semantic embeddings with structured metadata, and automatically extracts search filters from natural language queries.

## Key Features

- ✅ **Multi-modal search** across images and text descriptions
- ✅ **Vision AI integration** using GPT-4 Vision for image analysis
- ✅ **Automatic filter extraction** from natural language queries
- ✅ **Hybrid search architecture** combining exact matching + semantic search
- ✅ **Systematic evaluation** with precision, recall, and F1 metrics
- ✅ **Production-ready** error handling and modular design

## Architecture

```
User Query: "Show me red shoes under $150"
    ↓
Filter Extraction (GPT-4 mini)
    → Extracts: {color: "red", category: "footwear", price: {$lt: 150}}
    ↓
Hybrid Search
    → Step 1: Filter products by metadata (exact match)
    → Step 2: Semantic search within filtered results (embeddings)
    ↓
Results: Red running shoes ($120) ✓
```

### Components

1. **Vision Analysis**: GPT-4 Vision generates detailed descriptions of product images
2. **Filter Extraction**: GPT-4 mini extracts structured filters from natural language
3. **Dual Embeddings**: Separate vector spaces for text metadata and visual descriptions
4. **Hybrid Search**: Combines metadata filters (category, color, price) with semantic search
5. **Evaluation Framework**: Systematic testing with ground truth labels

## Performance Evolution

### Baseline (Semantic Search Only)
```
Average F1: 59%
- Category queries: 100%
- Visual features: 63%
- Color queries: 53%
- Use case queries: 43%
```

**Problem**: Semantic search alone couldn't handle exact requirements like "white products" or "under $300"

### Hybrid Search (Manual Filters)
```
Average F1: 86% (+27 points)
- Category queries: 100%
- Color + category: 100%
- Price filtering: 100%
- Color queries: 82%
```

**Improvement**: Combining exact metadata filters with semantic search dramatically improved accuracy

### Smart Search (Auto Filter Extraction)
```
Average F1: 83% (+24 points from baseline)
- Category queries: 100%
- Color + category: 100%
- Price filtering: 83%
- Color queries: 83%
```

**Achievement**: Fully automatic end-to-end system with minimal accuracy drop vs manual filters

## Installation

```bash
# Clone repository
git clone https://github.com/roppel/multimodal-rag.git
cd multimodal-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. **Get OpenAI API key** from https://platform.openai.com/api-keys

2. **Set environment variable:**
```bash
export OPENAI_API_KEY='your-key-here'
```

3. **Download product images** (or use your own):
```bash
python download_real_images.py
```

## Usage

### Basic Search

```python
from main import MultiModalRAG

rag = MultiModalRAG()
rag.index_data()

# Semantic search only
results = rag.search("running shoes", n_results=3)

# Hybrid search with manual filters
results = rag.hybrid_search(
    "athletic footwear",
    filters={"category": "footwear", "color": "red"}
)

# Smart search with automatic filter extraction
results = rag.smart_search("Show me red shoes under $150")
# Automatically extracts: {color: "red", category: "footwear", price: {$lt: 150}}
```

### Run Evaluations

```bash
# Baseline semantic search
python eval.py

# Hybrid search with manual filters
python eval_hybrid.py

# Smart search with auto filter extraction
python eval_smart.py
```

## Example: Auto Filter Extraction

```python
Query: "Show me red footwear"
→ Filters: {'color': 'red', 'category': 'footwear'}
→ Result: red_running_shoes.jpg (100% accuracy)

Query: "What electronics under $300?"
→ Filters: {'category': 'electronics', 'price': {'$lt': 300}}
→ Result: white_headphones.jpg (100% accuracy)

Query: "white products"
→ Filters: {'color': 'white'}
→ Results: white_sneakers.jpg, white_headphones.jpg (100% accuracy)
```

## Project Structure

```
multimodal-rag/
├── main.py                      # Core multi-modal RAG system
├── eval.py                      # Baseline evaluation
├── eval_hybrid.py               # Hybrid search evaluation
├── eval_smart.py                # Smart search evaluation
├── download_real_images.py      # Dataset preparation
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── images/                      # Product images
│   ├── red_running_shoes.jpg
│   ├── white_sneakers.jpg
│   └── ...
├── data/
│   └── descriptions.json        # Product metadata
├── eval_results.json            # Baseline metrics
├── hybrid_eval_results.json     # Hybrid search metrics
└── smart_eval_results.json      # Smart search metrics
```

## Technical Details

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Dimension**: 384
- **Use**: Both text and visual descriptions

### Vision Model
- **Model**: GPT-4 Vision (gpt-4o)
- **Purpose**: Generate visual descriptions from images
- **Output**: Detailed text descriptions of visual features

### Filter Extraction
- **Model**: GPT-4 mini (gpt-4o-mini)
- **Purpose**: Extract structured filters from natural language
- **Output**: JSON with category, color, price filters

### Vector Database
- **Database**: ChromaDB (in-memory)
- **Collections**: Separate for text and images
- **Search**: Cosine similarity with metadata filtering

## What I Learned

### Technical Insights

1. **Hybrid > Pure ML**: Combining exact filters with semantic search improves accuracy by 27 points
2. **Automatic extraction works**: LLM-based filter extraction achieves 83% F1 with minimal drop vs manual
3. **Multi-modal embeddings**: Text descriptions of images work well for visual search
4. **Evaluation is critical**: Systematic testing reveals where systems fail

### Key Takeaways

- **Don't over-rely on AI**: Use structured data (metadata) when available
- **Iterate based on metrics**: Each improvement was driven by evaluation results
- **Production patterns matter**: Hybrid search + auto extraction = production-ready
- **Vision AI is practical**: GPT-4 Vision accurately describes images for semantic search

## Improvements for Production

To achieve even higher accuracy:

1. **Better filter extraction**: Fine-tune a model specifically for query parsing
2. **Expanded metadata**: Add more structured fields (brand, material, size)
3. **Reranking**: Use a cross-encoder to rerank final results
4. **User feedback**: Learn from user interactions to improve over time
5. **Larger dataset**: More products would better demonstrate scalability

## Real-World Applications

This architecture applies to:
- **E-commerce**: Search products by appearance and specifications
- **Digital asset management**: Find images and documents semantically
- **Healthcare**: Search medical images with accompanying reports
- **Real estate**: Find properties by visual features and structured data
- **Manufacturing**: Locate parts by photos and specifications

## Performance Metrics

| Approach | Precision | Recall | F1 Score | Improvement |
|----------|-----------|--------|----------|-------------|
| Baseline (semantic) | 50.0% | 78.6% | 58.6% | - |
| Hybrid (manual) | 87.5% | 87.5% | 85.8% | +27 pts |
| Smart (auto) | 90.0% | 80.0% | 83.3% | +24 pts |

## Cost Analysis

**Development/Testing** (10 products, 30 queries):
- GPT-4 Vision (image analysis): ~$0.30
- GPT-4 mini (filter extraction): ~$0.05
- Embedding generation: Free (local)
- **Total**: ~$0.35

**Production Scaling** (1000 products, 10K queries/month):
- Initial indexing: ~$30
- Monthly queries: ~$15
- **Total**: ~$45/month

## Technologies Used

- **Python 3.9+**
- **OpenAI API** (GPT-4 Vision, GPT-4 mini)
- **SentenceTransformers** (embeddings)
- **ChromaDB** (vector database)
- **Pillow** (image processing)

## License

MIT

## Author

Built as a learning project to explore multi-modal RAG patterns, hybrid search architectures, and ML evaluation methodology.

---

**Key Achievement**: Improved search accuracy from 59% to 83% F1 through systematic iteration: baseline semantic search → hybrid architecture with metadata filtering → automatic filter extraction from natural language. Demonstrates production-ready ML engineering with emphasis on evaluation-driven improvement.