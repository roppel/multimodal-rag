# Multi-Modal RAG System

A production-ready Retrieval-Augmented Generation system that searches across both images and text using vision models and semantic embeddings.

## Overview

This project demonstrates advanced RAG patterns by combining:
- **Vision AI** (GPT-4 Vision) for image understanding
- **Text embeddings** for semantic search
- **Multi-modal retrieval** across different data types
- **Systematic evaluation** with precision/recall metrics

## Architecture

```
Query → Embedding Model
         ↓
    Vector Search (ChromaDB)
         ↓
    [Text Results] + [Image Results]
         ↓
    LLM (GPT-4) → Answer
```

### Components

1. **Image Processing**: GPT-4 Vision analyzes product images and generates detailed visual descriptions
2. **Dual Embeddings**: Separate vector spaces for text metadata and visual descriptions
3. **Multi-Modal Search**: Queries search both text and image embeddings simultaneously
4. **RAG Pipeline**: Retrieved context feeds into LLM for natural language responses

## Features

- ✅ Real image understanding (not just metadata)
- ✅ Semantic search across text and visual data
- ✅ Configurable search modes (text-only, image-only, or both)
- ✅ Systematic evaluation framework with metrics
- ✅ Production-ready error handling and logging

## Performance Metrics

Based on 14 test queries across different categories:

| Metric | Score |
|--------|-------|
| **Precision** | 50.0% |
| **Recall** | 78.6% |
| **F1 Score** | 58.6% |

### Performance by Query Type

| Query Type | F1 Score | Examples |
|------------|----------|----------|
| Category-based | 100% | "Show me footwear", "What furniture?" |
| Visual features | 63% | "Items with laces", "Products with straps" |
| Color-based | 53% | "Show me red products" |
| Use case | 43% | "What can I use for running?" |

### Key Insights

**Strengths:**
- Excellent at direct category matching
- Strong visual feature detection (laces, straps)
- Perfect recall on well-defined categories

**Limitations:**
- Struggles with abstract use-case inference ("music" → headphones)
- Moderate accuracy on subtle visual properties (shiny, metallic)
- Needs improvement on complex color queries

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd multimodal-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. **Get OpenAI API key** from https://platform.openai.com/api-keys

2. **Set environment variable:**
```bash
export OPENAI_API_KEY='your-key-here'
```

3. **Download product images:**
```bash
python download_real_images.py
```

## Usage

### Basic Demo

```bash
python main.py
```

Runs the system with pre-defined test queries and shows results.

### Run Evaluation

```bash
python eval.py
```

Evaluates system performance on 14 test queries with ground truth labels. Outputs:
- Per-query precision, recall, F1
- Overall metrics
- Performance breakdown by query type
- Results saved to `eval_results.json`

### Custom Queries

```python
from main import MultiModalRAG

rag = MultiModalRAG()
rag.index_data()

# Search across both text and images
results = rag.search("Show me red shoes", n_results=3)

# Text-only search
results = rag.search("electronics", search_images=False)

# Get natural language answer
response = rag.answer_question("What furniture do you have?")
print(response['answer'])
```

## Project Structure

```
multimodal-rag/
├── main.py                    # Core multi-modal RAG system
├── eval.py                    # Evaluation framework
├── download_real_images.py    # Dataset preparation
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── images/                    # Product images
│   ├── red_running_shoes.jpg
│   ├── white_sneakers.jpg
│   └── ...
├── data/
│   └── descriptions.json      # Product metadata
└── eval_results.json          # Evaluation metrics
```

## Technical Details

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Dimension**: 384
- **Speed**: ~1ms per embedding
- **Use**: Both text and visual descriptions

### Vision Model
- **Model**: GPT-4 Vision (gpt-4o)
- **Purpose**: Generate visual descriptions from images
- **Processing**: ~2-3 seconds per image
- **Output**: Detailed text descriptions of visual features

### Vector Database
- **Database**: ChromaDB (in-memory)
- **Collections**: Separate for text and images
- **Search**: Cosine similarity
- **Retrieval**: Configurable top-k results

## What I Learned

### Technical Skills
- How vision models process and understand images
- Multi-modal embedding strategies
- Evaluation methodology for AI systems
- Production RAG architecture patterns

### Key Takeaways
1. **Multi-modal RAG isn't magic** - works best for direct visual matching, struggles with abstract reasoning
2. **Evaluation is critical** - measuring accuracy reveals where systems succeed/fail
3. **Hybrid approaches win** - combining structured metadata with semantic search improves results
4. **Vision models are good but imperfect** - catch obvious features, miss subtle details

## Improvements for Production

To achieve higher accuracy, consider:

1. **Hybrid search**: Combine vector search with structured filters (price, category)
2. **Query classification**: Route queries to text-only vs multi-modal search based on type
3. **Fine-tuned embeddings**: Train custom embedding model on domain-specific data
4. **Ensemble retrieval**: Combine multiple retrieval strategies and re-rank
5. **User feedback loop**: Learn from corrections to improve over time

## Use Cases

This architecture applies to:
- **E-commerce**: Search products by appearance and description
- **Digital asset management**: Find images and documents semantically
- **Healthcare**: Search medical images with accompanying reports
- **Real estate**: Find properties by visual features and specs
- **Manufacturing**: Locate parts by photos and specifications

## Cost

Running the evaluation (~15 images, 14 queries):
- **GPT-4 Vision calls**: ~$0.20
- **Embedding generation**: Free (local)
- **Storage**: Negligible

Production scaling:
- 1000 images: ~$2-3 for initial indexing
- 10K queries/month: ~$5-10 in API costs

## License

MIT

## Author

Built as a learning project to explore multi-modal RAG patterns and ML evaluation methodology.

---

**Key Learning**: Multi-modal RAG achieves 59% F1 on diverse queries. Category-based queries reach 100% accuracy, while abstract use-case inference needs improvement. Systematic evaluation reveals that combining vision AI with structured metadata is the path to production-ready accuracy.