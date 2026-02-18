"""
Hybrid Search Evaluation
Compare original semantic-only search vs hybrid (semantic + metadata filters)
"""
from main import MultiModalRAG
import json

# Test cases with explicit filters
HYBRID_EVAL_SET = [
    {
        "query": "Show me red footwear",
        "filters": {"color": "red", "category": "footwear"},
        "expected_items": ["red_running_shoes.jpg"],
        "category": "color + category"
    },
    {
        "query": "What furniture is available?",
        "filters": {"category": "furniture"},
        "expected_items": ["black_office_chair.jpg", "blue_armchair.jpg", "wooden_table.jpg"],
        "category": "category"
    },
    {
        "query": "Show me white products",
        "filters": {"color": "white"},
        "expected_items": ["white_sneakers.jpg", "white_headphones.jpg"],
        "category": "color"
    },
    {
        "query": "What electronics under $300?",
        "filters": {"category": "electronics", "price": {"$lt": 300}},
        "expected_items": ["white_headphones.jpg"],
        "category": "category + price"
    },
    {
        "query": "Show me brown items",
        "filters": {"color": "brown"},
        "expected_items": ["brown_boots.jpg", "wooden_table.jpg"],
        "category": "color"
    },
    {
        "query": "Show me office equipment",
        "filters": {"category": "furniture"},  # Partial match
        "expected_items": ["black_office_chair.jpg", "silver_laptop.jpg"],
        "category": "use case"
    },
    {
        "query": "What silver or gray products do you have?",
        "filters": None,  # Multiple colors, can't filter
        "expected_items": ["silver_laptop.jpg", "gray_backpack.jpg"],
        "category": "color"
    },
    {
        "query": "Show me footwear",
        "filters": {"category": "footwear"},
        "expected_items": ["red_running_shoes.jpg", "white_sneakers.jpg", "brown_boots.jpg"],
        "category": "category"
    },
]

def calculate_metrics(predicted, expected):
    """Calculate precision, recall, F1"""
    predicted_set = set(predicted)
    expected_set = set(expected)
    
    tp = len(predicted_set & expected_set)
    fp = len(predicted_set - expected_set)
    fn = len(expected_set - predicted_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def run_hybrid_evaluation(rag, top_k=3):
    """Run evaluation with hybrid search"""
    print("\n" + "="*70)
    print("HYBRID SEARCH EVALUATION")
    print("="*70)
    
    results = []
    
    for i, test_case in enumerate(HYBRID_EVAL_SET, 1):
        query = test_case["query"]
        filters = test_case["filters"]
        expected = test_case["expected_items"]
        category = test_case["category"]
        
        print(f"\n[{i}/{len(HYBRID_EVAL_SET)}] Testing: {query}")
        if filters:
            print(f"Filters: {filters}")
        print(f"Expected: {expected}")
        
        # Hybrid search with filters
        search_results = rag.hybrid_search(query, n_results=top_k, filters=filters)
        
        # Get predicted items
        predicted = []
        for r in search_results['text_results']:
            predicted.append(r['metadata']['filename'])
        for r in search_results['image_results']:
            predicted.append(r['metadata']['filename'])
        
        predicted = list(set(predicted))[:top_k]
        
        print(f"Predicted: {predicted}")
        
        # Calculate metrics
        metrics = calculate_metrics(predicted, expected)
        
        print(f"Precision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f} | F1: {metrics['f1']:.2f}")
        
        results.append({
            "query": query,
            "filters": filters,
            "category": category,
            "expected": expected,
            "predicted": predicted,
            "metrics": metrics
        })
    
    return results

def compare_results(hybrid_results):
    """Analyze hybrid search performance"""
    print("\n" + "="*70)
    print("HYBRID SEARCH RESULTS")
    print("="*70)
    
    total_precision = sum(r['metrics']['precision'] for r in hybrid_results) / len(hybrid_results)
    total_recall = sum(r['metrics']['recall'] for r in hybrid_results) / len(hybrid_results)
    total_f1 = sum(r['metrics']['f1'] for r in hybrid_results) / len(hybrid_results)
    
    print(f"\nAverage Precision: {total_precision:.2%}")
    print(f"Average Recall: {total_recall:.2%}")
    print(f"Average F1 Score: {total_f1:.2%}")
    
    print("\n" + "-"*70)
    print("IMPROVEMENT vs BASELINE")
    print("-"*70)
    print("Baseline (semantic-only):  59% F1")
    print(f"Hybrid (semantic+filter):  {total_f1:.0%} F1")
    print(f"Improvement:               +{(total_f1 - 0.59)*100:.0f} percentage points")
    
    # Performance by category
    print("\n" + "-"*70)
    print("PERFORMANCE BY QUERY TYPE")
    print("-"*70)
    
    categories = {}
    for result in hybrid_results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result['metrics']['f1'])
    
    for cat, f1_scores in sorted(categories.items()):
        avg_f1 = sum(f1_scores) / len(f1_scores)
        print(f"{cat:20s}: {avg_f1:.2%} (n={len(f1_scores)})")
    
    # Save results
    with open("hybrid_eval_results.json", "w") as f:
        json.dump({
            "overall": {
                "precision": total_precision,
                "recall": total_recall,
                "f1": total_f1,
                "improvement_over_baseline": (total_f1 - 0.59) * 100
            },
            "by_category": {cat: sum(scores)/len(scores) for cat, scores in categories.items()},
            "detailed_results": hybrid_results
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("Results saved to: hybrid_eval_results.json")
    print("="*70)

def main():
    print("Initializing Multi-Modal RAG with Hybrid Search...")
    rag = MultiModalRAG()
    
    print("\nIndexing products...")
    rag.index_data()
    
    print("\nRunning hybrid search evaluation...")
    results = run_hybrid_evaluation(rag, top_k=3)
    
    compare_results(results)
    
    print("\n" + "="*70)
    print("HYBRID SEARCH EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
