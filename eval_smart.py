"""
Smart Search Evaluation - Automatic Filter Extraction
Tests the end-to-end system where filters are automatically extracted from queries
"""
from main import MultiModalRAG
import json

# Test cases WITHOUT manual filters - the system should figure them out
SMART_EVAL_SET = [
    {
        "query": "Show me red footwear",
        "expected_items": ["red_running_shoes.jpg"],
        "category": "color + category"
    },
    {
        "query": "What furniture is available?",
        "expected_items": ["black_office_chair.jpg", "blue_armchair.jpg", "wooden_table.jpg"],
        "category": "category"
    },
    {
        "query": "Show me white products",
        "expected_items": ["white_sneakers.jpg", "white_headphones.jpg"],
        "category": "color"
    },
    {
        "query": "What electronics under $300?",
        "expected_items": ["white_headphones.jpg"],
        "category": "category + price"
    },
    {
        "query": "Show me brown items",
        "expected_items": ["brown_boots.jpg", "wooden_table.jpg"],
        "category": "color"
    },
    {
        "query": "Show me footwear",
        "expected_items": ["red_running_shoes.jpg", "white_sneakers.jpg", "brown_boots.jpg"],
        "category": "category"
    },
    {
        "query": "I need something for the office",
        "expected_items": ["black_office_chair.jpg", "silver_laptop.jpg"],
        "category": "use case"
    },
    {
        "query": "What do you have in electronics?",
        "expected_items": ["silver_laptop.jpg", "smartphone.jpg", "white_headphones.jpg"],
        "category": "category"
    },
    {
        "query": "Show me accessories under $200",
        "expected_items": ["white_headphones.jpg", "gray_backpack.jpg"],
        "category": "category + price"
    },
    {
        "query": "red shoes",
        "expected_items": ["red_running_shoes.jpg"],
        "category": "color + category"
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

def run_smart_evaluation(rag, top_k=3):
    """Run evaluation with automatic filter extraction"""
    print("\n" + "="*70)
    print("SMART SEARCH EVALUATION (Automatic Filter Extraction)")
    print("="*70)
    
    results = []
    
    for i, test_case in enumerate(SMART_EVAL_SET, 1):
        query = test_case["query"]
        expected = test_case["expected_items"]
        category = test_case["category"]
        
        print(f"\n[{i}/{len(SMART_EVAL_SET)}] Query: {query}")
        print(f"Expected: {expected}")
        
        # Smart search - automatically extracts filters
        search_results = rag.smart_search(query, n_results=top_k)
        
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
            "auto_filters": search_results.get('filters'),
            "category": category,
            "expected": expected,
            "predicted": predicted,
            "metrics": metrics
        })
    
    return results

def compare_results(smart_results):
    """Analyze smart search performance"""
    print("\n" + "="*70)
    print("SMART SEARCH RESULTS")
    print("="*70)
    
    total_precision = sum(r['metrics']['precision'] for r in smart_results) / len(smart_results)
    total_recall = sum(r['metrics']['recall'] for r in smart_results) / len(smart_results)
    total_f1 = sum(r['metrics']['f1'] for r in smart_results) / len(smart_results)
    
    print(f"\nAverage Precision: {total_precision:.2%}")
    print(f"Average Recall: {total_recall:.2%}")
    print(f"Average F1 Score: {total_f1:.2%}")
    
    print("\n" + "-"*70)
    print("PROGRESSION")
    print("-"*70)
    print("Baseline (semantic-only):        59% F1")
    print("Hybrid (manual filters):         86% F1")
    print(f"Smart (auto filter extraction):  {total_f1:.0%} F1")
    
    # Performance by category
    print("\n" + "-"*70)
    print("PERFORMANCE BY QUERY TYPE")
    print("-"*70)
    
    categories = {}
    for result in smart_results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result['metrics']['f1'])
    
    for cat, f1_scores in sorted(categories.items()):
        avg_f1 = sum(f1_scores) / len(f1_scores)
        print(f"{cat:20s}: {avg_f1:.2%} (n={len(f1_scores)})")
    
    # Show filter extraction examples
    print("\n" + "-"*70)
    print("FILTER EXTRACTION EXAMPLES")
    print("-"*70)
    for result in smart_results[:5]:
        print(f"Query: {result['query']}")
        print(f"  → Filters: {result['auto_filters']}")
    
    # Save results
    with open("smart_eval_results.json", "w") as f:
        json.dump({
            "overall": {
                "precision": total_precision,
                "recall": total_recall,
                "f1": total_f1,
            },
            "progression": {
                "baseline_semantic": 0.59,
                "hybrid_manual": 0.86,
                "smart_auto": total_f1
            },
            "by_category": {cat: sum(scores)/len(scores) for cat, scores in categories.items()},
            "detailed_results": smart_results
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("Results saved to: smart_eval_results.json")
    print("="*70)

def main():
    print("Initializing Multi-Modal RAG with Smart Search...")
    rag = MultiModalRAG()
    
    print("\nIndexing products...")
    rag.index_data()
    
    print("\nRunning smart search evaluation (with auto filter extraction)...")
    results = run_smart_evaluation(rag, top_k=3)
    
    compare_results(results)
    
    print("\n" + "="*70)
    print("SMART SEARCH EVALUATION COMPLETE")
    print("="*70)
    print("\nThe system now automatically extracts filters from natural language!")
    print("Try it yourself: rag.smart_search('Show me red products under $100')")

if __name__ == "__main__":
    main()
