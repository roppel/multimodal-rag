"""
Evaluation Framework for Multi-Modal RAG
Measures accuracy, precision, and recall on known test queries
"""
from main import MultiModalRAG
import json

# Evaluation dataset: queries with known correct answers
EVAL_SET = [
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
        "query": "What can I use for running?",
        "expected_items": ["red_running_shoes.jpg"],
        "category": "use case"
    },
    {
        "query": "Show me office equipment",
        "expected_items": ["black_office_chair.jpg", "silver_laptop.jpg"],
        "category": "use case"
    },
    {
        "query": "What silver or gray products do you have?",
        "expected_items": ["silver_laptop.jpg", "gray_backpack.jpg"],
        "category": "color"
    },
    {
        "query": "Show me footwear",
        "expected_items": ["red_running_shoes.jpg", "white_sneakers.jpg", "brown_boots.jpg"],
        "category": "category"
    },
    {
        "query": "What can I use for listening to music?",
        "expected_items": ["white_headphones.jpg"],
        "category": "use case"
    },
    # Visual-only queries (not in text descriptions)
    {
        "query": "Show me items with laces",
        "expected_items": ["red_running_shoes.jpg", "white_sneakers.jpg", "brown_boots.jpg"],
        "category": "visual feature"
    },
    {
        "query": "What has a screen?",
        "expected_items": ["silver_laptop.jpg", "smartphone.jpg"],
        "category": "visual feature"
    },
    {
        "query": "Show me items with straps",
        "expected_items": ["gray_backpack.jpg"],
        "category": "visual feature"
    },
    {
        "query": "What products are shiny or metallic?",
        "expected_items": ["silver_laptop.jpg", "smartphone.jpg"],
        "category": "visual property"
    },
]

def calculate_metrics(predicted, expected):
    """Calculate precision, recall, F1"""
    predicted_set = set(predicted)
    expected_set = set(expected)

    # True positives: predicted AND expected
    tp = len(predicted_set & expected_set)

    # False positives: predicted but NOT expected
    fp = len(predicted_set - expected_set)

    # False negatives: expected but NOT predicted
    fn = len(expected_set - predicted_set)

    # Precision: of what we returned, how much was correct?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall: of what should be returned, how much did we find?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1: harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def run_evaluation(rag, top_k=3):
    """Run full evaluation on the eval set"""
    print("\n" + "="*70)
    print("MULTI-MODAL RAG EVALUATION")
    print("="*70)

    results = []

    for i, test_case in enumerate(EVAL_SET, 1):
        query = test_case["query"]
        expected = test_case["expected_items"]
        category = test_case["category"]

        print(f"\n[{i}/{len(EVAL_SET)}] Testing: {query}")
        print(f"Expected: {expected}")

        # Search using multi-modal RAG
        search_results = rag.search(query, n_results=top_k)

        # Get predicted items (from both text and image results)
        predicted = []
        for r in search_results['text_results']:
            predicted.append(r['metadata']['filename'])
        for r in search_results['image_results']:
            predicted.append(r['metadata']['filename'])

        # Remove duplicates
        predicted = list(set(predicted))[:top_k]

        print(f"Predicted: {predicted}")

        # Calculate metrics
        metrics = calculate_metrics(predicted, expected)

        print(f"Precision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f} | F1: {metrics['f1']:.2f}")

        results.append({
            "query": query,
            "category": category,
            "expected": expected,
            "predicted": predicted,
            "metrics": metrics
        })

    return results

def analyze_results(results):
    """Analyze overall performance"""
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)

    # Overall metrics
    total_precision = sum(r['metrics']['precision'] for r in results) / len(results)
    total_recall = sum(r['metrics']['recall'] for r in results) / len(results)
    total_f1 = sum(r['metrics']['f1'] for r in results) / len(results)

    print(f"\nAverage Precision: {total_precision:.2%}")
    print(f"Average Recall: {total_recall:.2%}")
    print(f"Average F1 Score: {total_f1:.2%}")

    # Performance by category
    print("\n" + "-"*70)
    print("PERFORMANCE BY QUERY TYPE")
    print("-"*70)

    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result['metrics']['f1'])

    for cat, f1_scores in sorted(categories.items()):
        avg_f1 = sum(f1_scores) / len(f1_scores)
        print(f"{cat:20s}: {avg_f1:.2%} (n={len(f1_scores)})")

    # Best and worst performing queries
    print("\n" + "-"*70)
    print("BEST PERFORMING QUERIES")
    print("-"*70)

    sorted_results = sorted(results, key=lambda x: x['metrics']['f1'], reverse=True)
    for result in sorted_results[:3]:
        print(f"✓ {result['query']:40s} F1: {result['metrics']['f1']:.2%}")

    print("\n" + "-"*70)
    print("WORST PERFORMING QUERIES")
    print("-"*70)

    for result in sorted_results[-3:]:
        print(f"✗ {result['query']:40s} F1: {result['metrics']['f1']:.2%}")
        print(f"  Expected: {result['expected']}")
        print(f"  Got: {result['predicted']}\n")

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump({
            "overall": {
                "precision": total_precision,
                "recall": total_recall,
                "f1": total_f1
            },
            "by_category": {cat: sum(scores)/len(scores) for cat, scores in categories.items()},
            "detailed_results": results
        }, f, indent=2)

    print("\n" + "="*70)
    print("Results saved to: eval_results.json")
    print("="*70)

def main():
    print("Initializing Multi-Modal RAG system...")
    rag = MultiModalRAG()

    print("\nIndexing products...")
    rag.index_data()

    print("\nRunning evaluation...")
    results = run_evaluation(rag, top_k=3)

    analyze_results(results)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nKey Insights:")
    print("1. How accurate is the system overall?")
    print("2. Which query types work best?")
    print("3. Where does it struggle?")
    print("4. Text-based vs visual-based queries?")

if __name__ == "__main__":
    main()