"""
Final Evaluation - Compare Baseline (17% F1) vs GPT Categorization
"""
import json
from cheese_rag_gpt import CheeseRAGWithGPT

# Test queries
EVAL_QUERIES = [
    {"query": "cheese gift basket under $100", "filters": {"category": "gift_basket", "price": {"$lt": 100}}},
    {"query": "cheese gift basket for Christmas", "filters": {"category": "gift_basket"}},
    {"query": "cheese sampler gift set", "filters": {"category": "gift_basket"}},
    {"query": "aged cheddar", "filters": {"cheese_type": "cheddar"}},
    {"query": "Stilton cheese", "filters": {"cheese_type": "stilton"}},
    {"query": "Vermont cheddar", "filters": {"region": "american", "cheese_type": "cheddar"}},
    {"query": "English cheddar", "filters": {"region": "english", "cheese_type": "cheddar"}},
    {"query": "French cheese", "filters": {"region": "french"}},
    {"query": "Italian cheese", "filters": {"region": "italian"}},
    {"query": "Spanish cheese", "filters": {"region": "spanish"}},
    {"query": "cheese board", "filters": {"category": "accessory"}},
    {"query": "cheese knife set", "filters": {"category": "accessory"}},
    {"query": "cheese under $50", "filters": {"price": {"$lt": 50}}},
    {"query": "cheese of the month", "filters": {"category": "subscription"}},
]

def calculate_metrics(predicted_titles, expected_filters, all_products):
    """Calculate precision and recall"""
    if not expected_filters:
        return None

    # Find relevant products
    relevant_products = []
    for p in all_products:
        matches = True
        for key, value in expected_filters.items():
            if key == "price" and isinstance(value, dict):
                if "$lt" in value:
                    if not p.get('price') or p['price'] >= value["$lt"]:
                        matches = False
                        break
            else:
                if p.get(key) != value:
                    matches = False
                    break
        if matches:
            relevant_products.append(p['title'])

    correct = sum(1 for title in predicted_titles if title in relevant_products)

    precision = correct / len(predicted_titles) if predicted_titles else 0
    recall = correct / len(relevant_products) if relevant_products else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "total_predicted": len(predicted_titles),
        "total_relevant": len(relevant_products)
    }

def run_evaluation():
    """Run final evaluation"""

    print("\n" + "="*70)
    print("CHEESE RAG - FINAL EVALUATION")
    print("="*70)

    # Load GPT-categorized products
    categorized_file = 'cheese_data/categorized_gpt_indexed.json'

    try:
        with open(categorized_file, 'r') as f:
            all_products = json.load(f)
        print(f"\nLoaded {len(all_products)} GPT-categorized products")
    except FileNotFoundError:
        print(f"\nERROR: {categorized_file} not found!")
        print("Run: python cheese_rag_gpt.py first!")
        return

    # Initialize RAG (uses already-indexed data in ChromaDB)
    print("Initializing RAG system...")
    rag = CheeseRAGWithGPT()

    print("\nRunning evaluation queries...\n")

    results = []

    for i, test_case in enumerate(EVAL_QUERIES, 1):
        query = test_case["query"]
        filters = test_case["filters"]

        print(f"[{i}/{len(EVAL_QUERIES)}] {query}")

        # Search
        search_results = rag.smart_search(query, n_results=5)

        # Get predicted titles
        predicted = []
        for r in search_results['text_results'] + search_results['image_results']:
            title = r['metadata']['title']
            if title not in predicted:
                predicted.append(title)
        predicted = predicted[:5]

        # Calculate metrics
        metrics = calculate_metrics(predicted, filters, all_products)

        if metrics:
            print(f"  F1: {metrics['f1']:.1%} | P: {metrics['precision']:.1%} | R: {metrics['recall']:.1%}")
            print(f"  ({metrics['correct']}/{metrics['total_predicted']} correct, {metrics['total_relevant']} relevant in dataset)")
            results.append(metrics)

        print()

    # Calculate overall performance
    if results:
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)

        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nBaseline (keyword categorization): 17.1% F1")
        print(f"GPT Categorization:                 {avg_f1:.1%} F1")

        improvement = avg_f1 - 0.171
        print(f"\nImprovement: {improvement:+.1%} points")

        print(f"\nDetailed Metrics:")
        print(f"  Precision: {avg_precision:.1%}")
        print(f"  Recall:    {avg_recall:.1%}")
        print(f"  F1 Score:  {avg_f1:.1%}")

        # Save results
        output = {
            "baseline_f1": 0.171,
            "gpt_f1": avg_f1,
            "improvement": improvement,
            "precision": avg_precision,
            "recall": avg_recall,
            "num_queries": len(results)
        }

        with open("final_results.json", "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: final_results.json")
        print("="*70)

        # Conclusion
        if avg_f1 >= 0.60:
            print("\n✓ SUCCESS! GPT categorization significantly improved performance.")
            print("  This proves hybrid search + better categorization works.")
        elif avg_f1 >= 0.40:
            print("\n✓ IMPROVEMENT! GPT categorization helped, but more work needed.")
            print("  Consider: manual labeling, better test queries, or reranking.")
        else:
            print("\n✗ Limited improvement. Debug categorization or test queries.")

    print("\n" + "="*70)

if __name__ == "__main__":
    run_evaluation()