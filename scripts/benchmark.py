"""
Cognitive Router benchmarking script
Run with: poetry run cognitive-benchmark
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_router import create_similarity_router, create_keyword_router
from cognitive_router.utils import RouterBenchmark

def main():
    parser = argparse.ArgumentParser(description="Benchmark Cognitive Router performance")
    parser.add_argument("--router", choices=["similarity", "keyword", "both"], 
                       default="both", help="Router type to benchmark")
    parser.add_argument("--queries", type=int, default=100, 
                       help="Number of queries to benchmark")
    parser.add_argument("--runs", type=int, default=50, 
                       help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    print("🧠 Cognitive Router Benchmark")
    print("=" * 30)
    
    # Sample queries for benchmarking
    test_queries = [
        "I need help with my account",
        "How do I use the API?",
        "Billing question",
        "Product features"
    ] * (args.queries // 4)
    
    benchmark = RouterBenchmark()
    
    if args.router in ["similarity", "both"]:
        try:
            print("\n🧠 Benchmarking Similarity Router...")
            similarity_router = create_similarity_router()
            results = benchmark.benchmark_speed(similarity_router, test_queries, args.runs)
            print(f"   Speed: {results['mean_per_query_ms']:.2f}ms per query")
            print(f"   Throughput: {results['queries_per_second']:.0f} queries/sec")
        except Exception as e:
            print(f"   ❌ Similarity benchmark failed: {e}")
    
    if args.router in ["keyword", "both"]:
        try:
            print("\n⚡ Benchmarking Keyword Router...")
            keyword_router = create_keyword_router()
            results = benchmark.benchmark_speed(keyword_router, test_queries, args.runs)
            print(f"   Speed: {results['mean_per_query_ms']:.2f}ms per query")
            print(f"   Throughput: {results['queries_per_second']:.0f} queries/sec")
        except Exception as e:
            print(f"   ❌ Keyword benchmark failed: {e}")

if __name__ == "__main__":
    main()
