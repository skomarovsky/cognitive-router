"""
Basic usage example for Cognitive Router with Poetry
Demonstrates cognitive understanding vs traditional pattern matching
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from cognitive_router import create_similarity_router, create_keyword_router
except ImportError:
    print("❌ Please install cognitive_router first:")
    print("   poetry install")
    sys.exit(1)

def main():
    print("🧠 Cognitive Router - Basic Usage Example (Poetry)")
    print("=" * 55)
    
    # Sample queries demonstrating cognitive understanding
    test_queries = [
        "I can't access my account",
        "How do I call your API?",
        "I was charged incorrectly", 
        "What features are in the Pro plan?",
        "My login seems broken",  # Cognitive understanding test
        "Integration documentation needed",  # Cognitive context test
        "Billing looks wrong this month",  # Cognitive inference test
        "Show me what's included",  # Cognitive context understanding
        "Can you play Dark Side of the Moon"
    ]
    
    # Test cognitive keyword router (fast cognitive pattern recognition)
    print("\n⚡ Cognitive Keyword Router Results:")
    print("   (Fast cognitive pattern recognition)")
    try:
        cognitive_keyword = create_keyword_router()
        
        for query in test_queries:
            intent, confidence = cognitive_keyword.classify(query)
            print(f"  '{query}' → {intent} ({confidence:.3f})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test cognitive similarity router (deep cognitive understanding)
    print("\n🧠 Cognitive Similarity Router Results:")
    print("   (Deep semantic cognitive understanding)")
    try:
        cognitive_similarity = create_similarity_router()
        
        for query in test_queries:
            intent, confidence = cognitive_similarity.classify(query)
            print(f"  '{query}' → {intent} ({confidence:.3f})")
            
    except ImportError:
        print("   ⚠️ Cognitive similarity router requires sentence-transformers")
        print("   Install with: poetry install --extras 'all'")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Cognitive example completed!")
    print("\n🧠 Notice how cognitive understanding goes beyond simple keyword matching")
    print("   to grasp context, intent, and meaning like humans do.")
    print("\n🎭 Powered by Poetry dependency management")

if __name__ == "__main__":
    main()
