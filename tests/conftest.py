"""
Pytest configuration for Cognitive Router tests
"""

import pytest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return [
        ("I can't log into my account", "support"),
        ("How do I use the API?", "technical"),
        ("I was charged twice", "billing"),
        ("What features are available?", "product")
    ]

@pytest.fixture
def cognitive_test_data():
    """Cognitive test data for advanced testing"""
    return {
        "support": [
            "I need help with my account",
            "Something is broken",
            "Can't access the system"
        ],
        "technical": [
            "How do I integrate the API?",
            "Documentation needed",
            "SDK examples"
        ],
        "billing": [
            "Payment issue",
            "Invoice question",
            "Subscription problem"
        ],
        "product": [
            "What features are included?",
            "Product capabilities",
            "Free trial info"
        ]
    }
