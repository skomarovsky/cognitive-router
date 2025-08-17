"""
Similarity-based router for deep semantic understanding
"""

from typing import Tuple, List, Dict, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SimilarityRouter:
    """Router using semantic similarity for intent classification"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SimilarityRouter. "
                "Install with: poetry install or pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name)
        self.intent_examples = self._get_default_intents()
        self.intent_embeddings = self._compute_intent_embeddings()
    
    def _get_default_intents(self) -> Dict[str, List[str]]:
        """Get default intent examples"""
        return {
            "account_access": [
                "I can't log in",
                "Password not working",
                "Account is locked",
                "Can't access my account",
                "Login failed"
            ],
            "technical_support": [
                "API integration help",
                "How to use the API",
                "Technical documentation",
                "Integration guide",
                "API tutorial"
            ],
            "billing": [
                "Incorrect charge",
                "Billing issue",
                "Payment problem",
                "Wrong amount charged",
                "Invoice question"
            ],
            "product_info": [
                "What features are available",
                "Product capabilities",
                "Plan comparison",
                "What's included",
                "Feature list"
            ]
        }
    
    def _compute_intent_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute embeddings for intent examples"""
        embeddings = {}
        for intent, examples in self.intent_examples.items():
            # Compute embeddings for all examples
            example_embeddings = self.model.encode(examples)
            # Use mean of examples as intent representation
            embeddings[intent] = np.mean(example_embeddings, axis=0)
        return embeddings
    
    def classify(self, query: str, threshold: float = 0.3) -> Tuple[str, float]:
        """
        Classify query into intent using semantic similarity
        
        Args:
            query: Input text to classify
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        # Encode the query
        query_embedding = self.model.encode([query])[0]
        
        # Compute similarities with all intents
        similarities = {}
        for intent, intent_embedding in self.intent_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, intent_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(intent_embedding)
            )
            similarities[intent] = similarity
        
        # Get best match
        best_intent = max(similarities, key=similarities.get)
        best_score = similarities[best_intent]
        
        if best_score < threshold:
            return "unknown", best_score
        
        return best_intent, best_score


def create_similarity_router(model_name: Optional[str] = None) -> SimilarityRouter:
    """
    Create a similarity-based router for deep semantic understanding
    
    Args:
        model_name: Optional sentence transformer model name
        
    Returns:
        Configured SimilarityRouter instance
    """
    if model_name:
        return SimilarityRouter(model_name=model_name)
    return SimilarityRouter()