"""
Keyword-based router for fast pattern recognition with cognitive understanding
"""

from typing import Tuple, List, Dict, Set
import re


class KeywordRouter:
    """Router using intelligent keyword matching with semantic patterns"""
    
    def __init__(self):
        self.intent_patterns = self._get_default_patterns()
    
    def _get_default_patterns(self) -> Dict[str, Dict[str, any]]:
        """Get default intent patterns with cognitive keyword groups"""
        return {
            "account_access": {
                "keywords": {
                    "primary": ["login", "access", "account", "password", "sign in"],
                    "secondary": ["can't", "unable", "locked", "forgot", "reset", "broken"],
                    "context": ["my", "help", "issue", "problem"]
                },
                "patterns": [
                    r"(can't|cannot|unable to).*(access|log|sign)",
                    r"(login|password|account).*(issue|problem|broken|fail)",
                    r"(locked|forgot|reset).*(account|password)"
                ],
                "weight": 1.0
            },
            "technical_support": {
                "keywords": {
                    "primary": ["api", "integration", "documentation", "technical", "code"],
                    "secondary": ["help", "guide", "tutorial", "example", "how to"],
                    "context": ["use", "implement", "call", "setup", "configure"]
                },
                "patterns": [
                    r"(api|integration).*(help|documentation|guide)",
                    r"how.*(to|do).*(api|integrate|implement)",
                    r"(technical|code).*(support|help|documentation)"
                ],
                "weight": 1.0
            },
            "billing": {
                "keywords": {
                    "primary": ["billing", "charge", "payment", "invoice", "price"],
                    "secondary": ["incorrect", "wrong", "issue", "problem", "question"],
                    "context": ["month", "charged", "amount", "cost", "fee"]
                },
                "patterns": [
                    r"(charge|bill|payment).*(incorrect|wrong|issue)",
                    r"(billing|invoice).*(problem|question|looks wrong)",
                    r"(was|been).*(charged|billed).*(incorrect|wrong)"
                ],
                "weight": 1.0
            },
            "product_info": {
                "keywords": {
                    "primary": ["feature", "plan", "product", "capability", "included"],
                    "secondary": ["what", "show", "list", "available", "offer"],
                    "context": ["pro", "premium", "basic", "enterprise", "pricing"]
                },
                "patterns": [
                    r"what.*(feature|included|available)",
                    r"(plan|product).*(include|offer|capability)",
                    r"show.*(feature|what|included)"
                ],
                "weight": 1.0
            }
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        return text.lower().strip()
    
    def _calculate_keyword_score(self, query: str, intent_config: Dict) -> float:
        """Calculate cognitive keyword matching score"""
        query_normalized = self._normalize_text(query)
        query_words = set(query_normalized.split())
        
        score = 0.0
        
        # Check primary keywords (higher weight)
        primary_matches = sum(1 for kw in intent_config["keywords"]["primary"] 
                             if kw in query_normalized)
        score += primary_matches * 0.5
        
        # Check secondary keywords (medium weight)
        secondary_matches = sum(1 for kw in intent_config["keywords"]["secondary"] 
                               if kw in query_normalized)
        score += secondary_matches * 0.3
        
        # Check context keywords (lower weight)
        context_matches = sum(1 for kw in intent_config["keywords"]["context"] 
                            if kw in query_normalized)
        score += context_matches * 0.2
        
        # Check regex patterns (bonus score)
        for pattern in intent_config["patterns"]:
            if re.search(pattern, query_normalized):
                score += 0.5
                break
        
        # Apply intent weight
        score *= intent_config["weight"]
        
        # Normalize score to 0-1 range
        max_possible_score = (
            len(intent_config["keywords"]["primary"]) * 0.5 +
            len(intent_config["keywords"]["secondary"]) * 0.3 +
            len(intent_config["keywords"]["context"]) * 0.2 +
            0.5  # pattern bonus
        )
        
        if max_possible_score > 0:
            score = min(score / max_possible_score, 1.0)
        
        return score
    
    def classify(self, query: str, threshold: float = 0.15) -> Tuple[str, float]:
        """
        Classify query using cognitive keyword matching
        
        Args:
            query: Input text to classify
            threshold: Minimum score threshold
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = self._calculate_keyword_score(query, config)
            scores[intent] = score
        
        # Get best match
        if not scores:
            return "unknown", 0.0
        
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        if best_score < threshold:
            return "unknown", best_score
        
        return best_intent, best_score


def create_keyword_router() -> KeywordRouter:
    """
    Create a keyword-based router for fast cognitive pattern recognition
    
    Returns:
        Configured KeywordRouter instance
    """
    return KeywordRouter()