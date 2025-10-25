"""
Intelligent Keyword Matching System
Uses cosine similarity to find top 10 most relevant terms for any query
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vocabulary import ALL_TERMS, VOCABULARY
import re
from typing import List, Tuple, Dict

class IntelligentKeywordMatcher:
    def __init__(self):
        """Initialize the keyword matcher with TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            max_features=5000,
            min_df=1,
            max_df=0.95
        )
        
        # Prepare vocabulary for vectorization
        self.vocab_terms = ALL_TERMS
        self.vocab_matrix = None
        self._build_vocabulary_matrix()
    
    def _build_vocabulary_matrix(self):
        """Build TF-IDF matrix for all vocabulary terms."""
        print(f"Building vocabulary matrix for {len(self.vocab_terms)} terms...")
        
        # Create documents for each term (include the term itself and related terms)
        vocab_docs = []
        for term in self.vocab_terms:
            # Find related terms from the same category
            related_terms = []
            for category, terms in VOCABULARY.items():
                if term in terms:
                    related_terms.extend(terms[:10])  # Add first 10 related terms
            
            # Create a document with the term and its related terms
            doc = f"{term} {' '.join(related_terms[:5])}"  # Limit to avoid too long docs
            vocab_docs.append(doc)
        
        # Fit vectorizer and transform vocabulary
        self.vocab_matrix = self.vectorizer.fit_transform(vocab_docs)
        print(f"Vocabulary matrix built: {self.vocab_matrix.shape}")
    
    def find_similar_terms(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find top-k most similar terms to the given query using cosine similarity.
        
        Args:
            query: Input query string
            top_k: Number of top similar terms to return
            
        Returns:
            List of tuples (term, similarity_score) sorted by similarity
        """
        if not query or not query.strip():
            return []
        
        # Clean and preprocess query
        query = self._preprocess_query(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.vocab_matrix).flatten()
        
        # Get top-k similar terms
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include terms with positive similarity
                term = self.vocab_terms[idx]
                score = similarities[idx]
                results.append((term, score))
        
        return results
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching."""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove special characters but keep spaces and hyphens
        query = re.sub(r'[^\w\s-]', ' ', query)
        
        # Remove extra spaces
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    def get_expanded_query_terms(self, query: str, top_k: int = 10) -> List[str]:
        """
        Get expanded query terms for search.
        
        Args:
            query: Original query
            top_k: Number of terms to return
            
        Returns:
            List of expanded query terms
        """
        similar_terms = self.find_similar_terms(query, top_k)
        
        # Extract just the terms (without scores)
        expanded_terms = [term for term, score in similar_terms]
        
        # Always include the original query if it's not empty
        original_clean = self._preprocess_query(query)
        if original_clean and original_clean not in expanded_terms:
            expanded_terms.insert(0, original_clean)
        
        return expanded_terms[:top_k]
    
    def get_category_weights(self, query: str) -> Dict[str, float]:
        """
        Get category weights based on query similarity.
        
        Args:
            query: Input query
            
        Returns:
            Dictionary mapping category names to weights
        """
        similar_terms = self.find_similar_terms(query, 20)  # Get more terms for category analysis
        
        category_weights = {}
        for term, score in similar_terms:
            # Find which category this term belongs to
            for category, terms in VOCABULARY.items():
                if term in terms:
                    if category not in category_weights:
                        category_weights[category] = 0
                    category_weights[category] += score
        
        # Normalize weights
        total_weight = sum(category_weights.values())
        if total_weight > 0:
            category_weights = {k: v/total_weight for k, v in category_weights.items()}
        
        return category_weights

def test_keyword_matcher():
    """Test the keyword matcher with sample queries."""
    matcher = IntelligentKeywordMatcher()
    
    test_queries = [
        "reasoning",
        "math reasoning",
        "small model",
        "chain of thought",
        "language model",
        "fine-tuning",
        "quantization",
        "transformer",
        "bert",
        "gpt"
    ]
    
    print("\n" + "="*80)
    print("TESTING INTELLIGENT KEYWORD MATCHER")
    print("="*80)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        # Get expanded terms
        expanded_terms = matcher.get_expanded_query_terms(query, 10)
        print(f"📝 Expanded terms: {expanded_terms}")
        
        # Get category weights
        category_weights = matcher.get_category_weights(query)
        print(f"📊 Category weights: {category_weights}")
        
        # Get detailed similarities
        similar_terms = matcher.find_similar_terms(query, 5)
        print(f"🎯 Top 5 similar terms:")
        for term, score in similar_terms:
            print(f"   {term}: {score:.4f}")

if __name__ == "__main__":
    test_keyword_matcher()
