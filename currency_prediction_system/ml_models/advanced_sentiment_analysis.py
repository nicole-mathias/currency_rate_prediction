#!/usr/bin/env python3
"""
Advanced Sentiment Analysis with RAG Architecture
================================================

This module implements:
- Real-time sentiment analysis pipeline
- NLP transformers for advanced text processing
- RAG-based architecture with vector embeddings
- Topic relevance and trend identification
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Try to import advanced NLP libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    print("Advanced NLP libraries not available. Using basic sentiment analysis.")

# Vector database (if available)
try:
    import faiss
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    print("FAISS not available. Using simple vector storage.")

import logging
import json
from datetime import datetime, timedelta
from collections import defaultdict

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis with RAG architecture"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize basic NLP
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize advanced NLP if available
        if ADVANCED_NLP_AVAILABLE:
            self._setup_advanced_nlp()
        
        # Initialize vector storage
        self.vector_database = {}
        self.topic_embeddings = {}
        self.sentiment_history = defaultdict(list)
        
    def _setup_advanced_nlp(self):
        """Setup advanced NLP models"""
        try:
            # Sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Topic modeling (if available)
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import LatentDirichletAllocation
                self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
                self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
                self.topic_modeling_available = True
            except ImportError:
                self.topic_modeling_available = False
                
        except Exception as e:
            self.logger.warning(f"Advanced NLP setup failed: {e}")
            ADVANCED_NLP_AVAILABLE = False
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def analyze_sentiment_advanced(self, text):
        """Advanced sentiment analysis using multiple methods"""
        results = {}
        
        # Basic VADER sentiment
        vader_scores = self.sia.polarity_scores(text)
        results['vader'] = vader_scores
        
        # Advanced transformer-based sentiment
        if ADVANCED_NLP_AVAILABLE:
            try:
                transformer_results = self.sentiment_pipeline(text)
                results['transformer'] = {
                    'positive': transformer_results[0][2]['score'],
                    'negative': transformer_results[0][0]['score'],
                    'neutral': transformer_results[0][1]['score']
                }
            except Exception as e:
                self.logger.warning(f"Transformer sentiment failed: {e}")
        
        # Calculate composite sentiment score
        composite_score = self._calculate_composite_sentiment(results)
        results['composite_score'] = composite_score
        
        return results
    
    def _calculate_composite_sentiment(self, sentiment_results):
        """Calculate composite sentiment score"""
        vader_compound = sentiment_results['vader']['compound']
        
        if 'transformer' in sentiment_results:
            transformer_positive = sentiment_results['transformer']['positive']
            transformer_negative = sentiment_results['transformer']['negative']
            transformer_score = transformer_positive - transformer_negative
        else:
            transformer_score = vader_compound
        
        # Weighted average
        composite = (0.4 * vader_compound + 0.6 * transformer_score)
        return composite
    
    def extract_topics(self, text):
        """Extract topics from text"""
        if not self.topic_modeling_available:
            return []
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Create TF-IDF features
            tfidf_features = self.tfidf_vectorizer.fit_transform([processed_text])
            
            # Extract topics using LDA
            topic_distribution = self.lda_model.fit_transform(tfidf_features)
            
            # Get top topics
            top_topics = np.argsort(topic_distribution[0])[-3:]  # Top 3 topics
            
            return top_topics.tolist()
        except Exception as e:
            self.logger.warning(f"Topic extraction failed: {e}")
            return []
    
    def create_embeddings(self, text):
        """Create vector embeddings for text"""
        if not ADVANCED_NLP_AVAILABLE:
            return None
        
        try:
            embedding = self.sentence_transformer.encode(text)
            return embedding
        except Exception as e:
            self.logger.warning(f"Embedding creation failed: {e}")
            return None
    
    def store_in_vector_database(self, text, sentiment_results, currency_pair, timestamp):
        """Store text and sentiment in vector database"""
        embedding = self.create_embeddings(text)
        
        if embedding is not None:
            # Create document entry
            doc_entry = {
                'text': text,
                'sentiment': sentiment_results,
                'currency_pair': currency_pair,
                'timestamp': timestamp,
                'embedding': embedding,
                'topics': self.extract_topics(text)
            }
            
            # Store in vector database
            if currency_pair not in self.vector_database:
                self.vector_database[currency_pair] = []
            
            self.vector_database[currency_pair].append(doc_entry)
            
            # Update sentiment history
            self.sentiment_history[currency_pair].append({
                'timestamp': timestamp,
                'sentiment': sentiment_results['composite_score']
            })
    
    def search_similar_documents(self, query, currency_pair, top_k=5):
        """Search for similar documents using vector similarity"""
        if not ADVANCED_NLP_AVAILABLE or currency_pair not in self.vector_database:
            return []
        
        try:
            query_embedding = self.create_embeddings(query)
            
            if query_embedding is None:
                return []
            
            # Calculate similarities
            similarities = []
            for doc in self.vector_database[currency_pair]:
                similarity = np.dot(query_embedding, doc['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc['embedding'])
                )
                similarities.append((similarity, doc))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.warning(f"Document search failed: {e}")
            return []
    
    def identify_trends(self, currency_pair, time_window=24):
        """Identify sentiment trends over time"""
        if currency_pair not in self.sentiment_history:
            return {}
        
        # Get recent sentiment data
        cutoff_time = datetime.now() - timedelta(hours=time_window)
        recent_sentiments = [
            entry for entry in self.sentiment_history[currency_pair]
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_sentiments:
            return {}
        
        # Calculate trend metrics
        sentiments = [entry['sentiment'] for entry in recent_sentiments]
        
        trend_analysis = {
            'mean_sentiment': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'sentiment_trend': np.polyfit(range(len(sentiments)), sentiments, 1)[0],
            'sentiment_volatility': np.var(sentiments),
            'positive_ratio': sum(1 for s in sentiments if s > 0) / len(sentiments),
            'negative_ratio': sum(1 for s in sentiments if s < 0) / len(sentiments)
        }
        
        return trend_analysis
    
    def process_news_batch(self, news_data, currency_pair):
        """Process batch of news articles"""
        self.logger.info(f"Processing {len(news_data)} news articles for {currency_pair}")
        
        batch_results = []
        
        for _, row in news_data.iterrows():
            # Combine title and content
            text = f"{row.get('title', '')} {row.get('content', '')}"
            
            # Analyze sentiment
            sentiment_results = self.analyze_sentiment_advanced(text)
            
            # Extract topics
            topics = self.extract_topics(text)
            
            # Create embeddings and store
            timestamp = datetime.now()
            self.store_in_vector_database(text, sentiment_results, currency_pair, timestamp)
            
            # Create result entry
            result_entry = {
                'text': text,
                'sentiment': sentiment_results,
                'topics': topics,
                'timestamp': timestamp
            }
            
            batch_results.append(result_entry)
        
        # Identify trends
        trends = self.identify_trends(currency_pair)
        
        return pd.DataFrame(batch_results), trends
    
    def get_sentiment_summary(self, currency_pair):
        """Get sentiment summary for a currency pair"""
        if currency_pair not in self.sentiment_history:
            return {}
        
        recent_sentiments = self.sentiment_history[currency_pair][-100:]  # Last 100 entries
        
        if not recent_sentiments:
            return {}
        
        sentiments = [entry['sentiment'] for entry in recent_sentiments]
        
        summary = {
            'current_sentiment': sentiments[-1] if sentiments else 0,
            'average_sentiment': np.mean(sentiments),
            'sentiment_trend': np.polyfit(range(len(sentiments)), sentiments, 1)[0],
            'sentiment_volatility': np.std(sentiments),
            'positive_count': sum(1 for s in sentiments if s > 0),
            'negative_count': sum(1 for s in sentiments if s < 0),
            'neutral_count': sum(1 for s in sentiments if s == 0),
            'total_articles': len(sentiments)
        }
        
        return summary 