from gensim.models import Word2Vec
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import numpy as np
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class ContentBasedRecommender:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.word2vec_model = None
        self.lda_model = None
        self.dictionary = None

    def preprocess_text(self, text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        return tokens

    def train_word2vec(self, articles: List[dict]):
        processed_articles = [self.preprocess_text(article['content']) 
                            for article in articles]
        self.word2vec_model = Word2Vec(sentences=processed_articles, 
                                     vector_size=100, 
                                     window=5, 
                                     min_count=1)

    def train_lda(self, articles: List[dict], num_topics=10):
        processed_articles = [self.preprocess_text(article['content']) 
                            for article in articles]
        self.dictionary = Dictionary(processed_articles)
        corpus = [self.dictionary.doc2bow(text) for text in processed_articles]
        self.lda_model = LdaModel(corpus=corpus,
                                 id2word=self.dictionary,
                                 num_topics=num_topics)

    def get_article_vector(self, article: dict) -> np.ndarray:
        tokens = self.preprocess_text(article['content'])
        word_vectors = [self.word2vec_model.wv[word] 
                       for word in tokens 
                       if word in self.word2vec_model.wv]
        return np.mean(word_vectors, axis=0)

    def recommend(self, user_profile: dict, articles: List[dict], n=5) -> List[str]:
        user_preferences = np.array(list(user_profile['preferences'].values()))
        article_vectors = [self.get_article_vector(article) for article in articles]
        
        # Calculate similarity scores
        similarities = [np.dot(user_preferences, vec) / 
                       (np.linalg.norm(user_preferences) * np.linalg.norm(vec))
                       for vec in article_vectors]
        
        # Get top N recommendations
        top_indices = np.argsort(similarities)[-n:][::-1]
        return [articles[i]['article_id'] for i in top_indices]