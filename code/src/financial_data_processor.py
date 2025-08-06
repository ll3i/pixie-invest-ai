#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Financial Data Processor with RAG capabilities.
- Financial data collection and preprocessing
- Vector database management using FAISS
- News and alert context generation
- RAG-based context retrieval for AI agents
- docs/data_processing.ipynb 기반 통합
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

# Disable all progress bars in sentence-transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
except ImportError:
    SentenceTransformer = None

from config import config
import pandas as pd

# Configure logging to suppress progress bars
import transformers
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

class FinancialDataProcessor:
    """Handles financial data processing and RAG context generation."""
    
    def __init__(self):
        # REQUIRED: Initialize embedding model
        self.embedding_model = self._load_embedding_model()
        
        # REQUIRED: Data caches - Initialize BEFORE vector_db
        self.news_cache: Dict[str, Any] = {}
        self.alerts_cache: Dict[str, Any] = {}
        self.market_data_cache: Dict[str, Any] = {}
        self.stock_evaluation_cache: Dict[str, Any] = {}
        
        # REQUIRED: Cache expiry times
        self.cache_expiry_hours = 1
        
        # REQUIRED: Initialize vector database
        self.vector_db = None
        self.document_store: List[Dict[str, Any]] = []
        self._initialize_vector_db()
        
        # Initialize stock search engine
        self.stock_search_engine = None
        self._initialize_stock_search()
        
        logger.info("FinancialDataProcessor initialized successfully")
    
    def _load_embedding_model(self) -> Optional[SentenceTransformer]:
        """
        Load sentence transformer model for embeddings.
        
        Returns:
            SentenceTransformer model or None if not available
        """
        if not SentenceTransformer:
            logger.warning("sentence-transformers not available, using mock embeddings")
            return None
        
        try:
            # REQUIRED: Use Korean-optimized model
            # Disable progress bars to avoid console spam
            import os
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models')
            
            # Disable tqdm globally
            try:
                from tqdm import tqdm
                tqdm.pandas = lambda: None
                from functools import partialmethod
                tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
            except:
                pass
            
            model = SentenceTransformer('jhgan/ko-sbert-multitask', device='cpu')
            # Disable progress bars
            model.max_seq_length = 512
            model.eval()  # Set to evaluation mode
            
            logger.info("Financial embedding model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None
    
    def _initialize_vector_db(self) -> None:
        """Initialize FAISS vector database."""
        try:
            if not faiss:
                logger.warning("FAISS not available, using simple search")
                return
            
            # REQUIRED: Initialize with 768-dimensional embeddings
            dimension = 768
            self.vector_db = faiss.IndexFlatL2(dimension)
            
            # REQUIRED: Load existing data if available
            self._load_existing_data()
            
            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Vector database initialization failed: {e}")
            self.vector_db = None
    
    def _initialize_stock_search(self) -> None:
        """Initialize stock search engine with notebook-based TF-IDF."""
        try:
            from stock_search_engine import StockSearchEngine
            self.stock_search_engine = StockSearchEngine()
            # Load search index
            self.stock_search_engine.load_search_index()
            logger.info("Stock search engine initialized successfully")
        except Exception as e:
            logger.error(f"Stock search engine initialization failed: {e}")
            self.stock_search_engine = None
    
    def _load_existing_data(self) -> None:
        """Load existing financial data into vector database."""
        try:
            # Supabase에서 데이터 로드 시도
            from db_client import get_supabase_client
            supabase = get_supabase_client()
            
            if supabase:
                logger.info("Loading data from Supabase")
                self._load_supabase_data(supabase)
                return
            
            # SQLite 폴백
            import sqlite3
            from pathlib import Path
            
            # 데이터베이스 경로 설정
            db_path = Path("investment_data.db")
            if not db_path.exists():
                db_path = Path("src") / "investment_data.db"
                if not db_path.exists():
                    logger.warning("Investment database not found. Loading mock data.")
                    self._load_mock_data()
                    return
            
            conn = sqlite3.connect(str(db_path))
            
            # 최신 뉴스 로드
            news_query = '''
                SELECT title, description as content, pub_date as timestamp,
                       sentiment_score, keywords
                FROM news_articles
                ORDER BY pub_date DESC
                LIMIT 100
            '''
            
            news_df = pd.read_sql_query(news_query, conn)
            
            for _, row in news_df.iterrows():
                try:
                    # 키워드 파싱
                    keywords = json.loads(row['keywords']) if row['keywords'] else []
                    
                    doc = {
                        "title": row['title'],
                        "content": row['content'] or '',
                        "category": "news",
                        "timestamp": row['timestamp'],
                        "sentiment_score": float(row['sentiment_score']) if row['sentiment_score'] else 0.5,
                        "relevance_keywords": keywords
                    }
                    self.add_document(doc)
                except Exception as e:
                    logger.error(f"Error adding news document: {e}")
                    continue
            
            # 위험 알림 생성 (주가 급변동 감지)
            alerts_query = '''
                WITH price_changes AS (
                    SELECT 
                        ticker,
                        date,
                        close,
                        LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) as prev_close,
                        (close - LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date)) / 
                        LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) * 100 as change_pct
                    FROM korean_stock_prices
                    WHERE date >= date('now', '-7 days')
                )
                SELECT 
                    ticker,
                    MAX(ABS(change_pct)) as max_change,
                    AVG(ABS(change_pct)) as avg_volatility
                FROM price_changes
                WHERE change_pct IS NOT NULL
                GROUP BY ticker
                HAVING MAX(ABS(change_pct)) > 5
            '''
            
            try:
                alerts_df = pd.read_sql_query(alerts_query, conn)
                
                for _, row in alerts_df.iterrows():
                    severity = "high" if row['max_change'] > 10 else "medium"
                    
                    alert_doc = {
                        "title": f"{row['ticker']} 주가 급변동 알림",
                        "content": f"{row['ticker']} 종목이 최근 {row['max_change']:.1f}% 변동했습니다. 평균 변동성: {row['avg_volatility']:.1f}%",
                        "category": "alert",
                        "severity": severity,
                        "timestamp": datetime.now().isoformat(),
                        "relevance_keywords": [row['ticker'], "변동성", "주의"]
                    }
                    self.add_document(alert_doc)
                    
            except Exception as e:
                logger.error(f"Error generating alerts: {e}")
            
            conn.close()
            
            logger.info(f"Loaded {len(self.document_store)} documents from database")
            
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
            self._load_mock_data()
    
    def _load_supabase_data(self, supabase) -> None:
        """Load financial data from Supabase."""
        try:
            # 최신 한국 주식 가격 데이터 로드
            today = datetime.now()
            week_ago = today - timedelta(days=7)
            
            # 한국 주식 가격 데이터
            try:
                kor_prices = supabase.table("kor_stock_prices").select("*").gte(
                    "date", week_ago.strftime("%Y-%m-%d")
                ).execute()
                
                if kor_prices.data:
                    logger.info(f"Loaded {len(kor_prices.data)} Korean stock price records")
                    # 가격 데이터를 문서로 변환
                    for price in kor_prices.data[:50]:  # 최근 50개만
                        doc = {
                            "title": f"{price.get('name', price['ticker'])} 주가 정보",
                            "content": f"{price['ticker']} 종가: {price['close']:,.0f}원, 거래량: {price['volume']:,}",
                            "category": "stock_price",
                            "timestamp": price['date'],
                            "ticker": price['ticker'],
                            "close": price['close'],
                            "volume": price['volume'],
                            "relevance_keywords": [price['ticker'], price.get('name', ''), "주가"]
                        }
                        self.add_document(doc)
                        # 캐시에도 저장
                        self.market_data_cache[price['ticker']] = price
            except Exception as e:
                logger.error(f"Error loading Korean stock prices: {e}")
            
            # 한국 주식 밸류에이션 지표
            try:
                kor_valuations = supabase.table("kor_valuation_metrics").select("*").gte(
                    "date", week_ago.strftime("%Y-%m-%d")
                ).execute()
                
                if kor_valuations.data:
                    logger.info(f"Loaded {len(kor_valuations.data)} valuation records")
                    # 티커별로 최신 지표만 저장
                    ticker_valuations = {}
                    for val in kor_valuations.data:
                        ticker = val['ticker']
                        if ticker not in ticker_valuations:
                            ticker_valuations[ticker] = {}
                        ticker_valuations[ticker][val['metric']] = val['value']
                    
                    # 밸류에이션 캐시 업데이트
                    for ticker, metrics in ticker_valuations.items():
                        if ticker in self.market_data_cache:
                            self.market_data_cache[ticker]['valuations'] = metrics
            except Exception as e:
                logger.error(f"Error loading valuations: {e}")
            
            # 미국 주식 가격 데이터
            try:
                us_prices = supabase.table("us_stock_prices").select("*").gte(
                    "date", week_ago.strftime("%Y-%m-%d")
                ).execute()
                
                if us_prices.data:
                    logger.info(f"Loaded {len(us_prices.data)} US stock price records")
                    for price in us_prices.data[:20]:  # 최근 20개만
                        doc = {
                            "title": f"{price['ticker']} Stock Price",
                            "content": f"{price['ticker']} Close: ${price['close']:.2f}, Volume: {price['volume']:,}",
                            "category": "stock_price",
                            "timestamp": price['date'],
                            "ticker": price['ticker'],
                            "close": price['close'],
                            "volume": price['volume'],
                            "relevance_keywords": [price['ticker'], "US", "stock"]
                        }
                        self.add_document(doc)
            except Exception as e:
                logger.error(f"Error loading US stock prices: {e}")
            
            # 한국 주식 평가 데이터
            try:
                kor_evaluations = supabase.table("kor_stock_evaluations").select("*").execute()
                
                if kor_evaluations.data:
                    logger.info(f"Loaded {len(kor_evaluations.data)} Korean stock evaluation records")
                    for eval_data in kor_evaluations.data:
                        # 평가 캐시에 저장
                        self.stock_evaluation_cache[eval_data['ticker']] = {
                            'name': eval_data['name'],
                            'score': eval_data['score'],
                            'evaluation': eval_data['evaluation'],
                            'per': eval_data['per'],
                            'pbr': eval_data['pbr'],
                            'reasons': eval_data['reasons']
                        }
                        
                        # 높은 평가 점수를 받은 종목은 문서로도 추가
                        if eval_data['score'] >= 70:
                            doc = {
                                "title": f"{eval_data['name']} 투자 추천",
                                "content": f"{eval_data['name']}({eval_data['ticker']}) - 평가점수: {eval_data['score']}점, {eval_data['evaluation']}. {eval_data['reasons']}",
                                "category": "stock_recommendation",
                                "timestamp": eval_data.get('created_at', datetime.now().isoformat()),
                                "ticker": eval_data['ticker'],
                                "score": eval_data['score'],
                                "relevance_keywords": [eval_data['ticker'], eval_data['name'], "추천", eval_data['evaluation']]
                            }
                            self.add_document(doc)
            except Exception as e:
                logger.error(f"Error loading Korean stock evaluations: {e}")
            
            # 뉴스 데이터 (새로운 테이블이 있다면)
            try:
                news = supabase.table("news").select("*").gte(
                    "created_at", week_ago.isoformat()
                ).limit(50).execute()
                
                if news.data:
                    logger.info(f"Loaded {len(news.data)} news articles")
                    for article in news.data:
                        doc = {
                            "title": article.get('title', ''),
                            "content": article.get('content', ''),
                            "category": "news",
                            "timestamp": article.get('created_at', ''),
                            "sentiment_score": article.get('sentiment_score', 0.5),
                            "relevance_keywords": article.get('keywords', [])
                        }
                        self.add_document(doc)
            except Exception as e:
                logger.info(f"News table not found or error: {e}")
            
            logger.info(f"Total documents loaded from Supabase: {len(self.document_store)}")
            
        except Exception as e:
            logger.error(f"Failed to load Supabase data: {e}")
            self._load_mock_data()
    
    def _load_mock_data(self) -> None:
        """Load mock data as fallback"""
        mock_news = [
            {
                "title": "한국 증시 상승세 지속",
                "content": "코스피가 3일 연속 상승하며 2400대를 회복했습니다. 기술주 중심의 매수세가 이어지고 있습니다.",
                "category": "news",
                "timestamp": datetime.now().isoformat(),
                "sentiment_score": 0.8,
                "relevance_keywords": ["코스피", "상승", "기술주"]
            },
            {
                "title": "미국 연준 금리 인하 시사",
                "content": "미국 연방준비제도가 다음 회의에서 금리 인하를 검토할 것이라고 발표했습니다.",
                "category": "news",
                "timestamp": datetime.now().isoformat(),
                "sentiment_score": 0.7,
                "relevance_keywords": ["연준", "금리", "인하"]
            }
        ]
        
        for doc in mock_news:
            self.add_document(doc)
        
        logger.info("Loaded mock data as fallback")
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """
        Add document to vector database.
        
        Args:
            document: Document with title, content, category, etc.
            
        Returns:
            True if added successfully
        """
        try:
            if not self.embedding_model:
                # REQUIRED: Store without embeddings if model unavailable
                self.document_store.append(document)
                return True
            
            # REQUIRED: Generate embedding
            text = f"{document.get('title', '')} {document.get('content', '')}"
            # Disable progress bar for encoding
            embedding = self.embedding_model.encode(text, show_progress_bar=False, batch_size=1)
            
            # REQUIRED: Add to vector database
            if self.vector_db is not None:
                self.vector_db.add(embedding.reshape(1, -1))
            
            # REQUIRED: Store document with metadata
            document_with_id = {
                **document,
                "doc_id": len(self.document_store),
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else None
            }
            
            self.document_store.append(document_with_id)
            
            logger.debug(f"Added document: {document.get('title', 'Untitled')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    def search_relevant_documents(
        self, 
        query: str, 
        top_k: int = 5, 
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            category_filter: Filter by document category
            
        Returns:
            List of relevant documents
        """
        try:
            if not self.document_store:
                return []
            
            # REQUIRED: Vector search if available
            if self.embedding_model and self.vector_db is not None:
                return self._vector_search(query, top_k, category_filter)
            else:
                return self._keyword_search(query, top_k, category_filter)
                
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
    
    def _vector_search(
        self, 
        query: str, 
        top_k: int, 
        category_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        try:
            # REQUIRED: Generate query embedding
            query_embedding = self.embedding_model.encode(query, show_progress_bar=False, batch_size=1)
            
            # REQUIRED: Search in vector database
            distances, indices = self.vector_db.search(query_embedding.reshape(1, -1), top_k * 2)
            
            # REQUIRED: Get documents and apply filters
            results = []
            for i, doc_idx in enumerate(indices[0]):
                if doc_idx < len(self.document_store):
                    doc = self.document_store[doc_idx]
                    
                    # REQUIRED: Apply category filter
                    if category_filter and doc.get("category") != category_filter:
                        continue
                    
                    # REQUIRED: Add similarity score
                    doc_with_score = {
                        **doc,
                        "similarity_score": float(1.0 / (1.0 + distances[0][i]))
                    }
                    
                    results.append(doc_with_score)
                    
                    if len(results) >= top_k:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _keyword_search(
        self, 
        query: str, 
        top_k: int, 
        category_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform simple keyword search as fallback."""
        try:
            query_terms = query.lower().split()
            scored_docs = []
            
            for doc in self.document_store:
                # REQUIRED: Apply category filter
                if category_filter and doc.get("category") != category_filter:
                    continue
                
                # REQUIRED: Calculate keyword score
                text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
                score = sum(1 for term in query_terms if term in text)
                
                if score > 0:
                    scored_docs.append({
                        **doc,
                        "keyword_score": score
                    })
            
            # REQUIRED: Sort by score and return top_k
            scored_docs.sort(key=lambda x: x["keyword_score"], reverse=True)
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def get_news_context(self, user_query: str) -> str:
        """
        Get news context relevant to user query.
        
        Args:
            user_query: User's query for context relevance
            
        Returns:
            Formatted news context string
        """
        try:
            # REQUIRED: Check cache first
            cache_key = f"news_{hash(user_query)}"
            cached_result = self._get_cached_result(cache_key, "news")
            if cached_result:
                return cached_result
            
            # REQUIRED: Search for relevant news
            news_docs = self.search_relevant_documents(
                user_query, 
                top_k=5, 
                category_filter="news"
            )
            
            if not news_docs:
                context = "[최신 뉴스]\n관련 뉴스가 없습니다."
            else:
                news_items = []
                sentiment_scores = []
                
                for doc in news_docs:
                    title = doc.get("title", "제목 없음")
                    content = doc.get("content", "")[:100] + "..."
                    sentiment_score = doc.get("sentiment_score", 0.5)
                    
                    # 감정 점수에 따른 레이블
                    if sentiment_score > 0.7:
                        sentiment_label = "긍정"
                    elif sentiment_score < 0.3:
                        sentiment_label = "부정"
                    else:
                        sentiment_label = "중립"
                    
                    news_items.append(f"- [{sentiment_label}] {title}: {content}")
                    sentiment_scores.append(sentiment_score)
                
                # 전체 감정 분석 요약
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
                positive_count = sum(1 for s in sentiment_scores if s > 0.7)
                negative_count = sum(1 for s in sentiment_scores if s < 0.3)
                
                context = "[최신 뉴스]\n" + "\n".join(news_items)
                context += f"\n\n[감정 분석 요약] 긍정: {positive_count}개, 부정: {negative_count}개, 평균 점수: {avg_sentiment:.2f}"
            
            # REQUIRED: Cache result
            self._cache_result(cache_key, "news", context)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get news context: {e}")
            return "[최신 뉴스]\n뉴스 정보를 가져올 수 없습니다."
    
    def get_risk_alerts_context(self, user_query: str) -> str:
        """
        Get risk alerts context relevant to user query.
        
        Args:
            user_query: User's query for context relevance
            
        Returns:
            Formatted alerts context string
        """
        try:
            # REQUIRED: Check cache first
            cache_key = f"alerts_{hash(user_query)}"
            cached_result = self._get_cached_result(cache_key, "alerts")
            if cached_result:
                return cached_result
            
            # REQUIRED: Search for relevant alerts
            alert_docs = self.search_relevant_documents(
                user_query,
                top_k=5,
                category_filter="alert"
            )
            
            if not alert_docs:
                context = "[위험 알림]\n현재 특별한 위험 요소가 없습니다."
            else:
                alert_items = []
                for doc in alert_docs:
                    title = doc.get("title", "알림 없음")
                    content = doc.get("content", "")[:100] + "..."
                    severity = doc.get("severity", "medium")
                    severity_icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                    alert_items.append(f"{severity_icon} {title}: {content}")
                
                context = "[위험 알림]\n" + "\n".join(alert_items)
            
            # REQUIRED: Cache result  
            self._cache_result(cache_key, "alerts", context)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get alerts context: {e}")
            return "[위험 알림]\n알림 정보를 가져올 수 없습니다."
    
    def get_stock_evaluation_context(self, user_query: str) -> str:
        """
        Get stock evaluation context using notebook-based search engine.
        
        Args:
            user_query: User's query for stock recommendations
            
        Returns:
            Formatted stock evaluation context string
        """
        try:
            if not self.stock_search_engine:
                return "[주식 평가]\n평가 데이터를 사용할 수 없습니다."
            
            # Check cache first
            cache_key = f"stock_eval_{hash(user_query)}"
            cached_result = self._get_cached_result(cache_key, "stock_evaluation")
            if cached_result:
                return cached_result
            
            # Search stocks using TF-IDF search engine
            search_results = self.stock_search_engine.search_stocks(user_query, n_results=5)
            
            if not search_results:
                context = "[주식 추천 결과]\n관련 종목을 찾을 수 없습니다."
            else:
                context_items = []
                context_items.append("[주식 추천 결과]")
                
                for i, result in enumerate(search_results, 1):
                    stock_info = []
                    stock_info.append(f"\n{i}. {result.종목명} ({result.종목코드})")
                    stock_info.append(f"   - 현재가: {result.현재가:,.0f}원")
                    stock_info.append(f"   - 시가총액: {result.시가총액/100000000:,.0f}억원")
                    stock_info.append(f"   - 평가점수: {result.평가점수}점")
                    stock_info.append(f"   - 종합평가: {result.종합평가}")
                    
                    # 재무 지표 추가
                    if result.PER is not None:
                        stock_info.append(f"   - PER: {result.PER:.1f}, PBR: {result.PBR:.1f}")
                    if result.매출성장률 is not None:
                        stock_info.append(f"   - 매출성장률: {result.매출성장률:.1f}%")
                    if result.순이익률 is not None:
                        stock_info.append(f"   - 순이익률: {result.순이익률:.1f}%")
                    if result.부채비율 is not None:
                        stock_info.append(f"   - 부채비율: {result.부채비율:.1f}%")
                    
                    # 평가이유 추가 (stock_evaluation_results.csv에서)
                    if hasattr(result, '평가이유'):
                        stock_info.append(f"   - 평가이유: {result.평가이유}")
                    else:
                        # Supabase에서 평가이유 가져오기
                        if result.종목코드 in self.stock_evaluation_cache:
                            reasons = self.stock_evaluation_cache[result.종목코드].get('reasons', '')
                            if reasons:
                                stock_info.append(f"   - 평가이유: {reasons}")
                    
                    stock_info.append(f"   - 기준일: 2025년 8월 1일")
                    
                    context_items.extend(stock_info)
                
                context = "\n".join(context_items)
            
            # Cache result
            self._cache_result(cache_key, "stock_evaluation", context)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get stock evaluation context: {e}")
            return "[주식 평가]\n평가 정보를 가져올 수 없습니다."
    
    def get_top_stocks_context(self, criteria: str = "평가점수", n_results: int = 10) -> str:
        """
        Get top stocks based on evaluation criteria.
        
        Args:
            criteria: Evaluation criteria (평가점수, 매출성장률, etc.)
            n_results: Number of top stocks to return
            
        Returns:
            Formatted top stocks context string
        """
        try:
            if not self.stock_search_engine:
                return "[상위 종목]\n평가 데이터를 사용할 수 없습니다."
            
            # Get top stocks
            top_stocks = self.stock_search_engine.get_top_stocks(n_results, criteria)
            
            if not top_stocks:
                context = f"[{criteria} 상위 종목]\n상위 종목을 찾을 수 없습니다."
            else:
                context_items = []
                context_items.append(f"[{criteria} 상위 {n_results}개 종목]")
                
                for i, stock in enumerate(top_stocks, 1):
                    context_items.append(f"{i}. {stock.종목명} ({stock.종목코드}): {getattr(stock, criteria) if hasattr(stock, criteria) else stock.평가점수}")
                
                context = "\n".join(context_items)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get top stocks context: {e}")
            return "[상위 종목]\n상위 종목 정보를 가져올 수 없습니다."
    
    def get_market_data_context(self, user_query: str) -> str:
        """
        Get market data context relevant to user query including news sentiment.
        
        Args:
            user_query: User's query for context relevance
            
        Returns:
            Formatted market data context string with sentiment analysis
        """
        try:
            # REQUIRED: Check cache first
            cache_key = f"market_{hash(user_query)}"
            cached_result = self._get_cached_result(cache_key, "market")
            if cached_result:
                return cached_result
            
            # 뉴스 감정 분석 데이터 가져오기
            news_sentiment_context = ""
            try:
                # 최신 뉴스 데이터 로드
                import os
                from datetime import datetime
                import pandas as pd
                from pathlib import Path
                
                # 데이터 디렉토리 경로
                data_dir = Path(__file__).parent.parent / "data" / "raw"
                today = datetime.now().strftime('%Y%m%d')
                news_file = data_dir / f"news_{today}.csv"
                
                # 오늘 파일이 없으면 가장 최근 파일 찾기
                if not news_file.exists():
                    news_files = list(data_dir.glob("news_*.csv"))
                    if news_files:
                        news_file = max(news_files, key=lambda x: x.stat().st_mtime)
                
                if news_file.exists():
                    # 뉴스 데이터 로드
                    news_df = pd.read_csv(news_file, encoding='utf-8')
                    
                    # 감정 분석 수행
                    import sys
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    from news_sentiment_analyzer import NewsSentimentAnalyzer
                    analyzer = NewsSentimentAnalyzer()
                    sentiment_result = analyzer.analyze_news_sentiment(news_df)
                    
                    # 감정 분석 컨텍스트 생성
                    market_mood = sentiment_result['market_mood']
                    sentiment_dist = sentiment_result['sentiment_distribution']
                    
                    news_sentiment_context = f"""
[뉴스 기반 시장 분위기]
- 전체 감정 점수: {sentiment_result['overall_sentiment']:.2f} ({market_mood['mood']})
- 감정 분포: 긍정 {sentiment_dist['positive']}건, 중립 {sentiment_dist['neutral']}건, 부정 {sentiment_dist['negative']}건
- 시장 설명: {market_mood['description']}
- 투자 권고: {market_mood['recommendation']}
"""
                    
                    # 주요 키워드 추가
                    if sentiment_result['top_positive_keywords']:
                        pos_keywords = [f"{kw[0]}({kw[1]})" for kw in sentiment_result['top_positive_keywords'][:5]]
                        news_sentiment_context += f"- 긍정 키워드: {', '.join(pos_keywords)}\n"
                    
                    if sentiment_result['top_negative_keywords']:
                        neg_keywords = [f"{kw[0]}({kw[1]})" for kw in sentiment_result['top_negative_keywords'][:5]]
                        news_sentiment_context += f"- 부정 키워드: {', '.join(neg_keywords)}\n"
                    
                    # 종목별 감정 분석
                    if sentiment_result['stock_sentiments']:
                        stock_sentiments = []
                        for stock, data in sentiment_result['stock_sentiments'].items():
                            if data['count'] > 0:
                                sentiment_icon = "📈" if data['trend'] == 'positive' else "📉" if data['trend'] == 'negative' else "➡️"
                                stock_sentiments.append(f"{stock}{sentiment_icon}({data['sentiment']:.2f})")
                        
                        if stock_sentiments:
                            news_sentiment_context += f"- 종목별 감정: {', '.join(stock_sentiments[:5])}\n"
                            
            except Exception as e:
                logger.warning(f"Failed to get news sentiment: {e}")
                news_sentiment_context = ""
            
            # PLACEHOLDER: Mock market data
            # In real implementation, this would fetch from financial APIs
            market_data = {
                "KOSPI": {"value": 2400.50, "change": "+1.2%"},
                "KOSDAQ": {"value": 850.30, "change": "+0.8%"},
                "USD/KRW": {"value": 1345.20, "change": "+0.3%"},
                "VIX": {"value": 18.5, "change": "-2.1%"}
            }
            
            context_items = []
            for symbol, data in market_data.items():
                context_items.append(f"- {symbol}: {data['value']} ({data['change']})")
            
            context = "[시장 데이터]\n" + "\n".join(context_items)
            
            # 뉴스 감정 분석 컨텍스트 추가
            if news_sentiment_context:
                context += "\n\n" + news_sentiment_context
            
            # REQUIRED: Cache result
            self._cache_result(cache_key, "market", context)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get market data context: {e}")
            return "[시장 데이터]\n시장 데이터를 가져올 수 없습니다."
    
    def _get_cached_result(self, cache_key: str, cache_type: str) -> Optional[str]:
        """Get cached result if still valid."""
        try:
            cache_dict = getattr(self, f"{cache_type}_cache", {})
            
            if cache_key in cache_dict:
                cached_item = cache_dict[cache_key]
                cache_time = datetime.fromisoformat(cached_item["timestamp"])
                
                # REQUIRED: Check if cache is still valid
                if datetime.now() - cache_time < timedelta(hours=self.cache_expiry_hours):
                    return cached_item["data"]
                else:
                    # REQUIRED: Remove expired cache
                    del cache_dict[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None
    
    def _cache_result(self, cache_key: str, cache_type: str, data: str) -> None:
        """Cache result with timestamp."""
        try:
            cache_dict = getattr(self, f"{cache_type}_cache", {})
            
            cache_dict[cache_key] = {
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # REQUIRED: Limit cache size
            if len(cache_dict) > 100:
                # Remove oldest entries
                sorted_items = sorted(
                    cache_dict.items(),
                    key=lambda x: x[1]["timestamp"]
                )
                
                for key, _ in sorted_items[:50]:
                    del cache_dict[key]
                    
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
    
    def update_financial_data(self) -> bool:
        """
        Update financial data from external sources.
        
        Returns:
            True if update successful
        """
        try:
            # PLACEHOLDER: In real implementation, this would:
            # 1. Fetch data from financial APIs (Yahoo Finance, Alpha Vantage, etc.)
            # 2. Process and clean the data
            # 3. Generate embeddings for new documents
            # 4. Update vector database
            
            logger.info("Financial data update completed (mock)")
            return True
            
        except Exception as e:
            logger.error(f"Financial data update failed: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get vector database statistics.
        
        Returns:
            Database statistics dictionary
        """
        try:
            stats = {
                "total_documents": len(self.document_store),
                "news_documents": len([d for d in self.document_store if d.get("category") == "news"]),
                "alert_documents": len([d for d in self.document_store if d.get("category") == "alert"]),
                "vector_db_available": self.vector_db is not None,
                "embedding_model_available": self.embedding_model is not None,
                "cache_sizes": {
                    "news_cache": len(self.news_cache),
                    "alerts_cache": len(self.alerts_cache),
                    "market_cache": len(self.market_data_cache),
                    "stock_evaluation_cache": len(self.stock_evaluation_cache)
                },
                "stock_search_engine_available": self.stock_search_engine is not None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """
        Clear cache data.
        
        Args:
            cache_type: Specific cache to clear, or None for all
            
        Returns:
            True if cleared successfully
        """
        try:
            if cache_type is None or cache_type == "news":
                self.news_cache.clear()
            
            if cache_type is None or cache_type == "alerts":
                self.alerts_cache.clear()
            
            if cache_type is None or cache_type == "market":
                self.market_data_cache.clear()
            
            if cache_type is None or cache_type == "stock_evaluation":
                self.stock_evaluation_cache.clear()
            
            logger.info(f"Cache cleared: {cache_type or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return False
    
    def export_database(self, file_path: str) -> bool:
        """
        Export vector database to file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if export successful
        """
        try:
            export_data = {
                "documents": self.document_store,
                "export_timestamp": datetime.now().isoformat(),
                "stats": self.get_database_stats()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Database exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database export failed: {e}")
            return False
    
    def import_database(self, file_path: str) -> bool:
        """
        Import vector database from file.
        
        Args:
            file_path: Path to import file
            
        Returns:
            True if import successful
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            documents = import_data.get("documents", [])
            
            # REQUIRED: Clear existing data
            self.document_store.clear()
            if self.vector_db is not None:
                self.vector_db.reset()
            
            # REQUIRED: Import documents
            for doc in documents:
                # Remove old embedding and doc_id to regenerate
                if "embedding" in doc:
                    del doc["embedding"]
                if "doc_id" in doc:
                    del doc["doc_id"]
                
                self.add_document(doc)
            
            logger.info(f"Database imported from {file_path}: {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Database import failed: {e}")
            return False

    def count_unread_alerts(self, user_id: str) -> int:
        """
        Count unread alerts for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of unread alerts
        """
        try:
            # PLACEHOLDER: In real implementation, this would query the database
            # For now, return a mock count
            return 0
        except Exception as e:
            logger.error(f"Failed to count unread alerts: {e}")
            return 0

    def mark_all_alerts_read(self, user_id: str) -> bool:
        """
        Mark all alerts as read for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful
        """
        try:
            # PLACEHOLDER: In real implementation, this would update the database
            logger.info(f"Marked all alerts as read for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark alerts as read: {e}")
            return False

    def mark_alert_read(self, user_id: str, alert_id: int) -> bool:
        """
        Mark a specific alert as read.
        
        Args:
            user_id: User identifier
            alert_id: Alert identifier
            
        Returns:
            True if successful
        """
        try:
            # PLACEHOLDER: In real implementation, this would update the database
            logger.info(f"Marked alert {alert_id} as read for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark alert as read: {e}")
            return False

    def fetch_stock_price_from_api(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Fetch stock price from external API.
        
        Args:
            code: Stock code
            
        Returns:
            Stock price data or None
        """
        try:
            # PLACEHOLDER: In real implementation, this would call a real API
            # For now, return mock data
            mock_price = {
                "code": code,
                "price": 50000 + (hash(code) % 10000),
                "change": 500 + (hash(code) % 1000),
                "change_percent": 1.5 + (hash(code) % 5),
                "volume": 1000000 + (hash(code) % 500000),
                "timestamp": datetime.now().isoformat()
            }
            return mock_price
        except Exception as e:
            logger.error(f"Failed to fetch stock price: {e}")
            return None

    def load_latest_evaluation_data(self) -> bool:
        """
        Load latest stock evaluation data from notebook-based processing.
        
        Returns:
            True if data loaded successfully
        """
        try:
            import os
            from pathlib import Path
            import pandas as pd
            
            # Try to load processed evaluation results
            processed_dir = Path(__file__).parent.parent / "data" / "processed"
            eval_file = processed_dir / "stock_evaluation_results.csv"
            
            if eval_file.exists():
                # Load evaluation results
                eval_df = pd.read_csv(eval_file, dtype={'종목코드': str})
                eval_df['종목코드'] = eval_df['종목코드'].str.zfill(6)
                
                # Add to vector database
                for _, row in eval_df.iterrows():
                    doc = {
                        "title": f"{row['종목명']} ({row['종목코드']}) 평가 정보",
                        "content": f"{row['종목명']} 평가점수: {row['평가점수']}점 ({row['종합평가']}), 평가이유: {row['평가이유']}",
                        "category": "stock_evaluation",
                        "timestamp": datetime.now().isoformat(),
                        "stock_code": row['종목코드'],
                        "stock_name": row['종목명'],
                        "evaluation_score": row['평가점수'],
                        "evaluation_grade": row['종합평가'],
                        "relevance_keywords": [row['종목명'], row['종목코드'], "평가", row['종합평가']]
                    }
                    self.add_document(doc)
                
                logger.info(f"Loaded {len(eval_df)} stock evaluation records")
                return True
            else:
                logger.warning("Stock evaluation results file not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load evaluation data: {e}")
            return False
    
    def get_latest_stock_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        Get latest stock data for a specific stock code.
        
        Args:
            stock_code: Stock code (6 digits)
            
        Returns:
            Stock data dictionary or None
        """
        try:
            # Check cache first
            if stock_code in self.market_data_cache:
                return self.market_data_cache[stock_code]
            
            # Try to load from latest price data
            import os
            from pathlib import Path
            from datetime import datetime
            import pandas as pd
            
            data_dir = Path(__file__).parent.parent / "data" / "raw"
            today = datetime.now().strftime('%Y%m%d')
            
            # Find latest price file
            price_file = data_dir / f"kor_price_{today}.csv"
            if not price_file.exists():
                price_files = list(data_dir.glob("kor_price_*.csv"))
                if price_files:
                    price_file = max(price_files, key=lambda x: x.stat().st_mtime)
            
            if price_file.exists():
                # Load price data
                price_df = pd.read_csv(price_file, dtype={'종목코드': str})
                price_df['종목코드'] = price_df['종목코드'].str.zfill(6)
                
                # Find stock data
                stock_data = price_df[price_df['종목코드'] == stock_code]
                if not stock_data.empty:
                    stock_info = stock_data.iloc[0].to_dict()
                    
                    # Try to get valuation metrics
                    value_file = data_dir / f"kor_value_{today}.csv"
                    if not value_file.exists():
                        value_files = list(data_dir.glob("kor_value_*.csv"))
                        if value_files:
                            value_file = max(value_files, key=lambda x: x.stat().st_mtime)
                    
                    if value_file.exists():
                        value_df = pd.read_csv(value_file, dtype={'종목코드': str})
                        value_df['종목코드'] = value_df['종목코드'].str.zfill(6)
                        
                        # Get PER and PBR
                        per_data = value_df[(value_df['종목코드'] == stock_code) & (value_df['지표'] == 'PER')]
                        pbr_data = value_df[(value_df['종목코드'] == stock_code) & (value_df['지표'] == 'PBR')]
                        
                        if not per_data.empty:
                            stock_info['PER'] = float(per_data['값'].iloc[0])
                        if not pbr_data.empty:
                            stock_info['PBR'] = float(pbr_data['값'].iloc[0])
                    
                    # Cache and return
                    self.market_data_cache[stock_code] = stock_info
                    return stock_info
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest stock data: {e}")
            return None
    
    def get_today_news(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get today's news.
        
        Args:
            top_k: Number of news items to return
            
        Returns:
            List of news items
        """
        try:
            # PLACEHOLDER: In real implementation, this would fetch from news API
            mock_news = [
                {
                    "title": "한국 증시 상승세 지속",
                    "content": "코스피가 3일 연속 상승하며 2400대를 회복했습니다.",
                    "summary": "코스피 상승세 지속",
                    "sentiment": "positive",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "title": "미국 연준 금리 인하 시사",
                    "content": "미국 연방준비제도가 다음 회의에서 금리 인하를 검토할 것이라고 발표했습니다.",
                    "summary": "연준 금리 인하 시사",
                    "sentiment": "positive",
                    "timestamp": datetime.now().isoformat()
                }
            ]
            return mock_news[:top_k]
        except Exception as e:
            logger.error(f"Failed to get today's news: {e}")
            return []

    def filter_news_by_keywords(self, news_list: List[Dict[str, Any]], keywords: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Filter news by keywords using similarity.
        
        Args:
            news_list: List of news items
            keywords: List of keywords to filter by
            threshold: Similarity threshold
            
        Returns:
            Filtered news list
        """
        try:
            # PLACEHOLDER: In real implementation, this would use embeddings
            # For now, simple keyword matching
            filtered_news = []
            for news in news_list:
                content = f"{news.get('title', '')} {news.get('content', '')}".lower()
                for keyword in keywords:
                    if keyword.lower() in content:
                        filtered_news.append(news)
                        break
            return filtered_news
        except Exception as e:
            logger.error(f"Failed to filter news by keywords: {e}")
            return news_list 