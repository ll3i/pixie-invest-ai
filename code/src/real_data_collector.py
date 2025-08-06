#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real Data Collector - 실제 금융 데이터 수집기
- 한국 주식: FinanceDataReader 사용
- 미국 주식: yfinance 사용
- 뉴스: Naver Search API 사용
- 데이터베이스 저장 기능 포함
"""

import os
import sys
import json
import time
import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_data_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 필수 라이브러리 import
try:
    import FinanceDataReader as fdr
    HAS_FDR = True
except ImportError:
    logger.warning("FinanceDataReader not installed. Korean stock data will be limited.")
    HAS_FDR = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    logger.warning("yfinance not installed. US stock data will be limited.")
    HAS_YFINANCE = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    logger.warning("BeautifulSoup not installed. Some features will be limited.")
    HAS_BS4 = False


class RealDataCollector:
    """실제 금융 데이터 수집기"""
    
    def __init__(self, db_path: str = "investment_data.db"):
        """
        초기화
        
        Args:
            db_path: SQLite 데이터베이스 경로
        """
        self.db_path = db_path
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # 디렉토리 생성
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 초기화
        self._init_database()
        
        # API 키 로드 (환경변수에서)
        self.naver_client_id = os.getenv('NAVER_CLIENT_ID', '')
        self.naver_client_secret = os.getenv('NAVER_CLIENT_SECRET', '')
        
        logger.info("RealDataCollector initialized")
    
    def _init_database(self):
        """데이터베이스 테이블 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 한국 주식 가격 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS korean_stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        ''')
        
        # 미국 주식 가격 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS us_stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        ''')
        
        # 재무제표 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_statements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                metric_name VARCHAR(100),
                value REAL,
                frequency VARCHAR(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date, metric_name, frequency)
            )
        ''')
        
        # 뉴스 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                link TEXT,
                pub_date TIMESTAMP,
                source VARCHAR(100),
                sentiment_score REAL,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(title, pub_date)
            )
        ''')
        
        # 종목 정보 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker VARCHAR(10) NOT NULL UNIQUE,
                name VARCHAR(100),
                market VARCHAR(20),
                sector VARCHAR(50),
                industry VARCHAR(50),
                market_cap REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database tables initialized")
    
    def collect_korean_stocks(self, days: int = 365) -> pd.DataFrame:
        """
        한국 주식 데이터 수집 (FinanceDataReader 사용)
        
        Args:
            days: 수집할 일수 (기본 1년)
            
        Returns:
            수집된 데이터프레임
        """
        if not HAS_FDR:
            logger.error("FinanceDataReader not available")
            return pd.DataFrame()
        
        try:
            logger.info(f"Collecting Korean stock data for last {days} days")
            
            # KOSPI/KOSDAQ 종목 리스트 가져오기
            stock_list = fdr.StockListing('KRX')
            
            # 시가총액 상위 200개 종목만 선택
            stock_list = stock_list.nlargest(200, 'Marcap')
            
            all_data = []
            conn = sqlite3.connect(self.db_path)
            
            for idx, row in stock_list.iterrows():
                ticker = row['Code']
                name = row['Name']
                
                try:
                    # 주가 데이터 수집
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    df = fdr.DataReader(ticker, start_date, end_date)
                    
                    if not df.empty:
                        # 데이터 정리
                        df.reset_index(inplace=True)
                        df['ticker'] = ticker
                        df['name'] = name
                        
                        # 데이터베이스에 저장
                        for _, price_row in df.iterrows():
                            try:
                                conn.execute('''
                                    INSERT OR REPLACE INTO korean_stock_prices 
                                    (ticker, date, open, high, low, close, volume)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    ticker,
                                    price_row['Date'].strftime('%Y-%m-%d'),
                                    float(price_row['Open']),
                                    float(price_row['High']),
                                    float(price_row['Low']),
                                    float(price_row['Close']),
                                    int(price_row['Volume'])
                                ))
                            except Exception as e:
                                logger.error(f"Error inserting {ticker} data: {e}")
                        
                        # 종목 정보 업데이트
                        conn.execute('''
                            INSERT OR REPLACE INTO stock_info 
                            (ticker, name, market, market_cap)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            ticker,
                            name,
                            row.get('Market', 'KRX'),
                            float(row.get('Marcap', 0))
                        ))
                        
                        all_data.append(df)
                        logger.info(f"Collected data for {ticker} - {name}")
                    
                    # API 제한 방지
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error collecting {ticker}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            if all_data:
                result_df = pd.concat(all_data, ignore_index=True)
                
                # CSV 파일로도 저장
                today = datetime.now().strftime('%Y%m%d')
                output_file = self.raw_dir / f"kor_price_{today}.csv"
                result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                
                logger.info(f"Collected {len(result_df)} records for {len(stock_list)} Korean stocks")
                return result_df
            else:
                logger.warning("No Korean stock data collected")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in collect_korean_stocks: {e}")
            return pd.DataFrame()
    
    def collect_us_stocks(self, days: int = 365) -> pd.DataFrame:
        """
        미국 주식 데이터 수집 (yfinance 사용)
        
        Args:
            days: 수집할 일수 (기본 1년)
            
        Returns:
            수집된 데이터프레임
        """
        if not HAS_YFINANCE:
            logger.error("yfinance not available")
            return pd.DataFrame()
        
        try:
            logger.info(f"Collecting US stock data for last {days} days")
            
            # 주요 미국 주식 리스트
            us_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
                'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
                'ADBE', 'NFLX', 'CRM', 'PFE', 'TMO', 'CSCO', 'PEP', 'ABT',
                'CVX', 'ABBV', 'NKE', 'WMT', 'MRK'
            ]
            
            all_data = []
            conn = sqlite3.connect(self.db_path)
            
            for ticker in us_tickers:
                try:
                    # yfinance로 데이터 수집
                    stock = yf.Ticker(ticker)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    df = stock.history(start=start_date, end=end_date)
                    
                    if not df.empty:
                        # 데이터 정리
                        df.reset_index(inplace=True)
                        df['ticker'] = ticker
                        
                        # 주식 정보 가져오기
                        info = stock.info
                        company_name = info.get('longName', ticker)
                        
                        # 데이터베이스에 저장
                        for _, row in df.iterrows():
                            try:
                                conn.execute('''
                                    INSERT OR REPLACE INTO us_stock_prices 
                                    (ticker, date, open, high, low, close, volume)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    ticker,
                                    row['Date'].strftime('%Y-%m-%d'),
                                    float(row['Open']),
                                    float(row['High']),
                                    float(row['Low']),
                                    float(row['Close']),
                                    int(row['Volume'])
                                ))
                            except Exception as e:
                                logger.error(f"Error inserting {ticker} data: {e}")
                        
                        # 종목 정보 업데이트
                        conn.execute('''
                            INSERT OR REPLACE INTO stock_info 
                            (ticker, name, market, sector, industry, market_cap)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            ticker,
                            company_name,
                            info.get('exchange', 'NYSE'),
                            info.get('sector', ''),
                            info.get('industry', ''),
                            float(info.get('marketCap', 0))
                        ))
                        
                        all_data.append(df)
                        logger.info(f"Collected data for {ticker} - {company_name}")
                    
                    # API 제한 방지
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error collecting {ticker}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            if all_data:
                result_df = pd.concat(all_data, ignore_index=True)
                
                # CSV 파일로도 저장
                today = datetime.now().strftime('%Y%m%d')
                output_file = self.raw_dir / f"us_price_{today}.csv"
                result_df.to_csv(output_file, index=False, encoding='utf-8')
                
                logger.info(f"Collected {len(result_df)} records for {len(us_tickers)} US stocks")
                return result_df
            else:
                logger.warning("No US stock data collected")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in collect_us_stocks: {e}")
            return pd.DataFrame()
    
    def collect_news_naver(self, query: str = "주식 투자", max_items: int = 100) -> List[Dict]:
        """
        네이버 뉴스 검색 API를 사용한 뉴스 수집
        
        Args:
            query: 검색어
            max_items: 최대 수집 개수
            
        Returns:
            뉴스 리스트
        """
        if not self.naver_client_id or not self.naver_client_secret:
            logger.warning("Naver API credentials not set. Using alternative method.")
            return self._collect_news_rss()
        
        try:
            logger.info(f"Collecting news from Naver API with query: {query}")
            
            url = "https://openapi.naver.com/v1/search/news.json"
            headers = {
                "X-Naver-Client-Id": self.naver_client_id,
                "X-Naver-Client-Secret": self.naver_client_secret
            }
            
            all_news = []
            conn = sqlite3.connect(self.db_path)
            
            # 페이지네이션으로 뉴스 수집
            for start in range(1, min(max_items, 1000), 100):
                params = {
                    "query": query,
                    "display": min(100, max_items - len(all_news)),
                    "start": start,
                    "sort": "date"
                }
                
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('items', [])
                    
                    for item in items:
                        try:
                            # HTML 태그 제거
                            title = BeautifulSoup(item['title'], 'html.parser').get_text() if HAS_BS4 else item['title']
                            description = BeautifulSoup(item['description'], 'html.parser').get_text() if HAS_BS4 else item['description']
                            
                            # 날짜 파싱
                            pub_date = datetime.strptime(item['pubDate'], '%a, %d %b %Y %H:%M:%S %z').strftime('%Y-%m-%d %H:%M:%S')
                            
                            # 간단한 감정 분석 (향후 개선 필요)
                            sentiment_score = self._analyze_sentiment(title + " " + description)
                            
                            # 키워드 추출
                            keywords = self._extract_keywords(title + " " + description)
                            
                            # 데이터베이스에 저장
                            try:
                                conn.execute('''
                                    INSERT OR IGNORE INTO news_articles 
                                    (title, description, link, pub_date, source, sentiment_score, keywords)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    title,
                                    description,
                                    item['link'],
                                    pub_date,
                                    'Naver News',
                                    sentiment_score,
                                    json.dumps(keywords, ensure_ascii=False)
                                ))
                                
                                all_news.append({
                                    'title': title,
                                    'description': description,
                                    'link': item['link'],
                                    'pub_date': pub_date,
                                    'source': 'Naver News',
                                    'sentiment_score': sentiment_score,
                                    'keywords': keywords
                                })
                                
                            except Exception as e:
                                logger.error(f"Error inserting news: {e}")
                                
                        except Exception as e:
                            logger.error(f"Error processing news item: {e}")
                            continue
                    
                    if len(items) < params['display']:
                        break
                        
                else:
                    logger.error(f"Naver API error: {response.status_code}")
                    break
                
                time.sleep(0.1)  # API 제한 방지
            
            conn.commit()
            conn.close()
            
            logger.info(f"Collected {len(all_news)} news articles")
            
            # CSV로도 저장
            if all_news:
                df = pd.DataFrame(all_news)
                today = datetime.now().strftime('%Y%m%d')
                output_file = self.raw_dir / f"news_{today}.csv"
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            return all_news
            
        except Exception as e:
            logger.error(f"Error in collect_news_naver: {e}")
            return self._collect_news_rss()
    
    def _collect_news_rss(self) -> List[Dict]:
        """RSS 피드를 사용한 대체 뉴스 수집"""
        logger.info("Collecting news from RSS feeds")
        
        rss_feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'http://www.yonhapnews.co.kr/RSS/economy.xml',
            'http://rss.hankyung.com/feed/economy.xml'
        ]
        
        all_news = []
        conn = sqlite3.connect(self.db_path)
        
        try:
            import feedparser
            
            for feed_url in rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:20]:  # 각 피드에서 최대 20개
                        try:
                            title = entry.title
                            description = entry.get('summary', '')[:500]
                            link = entry.link
                            
                            # 날짜 파싱
                            pub_date = datetime.now()
                            if hasattr(entry, 'published_parsed'):
                                pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                            
                            # 감정 분석
                            sentiment_score = self._analyze_sentiment(title + " " + description)
                            
                            # 키워드 추출
                            keywords = self._extract_keywords(title + " " + description)
                            
                            # 데이터베이스에 저장
                            conn.execute('''
                                INSERT OR IGNORE INTO news_articles 
                                (title, description, link, pub_date, source, sentiment_score, keywords)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                title,
                                description,
                                link,
                                pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                                'RSS Feed',
                                sentiment_score,
                                json.dumps(keywords, ensure_ascii=False)
                            ))
                            
                            all_news.append({
                                'title': title,
                                'description': description,
                                'link': link,
                                'pub_date': pub_date,
                                'source': 'RSS Feed',
                                'sentiment_score': sentiment_score,
                                'keywords': keywords
                            })
                            
                        except Exception as e:
                            logger.error(f"Error processing RSS entry: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Error parsing RSS feed {feed_url}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Collected {len(all_news)} news articles from RSS")
            return all_news
            
        except ImportError:
            logger.error("feedparser not installed. Cannot collect RSS news.")
            return []
        except Exception as e:
            logger.error(f"Error in _collect_news_rss: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        간단한 감정 분석 (0~1 점수)
        실제로는 KoBERT 등의 모델을 사용해야 함
        """
        positive_words = ['상승', '급등', '호재', '성장', '증가', '개선', '긍정', '수익', '이익']
        negative_words = ['하락', '급락', '악재', '감소', '하락', '부진', '손실', '적자', '위험']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count + negative_count == 0:
            return 0.5
        
        return positive_count / (positive_count + negative_count)
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        간단한 키워드 추출
        실제로는 KoNLPy, TextRank 등을 사용해야 함
        """
        import re
        from collections import Counter
        
        # 불용어 제거 및 토큰화
        stop_words = ['의', '가', '이', '은', '들', '는', '좀', '잘', '과', '를', '으로', '에', '와', '한', '하다']
        words = re.findall(r'[가-힣]+', text)
        words = [w for w in words if len(w) > 1 and w not in stop_words]
        
        # 빈도수 계산
        word_counts = Counter(words)
        
        # 상위 키워드 추출
        keywords = [word for word, count in word_counts.most_common(top_n)]
        
        return keywords
    
    def collect_financial_statements(self, ticker: str) -> pd.DataFrame:
        """
        재무제표 데이터 수집
        현재는 기본적인 재무 지표만 수집
        """
        logger.info(f"Collecting financial statements for {ticker}")
        
        # 실제로는 DART API나 별도의 재무제표 API를 사용해야 함
        # 여기서는 yfinance의 기본 정보만 사용
        
        if not HAS_YFINANCE:
            logger.warning("yfinance not available for financial data")
            return pd.DataFrame()
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # 기본 재무 지표
            metrics = {
                'PER': info.get('trailingPE', np.nan),
                'PBR': info.get('priceToBook', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'ROA': info.get('returnOnAssets', np.nan),
                'DebtToEquity': info.get('debtToEquity', np.nan),
                'CurrentRatio': info.get('currentRatio', np.nan),
                'ProfitMargin': info.get('profitMargins', np.nan),
                'OperatingMargin': info.get('operatingMargins', np.nan)
            }
            
            conn = sqlite3.connect(self.db_path)
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for metric_name, value in metrics.items():
                if not pd.isna(value):
                    try:
                        conn.execute('''
                            INSERT OR REPLACE INTO financial_statements 
                            (ticker, date, metric_name, value, frequency)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            ticker,
                            current_date,
                            metric_name,
                            float(value),
                            'current'
                        ))
                    except Exception as e:
                        logger.error(f"Error inserting {metric_name}: {e}")
            
            conn.commit()
            conn.close()
            
            df = pd.DataFrame([metrics])
            df['ticker'] = ticker
            df['date'] = current_date
            
            logger.info(f"Collected financial metrics for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting financial statements for {ticker}: {e}")
            return pd.DataFrame()
    
    def update_all_data(self):
        """모든 데이터 업데이트"""
        logger.info("Starting full data update")
        
        # 1. 한국 주식 데이터
        korean_stocks = self.collect_korean_stocks(days=30)  # 최근 30일
        logger.info(f"Korean stocks: {len(korean_stocks)} records")
        
        # 2. 미국 주식 데이터
        us_stocks = self.collect_us_stocks(days=30)  # 최근 30일
        logger.info(f"US stocks: {len(us_stocks)} records")
        
        # 3. 뉴스 데이터
        news_data = self.collect_news_naver(query="주식 증권 투자", max_items=50)
        logger.info(f"News articles: {len(news_data)} items")
        
        # 4. 주요 종목 재무제표
        main_tickers = ['005930', '000660', 'AAPL', 'MSFT']  # 삼성전자, SK하이닉스, Apple, Microsoft
        for ticker in main_tickers:
            self.collect_financial_statements(ticker)
        
        logger.info("Full data update completed")
    
    def get_latest_prices(self, ticker: str, market: str = 'KR', days: int = 30) -> pd.DataFrame:
        """
        최신 주가 데이터 조회
        
        Args:
            ticker: 종목 코드
            market: 시장 구분 (KR/US)
            days: 조회 일수
            
        Returns:
            주가 데이터프레임
        """
        conn = sqlite3.connect(self.db_path)
        
        table_name = 'korean_stock_prices' if market == 'KR' else 'us_stock_prices'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        query = f'''
            SELECT * FROM {table_name}
            WHERE ticker = ? AND date >= ?
            ORDER BY date DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(ticker, start_date.strftime('%Y-%m-%d')))
        conn.close()
        
        return df
    
    def get_latest_news(self, limit: int = 10, keywords: List[str] = None) -> pd.DataFrame:
        """
        최신 뉴스 조회
        
        Args:
            limit: 조회 개수
            keywords: 필터링 키워드
            
        Returns:
            뉴스 데이터프레임
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM news_articles
            ORDER BY pub_date DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        # 키워드 필터링
        if keywords:
            mask = df['title'].str.contains('|'.join(keywords), case=False, na=False)
            df = df[mask]
        
        return df


# 테스트 및 사용 예제
if __name__ == "__main__":
    # 데이터 수집기 초기화
    collector = RealDataCollector()
    
    # 전체 데이터 업데이트
    collector.update_all_data()
    
    # 특정 종목 최신 가격 조회
    samsung_prices = collector.get_latest_prices('005930', market='KR')
    print(f"Samsung Electronics latest prices:\n{samsung_prices.head()}")
    
    # 최신 뉴스 조회
    latest_news = collector.get_latest_news(limit=5)
    print(f"\nLatest news:\n{latest_news[['title', 'sentiment_score']].head()}")