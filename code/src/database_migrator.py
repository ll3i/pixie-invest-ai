#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Migration Tool - CSV to Supabase Migration
- 기존 data_processing.ipynb 결과를 DB로 마이그레이션
- 5년치 국내주식 데이터 + 미국주식(yfinance) 통합
- 배포 환경 대응을 위한 완전한 DB 전환
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from tqdm import tqdm
import time
import math
import pickle

from db_client import get_supabase_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """CSV 데이터를 Supabase PostgreSQL DB로 마이그레이션"""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.batch_size = 500
        
        # 미국 주요 종목 리스트 (20개)
        self.us_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'TSLA', 'META', 'BRK-B', 'JNJ', 'V',
            'WMT', 'JPM', 'MA', 'PG', 'HD',
            'SPY', 'QQQ', 'VTI', 'DIS', 'NFLX'
        ]
    
    def display_table_creation_sql(self):
        """테이블 생성 SQL 출력 (Supabase에서 수동 실행용)"""
        
        sql_statements = {
            'korean_stock_prices': '''
            CREATE TABLE korean_stock_prices (
                id BIGSERIAL PRIMARY KEY,
                ticker VARCHAR(6) NOT NULL,
                date DATE NOT NULL,
                open_price NUMERIC(12,2),
                high_price NUMERIC(12,2),
                low_price NUMERIC(12,2),
                close_price NUMERIC(12,2),
                volume BIGINT,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, date)
            );
            CREATE INDEX idx_korean_prices_ticker_date ON korean_stock_prices(ticker, date);
            ''',
            
            'us_stock_prices': '''
            CREATE TABLE us_stock_prices (
                id BIGSERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                open_price NUMERIC(12,2),
                high_price NUMERIC(12,2),
                low_price NUMERIC(12,2),
                close_price NUMERIC(12,2),
                adj_close NUMERIC(12,2),
                volume BIGINT,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, date)
            );
            CREATE INDEX idx_us_prices_ticker_date ON us_stock_prices(ticker, date);
            ''',
            
            'stock_metadata': '''
            CREATE TABLE stock_metadata (
                id BIGSERIAL PRIMARY KEY,
                ticker VARCHAR(10) UNIQUE NOT NULL,
                company_name VARCHAR(200),
                market VARCHAR(20),
                sector VARCHAR(100),
                industry VARCHAR(100),
                country VARCHAR(3),
                currency VARCHAR(3),
                market_cap BIGINT,
                stock_type VARCHAR(20),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            
            'financial_statements': '''
            CREATE TABLE financial_statements (
                id BIGSERIAL PRIMARY KEY,
                ticker VARCHAR(6) NOT NULL,
                date DATE NOT NULL,
                account_name VARCHAR(100) NOT NULL,
                value NUMERIC(15,2),
                period_type VARCHAR(1) CHECK (period_type IN ('Y', 'Q')),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, date, account_name, period_type)
            );
            CREATE INDEX idx_fs_ticker_date ON financial_statements(ticker, date);
            ''',
            
            'valuation_metrics': '''
            CREATE TABLE valuation_metrics (
                id BIGSERIAL PRIMARY KEY,
                ticker VARCHAR(6) NOT NULL,
                date DATE NOT NULL,
                metric_type VARCHAR(10) NOT NULL,
                value NUMERIC(10,4),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, date, metric_type)
            );
            CREATE INDEX idx_valuation_ticker_date ON valuation_metrics(ticker, date);
            ''',
            
            'stock_analysis': '''
            CREATE TABLE stock_analysis (
                id BIGSERIAL PRIMARY KEY,
                ticker VARCHAR(6) NOT NULL,
                company_name VARCHAR(200),
                current_price NUMERIC(12,2),
                market_cap BIGINT,
                revenue_growth NUMERIC(8,4),
                profit_margin NUMERIC(8,4),
                debt_ratio NUMERIC(8,4),
                per_ratio NUMERIC(8,4),
                pbr_ratio NUMERIC(8,4),
                evaluation_score INTEGER,
                evaluation_grade VARCHAR(20),
                evaluation_reasons TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker)
            );
            ''',
            
            'enhanced_news': '''
            CREATE TABLE enhanced_news (
                id BIGSERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT,
                url TEXT UNIQUE,
                published_date TIMESTAMP,
                source VARCHAR(100),
                sentiment_score NUMERIC(5,4),
                keywords TEXT[],
                related_tickers VARCHAR(6)[],
                importance_score INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE INDEX idx_news_date ON enhanced_news(published_date);
            CREATE INDEX idx_news_tickers ON enhanced_news USING GIN(related_tickers);
            '''
        }
        
        print("🗄️ Supabase 테이블 생성 SQL")
        print("=" * 60)
        print("다음 SQL을 Supabase SQL Editor에서 순서대로 실행하세요:\n")
        
        for table_name, sql in sql_statements.items():
            print(f"-- {table_name.upper()} 테이블")
            print(sql.strip())
            print("\n" + "-" * 40 + "\n")
    
    def load_korean_data_from_csv(self) -> Dict[str, pd.DataFrame]:
        """기존 CSV 파일에서 한국 주식 데이터 로드"""
        
        data = {}
        csv_files = {
            'ticker': 'kor_ticker_*.csv',
            'prices': 'kor_price_*.csv',
            'financial': 'kor_fs_*.csv',
            'valuation': 'kor_value_*.csv',
            'analysis': 'stock_evaluation_results.csv'
        }
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        for data_type, pattern in csv_files.items():
            try:
                # 최신 파일 찾기
                import glob
                search_paths = [
                    base_dir,
                    os.path.join(base_dir, 'docs'),
                    os.path.join(base_dir, 'data', 'raw'),
                    os.path.join(base_dir, 'data'),
                    os.getcwd(),  # 현재 작업 디렉토리
                ]
                
                files = []
                for search_path in search_paths:
                    if os.path.exists(search_path):
                        found_files = glob.glob(os.path.join(search_path, pattern))
                        files.extend(found_files)
                
                if files:
                    latest_file = max(files, key=os.path.getctime)
                    logger.info(f"{data_type} 파일 발견: {latest_file}")
                    
                    # 데이터 로드시 오류 처리 강화
                    df = pd.read_csv(latest_file, encoding='utf-8-sig')
                    
                    # 종목코드 컬럼 처리
                    if '종목코드' in df.columns:
                        df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
                    elif 'ticker' in df.columns:
                        df['ticker'] = df['ticker'].astype(str).str.zfill(6)
                    
                    # 빈 데이터프레임 체크
                    if len(df) == 0:
                        logger.warning(f"{data_type} 파일이 비어있습니다: {latest_file}")
                        continue
                    
                    data[data_type] = df
                    logger.info(f"{data_type} 데이터 로드 완료: {len(df):,} 레코드")
                    logger.info(f"컬럼: {list(df.columns)}")
                else:
                    logger.warning(f"{data_type} 파일을 찾을 수 없습니다: {pattern}")
                    logger.info(f"검색 경로: {search_paths}")
                    
            except Exception as e:
                logger.error(f"{data_type} 데이터 로드 실패: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        return data
    
    def download_us_stock_data(self, years: int = 5) -> pd.DataFrame:
        """yfinance로 미국 주식 5년치 데이터 다운로드"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        all_data = []
        
        logger.info(f"미국 주식 {len(self.us_stocks)}개 종목 다운로드...")
        
        for ticker in tqdm(self.us_stocks, desc="US Stocks"):
            try:
                stock = yf.Ticker(ticker)
                
                # 가격 데이터
                hist = stock.history(start=start_date, end=end_date)
                if not hist.empty:
                    hist.reset_index(inplace=True)
                    hist['ticker'] = ticker
                    hist['date'] = hist['Date'].dt.date
                    
                    # 컬럼 표준화
                    hist = hist[['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
                    hist.columns = ['ticker', 'date', 'open_price', 'high_price', 'low_price', 
                                  'close_price', 'adj_close', 'volume']
                    
                    all_data.append(hist)
                
                time.sleep(0.1)  # API 제한 방지
                
            except Exception as e:
                logger.warning(f"미국 주식 {ticker} 다운로드 실패: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"미국 주식 데이터 다운로드 완료: {len(combined_df):,} 레코드")
            return combined_df
        
        return pd.DataFrame()
    
    def batch_upsert(self, table_name: str, data: pd.DataFrame) -> bool:
        """배치로 데이터를 DB에 upsert"""
        
        if data.empty:
            logger.warning(f"업로드할 데이터가 없습니다: {table_name}")
            return False
        
        # NaN 값 처리
        data = data.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        total_success = 0
        total_chunks = (len(data) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"{table_name} 업로드 시작: {len(data):,} 레코드")
        
        for i in tqdm(range(0, len(data), self.batch_size), desc=f"Uploading {table_name}"):
            chunk = data.iloc[i:i+self.batch_size]
            
            try:
                records = chunk.to_dict('records')
                
                # 날짜 형식 변환
                for record in records:
                    for key, value in record.items():
                        if isinstance(value, (pd.Timestamp, np.datetime64)):
                            record[key] = pd.to_datetime(value).strftime('%Y-%m-%d')
                        elif key == 'date' and hasattr(value, 'strftime'):
                            record[key] = value.strftime('%Y-%m-%d')
                
                result = self.supabase.table(table_name).upsert(records).execute()
                
                if result.data:
                    total_success += len(result.data)
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"배치 업로드 실패 {table_name} [{i}:{i+self.batch_size}]: {e}")
                continue
        
        logger.info(f"{table_name} 업로드 완료: {total_success:,}/{len(data):,}")
        return total_success > 0
    
    def migrate_korean_prices(self, korean_data: Dict[str, pd.DataFrame]):
        """한국 주식 가격 데이터 마이그레이션"""
        
        if 'prices' not in korean_data:
            logger.error("가격 데이터를 찾을 수 없습니다")
            return False
        
        prices = korean_data['prices'].copy()
        logger.info(f"가격 데이터 원본 컬럼: {list(prices.columns)}")
        logger.info(f"가격 데이터 샘플:\n{prices.head()}")
        
        # 컬럼명 표준화
        column_mapping = {
            '종목코드': 'ticker',
            '날짜': 'date',
            '시가': 'open_price',
            '고가': 'high_price',
            '저가': 'low_price',  
            '종가': 'close_price',
            '거래량': 'volume'
        }
        
        # 실제 존재하는 컬럼만 매핑
        available_mapping = {}
        for old_col, new_col in column_mapping.items():
            if old_col in prices.columns:
                available_mapping[old_col] = new_col
            else:
                logger.warning(f"컬럼 '{old_col}'을 찾을 수 없습니다")
        
        if not available_mapping:
            logger.error("매핑 가능한 컬럼이 없습니다")
            return False
        
        prices = prices.rename(columns=available_mapping)
        
        # 필수 컬럼 확인
        required_cols = ['ticker', 'date', 'close_price']
        existing_cols = [col for col in required_cols if col in prices.columns]
        
        if len(existing_cols) < len(required_cols):
            logger.error(f"필수 컬럼 누락: {set(required_cols) - set(existing_cols)}")
            return False
        
        # 사용 가능한 컬럼만 선택
        final_cols = [col for col in ['ticker', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'] if col in prices.columns]
        prices = prices[final_cols]
        
        # 날짜 변환
        try:
            prices['date'] = pd.to_datetime(prices['date'])
        except Exception as e:
            logger.error(f"날짜 변환 실패: {e}")
            return False
        
        # 데이터 타입 확인 및 정리
        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
            if col in prices.columns:
                prices[col] = pd.to_numeric(prices[col], errors='coerce')
        
        # NaN 제거
        prices = prices.dropna()
        
        if len(prices) == 0:
            logger.error("정리 후 데이터가 없습니다")
            return False
        
        logger.info(f"최종 가격 데이터: {len(prices):,} 레코드, 컬럼: {list(prices.columns)}")
        
        return self.batch_upsert('korean_stock_prices', prices)
    
    def migrate_us_prices(self, us_data: pd.DataFrame):
        """미국 주식 가격 데이터 마이그레이션"""
        
        if us_data.empty:
            logger.error("미국 주식 데이터가 없습니다")
            return False
        
        return self.batch_upsert('us_stock_prices', us_data)
    
    def migrate_metadata(self, korean_data: Dict[str, pd.DataFrame]):
        """주식 메타데이터 마이그레이션"""
        
        metadata_records = []
        
        # 한국 주식 메타데이터
        if 'ticker' in korean_data:
            korean_tickers = korean_data['ticker']
            
            for _, row in korean_tickers.iterrows():
                metadata_records.append({
                    'ticker': row['종목코드'],
                    'company_name': row.get('종목명', ''),
                    'market': row.get('시장구분', ''),
                    'market_cap': row.get('시가총액', 0),
                    'stock_type': row.get('종목구분', ''),
                    'country': 'KOR',
                    'currency': 'KRW'
                })
        
        # 미국 주식 메타데이터
        for ticker in self.us_stocks:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                metadata_records.append({
                    'ticker': ticker,
                    'company_name': info.get('longName', ticker),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'country': 'USA',
                    'currency': 'USD'
                })
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"미국 주식 {ticker} 메타데이터 수집 실패: {e}")
        
        if metadata_records:
            metadata_df = pd.DataFrame(metadata_records)
            return self.batch_upsert('stock_metadata', metadata_df)
        
        return False
    
    def migrate_financial_statements(self, korean_data: Dict[str, pd.DataFrame]):
        """재무제표 데이터 마이그레이션"""
        
        if 'financial' not in korean_data:
            logger.error("재무제표 데이터를 찾을 수 없습니다")
            return False
        
        fs_data = korean_data['financial'].copy()
        
        # 컬럼명 표준화
        column_mapping = {
            '종목코드': 'ticker',
            '기준일': 'date',
            '계정': 'account_name',
            '값': 'value',
            '공시구분': 'period_type'
        }
        
        fs_data = fs_data.rename(columns=column_mapping)
        
        # 기간 구분 변환 (y -> Y, q -> Q)
        fs_data['period_type'] = fs_data['period_type'].str.upper()
        
        # 날짜 변환
        fs_data['date'] = pd.to_datetime(fs_data['date'])
        
        fs_data = fs_data[['ticker', 'date', 'account_name', 'value', 'period_type']]
        
        return self.batch_upsert('financial_statements', fs_data)
    
    def migrate_valuation_metrics(self, korean_data: Dict[str, pd.DataFrame]):
        """밸류에이션 지표 마이그레이션"""
        
        if 'valuation' not in korean_data:
            logger.error("밸류에이션 데이터를 찾을 수 없습니다")
            return False
        
        valuation_data = korean_data['valuation'].copy()
        
        # 컬럼명 표준화
        column_mapping = {
            '종목코드': 'ticker',
            '기준일': 'date',
            '지표': 'metric_type',
            '값': 'value'
        }
        
        valuation_data = valuation_data.rename(columns=column_mapping)
        
        # 날짜 변환
        valuation_data['date'] = pd.to_datetime(valuation_data['date'])
        
        valuation_data = valuation_data[['ticker', 'date', 'metric_type', 'value']]
        
        return self.batch_upsert('valuation_metrics', valuation_data)
    
    def migrate_stock_analysis(self, korean_data: Dict[str, pd.DataFrame]):
        """주식 분석 결과 마이그레이션"""
        
        if 'analysis' not in korean_data:
            logger.error("분석 결과 데이터를 찾을 수 없습니다")
            return False
        
        analysis_data = korean_data['analysis'].copy()
        
        # 컬럼명 표준화
        column_mapping = {
            '종목코드': 'ticker',
            '종목명': 'company_name',
            '현재가': 'current_price',
            '시가총액': 'market_cap',
            '매출성장률': 'revenue_growth',
            '순이익률': 'profit_margin',
            '부채비율': 'debt_ratio',
            'PER': 'per_ratio',
            'PBR': 'pbr_ratio',
            '평가점수': 'evaluation_score',
            '종합평가': 'evaluation_grade',
            '평가이유': 'evaluation_reasons'
        }
        
        analysis_data = analysis_data.rename(columns=column_mapping)
        
        # 필요한 컬럼만 선택
        columns_to_keep = ['ticker', 'company_name', 'current_price', 'market_cap', 
                          'revenue_growth', 'profit_margin', 'debt_ratio', 'per_ratio', 
                          'pbr_ratio', 'evaluation_score', 'evaluation_grade', 'evaluation_reasons']
        
        analysis_data = analysis_data[columns_to_keep]
        
        return self.batch_upsert('stock_analysis', analysis_data)
    
    def run_migration(self):
        """전체 마이그레이션 실행"""
        
        logger.info("🚀 데이터베이스 마이그레이션 시작")
        logger.info("=" * 50)
        
        # 1. 테이블 생성 SQL 표시
        self.display_table_creation_sql()
        
        confirm = input("\n테이블 생성이 완료되었나요? (y/N): ")
        if confirm.lower() not in ['y', 'yes']:
            print("테이블을 생성한 후 다시 실행해주세요.")
            return
        
        # 2. 한국 주식 데이터 로드
        logger.info("📊 한국 주식 데이터 로드")
        korean_data = self.load_korean_data_from_csv()
        
        # 3. 미국 주식 데이터 다운로드
        logger.info("🇺🇸 미국 주식 데이터 다운로드")
        us_data = self.download_us_stock_data(years=5)
        
        # 4. 각 테이블별 마이그레이션 실행
        migrations = [
            ("한국 주식 가격", lambda: self.migrate_korean_prices(korean_data)),
            ("미국 주식 가격", lambda: self.migrate_us_prices(us_data)),
            ("주식 메타데이터", lambda: self.migrate_metadata(korean_data)),
            ("재무제표", lambda: self.migrate_financial_statements(korean_data)),
            ("밸류에이션 지표", lambda: self.migrate_valuation_metrics(korean_data)),
            ("주식 분석", lambda: self.migrate_stock_analysis(korean_data))
        ]
        
        results = {}
        for name, migration_func in migrations:
            logger.info(f"📤 {name} 마이그레이션 시작")
            try:
                success = migration_func()
                results[name] = "✅ 성공" if success else "❌ 실패"
            except Exception as e:
                logger.error(f"{name} 마이그레이션 실패: {e}")
                results[name] = f"❌ 오류: {str(e)}"
        
        # 5. 결과 요약
        logger.info("\n" + "=" * 50)
        logger.info("📋 마이그레이션 결과 요약")
        logger.info("=" * 50)
        
        for name, status in results.items():
            logger.info(f"{name}: {status}")
        
        # 6. DB 상태 확인
        self.verify_migration()
    
    def verify_migration(self):
        """마이그레이션 결과 검증"""
        
        tables = [
            'korean_stock_prices',
            'us_stock_prices', 
            'stock_metadata',
            'financial_statements',
            'valuation_metrics',
            'stock_analysis'
        ]
        
        logger.info("\n📊 DB 상태 확인")
        logger.info("-" * 30)
        
        for table in tables:
            try:
                result = self.supabase.table(table).select('*', count='exact').limit(1).execute()
                count = result.count if hasattr(result, 'count') else '확인불가'
                logger.info(f"{table}: {count:,} 레코드" if isinstance(count, int) else f"{table}: {count}")
            except Exception as e:
                logger.info(f"{table}: ❌ 오류 ({e})")
        
        logger.info("\n🎉 마이그레이션 완료!")
        logger.info("이제 배포 환경에서도 안정적인 데이터 서비스가 가능합니다.")

def main():
    """메인 실행 함수"""
    
    print("🔄 MINERVA 데이터베이스 마이그레이션 도구")
    print("=" * 50)
    print("- 기존 CSV 파일 → Supabase PostgreSQL 마이그레이션")
    print("- 한국 주식 5년치 데이터 (기존 노트북 기반)")  
    print("- 미국 주식 5년치 데이터 (yfinance)")
    print("- 총 예상 데이터량: ~100만 레코드")
    print("- 예상 소요시간: 1-2시간")
    
    migrator = DatabaseMigrator()
    migrator.run_migration()

if __name__ == "__main__":
    main()