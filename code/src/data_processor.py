#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
금융 데이터 처리 모듈
- 수집된 데이터 전처리
- 벡터 DB 구축
- 데이터 분석 및 특성 추출
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer
import joblib
from typing import List, Dict, Any, Tuple, Optional
from db_client import get_supabase_client

class DataProcessor:
    """금융 데이터 처리 클래스"""
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.script_dir), "data")
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")
        
        # 디렉토리 초기화
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # 임베딩 모델 로드
        self.embedding_model = self._load_embedding_model()
        
        # 벡터 DB 관련 변수
        self.vector_db_index = None
        self.document_store = []
    
    def _load_embedding_model(self):
        """문장 임베딩 모델 로드"""
        try:
            # 한국어에 최적화된 모델
            return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            print(f"임베딩 모델 로드 실패: {e}")
            return None
    
    def process_stock_tickers(self):
        """종목 코드 데이터 처리"""
        try:
            print("종목 코드 데이터 처리 중...")
            
            # 최신 종목 코드 파일 찾기
            ticker_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.startswith('kor_ticker_')], reverse=True)
            if not ticker_files:
                print("종목 코드 파일이 없습니다.")
                return None
            
            ticker_file = os.path.join(self.raw_data_dir, ticker_files[0])
            ticker_df = pd.read_csv(ticker_file)
            
            # 데이터 전처리
            # 종목코드 6자리로 포맷팅
            ticker_df['종목코드'] = ticker_df['종목코드'].astype(str).str.zfill(6)
            
            # 결측치 처리
            if '섹터' in ticker_df.columns:
                ticker_df['섹터'] = ticker_df['섹터'].fillna('기타')
            
            # 처리된 데이터 저장
            output_file = os.path.join(self.processed_data_dir, "stock_tickers.pkl")
            ticker_df.to_pickle(output_file)
            
            print(f"종목 코드 데이터 처리 완료: {len(ticker_df)}개 종목, 저장 경로: {output_file}")
            return ticker_df
        except Exception as e:
            print(f"종목 코드 데이터 처리 실패: {e}")
            return None
    
    def process_stock_prices(self):
        """주식 가격 데이터 처리"""
        try:
            print("국내주식 가격 데이터 처리 중...")
            
            # 최신 가격 데이터 파일 찾기
            price_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.startswith('kor_price_')], reverse=True)
            if not price_files:
                print("가격 데이터 파일이 없습니다.")
                return None
            
            price_file = os.path.join(self.raw_data_dir, price_files[0])
            price_df = pd.read_csv(price_file)
            
            # 데이터 전처리
            # 날짜 형식 변환
            if '날짜' in price_df.columns:
                price_df['날짜'] = pd.to_datetime(price_df['날짜'])
            
            # 결측치 처리
            for col in ['시가', '고가', '저가', '종가', '거래량']:
                if col in price_df.columns:
                    price_df[col] = price_df[col].fillna(method='ffill')
            
            # 기술적 지표 추가
            price_df = self._add_technical_indicators(price_df)
            
            # 처리된 데이터 저장
            output_file = os.path.join(self.processed_data_dir, "stock_prices.pkl")
            price_df.to_pickle(output_file)
            
            print(f"국내주식 가격 데이터 처리 완료: {len(price_df)}개 레코드, 저장 경로: {output_file}")
            return price_df
        except Exception as e:
            print(f"국내주식 가격 데이터 처리 실패: {e}")
            return None
    
    def process_us_stock_prices(self):
        """
        미국주식 일별 가격 데이터 전처리
        - raw_data_dir의 us_price_YYYYMMDD.csv 파일 중 최신 파일 사용
        - 주요 컬럼: Date, Open, High, Low, Close, Volume, Ticker
        - 결측치 처리, 날짜 변환, 기술적 지표 추가, processed/us_stock_prices.pkl로 저장
        """
        try:
            print("미국주식 가격 데이터 처리 중...")
            # 최신 미국주식 가격 데이터 파일 찾기
            price_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.startswith('us_price_')], reverse=True)
            if not price_files:
                print("미국주식 가격 데이터 파일이 없습니다.")
                return None
            price_file = os.path.join(self.raw_data_dir, price_files[0])
            price_df = pd.read_csv(price_file)
            # 날짜 변환
            if 'Date' in price_df.columns:
                price_df['Date'] = pd.to_datetime(price_df['Date'])
            # 결측치 처리
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in price_df.columns:
                    price_df[col] = price_df[col].fillna(method='ffill')
            # 기술적 지표 추가 (종목별)
            result_dfs = []
            for ticker, group in price_df.groupby('Ticker'):
                group = group.sort_values('Date')
                # 이동평균선
                group['MA5'] = group['Close'].rolling(window=5).mean()
                group['MA20'] = group['Close'].rolling(window=20).mean()
                group['MA60'] = group['Close'].rolling(window=60).mean()
                # MACD
                ema_12 = group['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = group['Close'].ewm(span=26, adjust=False).mean()
                group['MACD'] = ema_12 - ema_26
                group['MACD_signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
                # RSI
                delta = group['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                group['RSI'] = 100 - (100 / (1 + rs))
                # 볼린저 밴드
                group['BB_mid'] = group['Close'].rolling(window=20).mean()
                std = group['Close'].rolling(window=20).std()
                group['BB_upper'] = group['BB_mid'] + 2 * std
                group['BB_lower'] = group['BB_mid'] - 2 * std
                result_dfs.append(group)
            if result_dfs:
                result_df = pd.concat(result_dfs, ignore_index=True)
            else:
                result_df = price_df
            # 처리된 데이터 저장
            output_file = os.path.join(self.processed_data_dir, "us_stock_prices.pkl")
            result_df.to_pickle(output_file)
            print(f"미국주식 가격 데이터 처리 완료: {len(result_df)}개 레코드, 저장 경로: {output_file}")
            return result_df
        except Exception as e:
            print(f"미국주식 가격 데이터 처리 실패: {e}")
            return None
    
    def process_us_stock_tickers(self):
        """
        미국 S&P 500 티커 데이터 전처리
        - raw_data_dir의 us_ticker_YYYYMMDD.csv 파일 중 최신 파일 사용
        - 주요 컬럼: Ticker, Name, Market, Sector
        - 결측치 처리, 컬럼명 정규화, processed/us_stock_tickers.pkl로 저장
        """
        try:
            print("미국 S&P 500 티커 데이터 처리 중...")
            # 최신 S&P 500 티커 파일 찾기
            ticker_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.startswith('us_ticker_')], reverse=True)
            if not ticker_files:
                print("미국 S&P 500 티커 파일이 없습니다.")
                return None
            ticker_file = os.path.join(self.raw_data_dir, ticker_files[0])
            ticker_df = pd.read_csv(ticker_file)
            # 컬럼명 정규화
            rename_map = {c: c.strip().capitalize() for c in ticker_df.columns}
            ticker_df = ticker_df.rename(columns=rename_map)
            # 결측치 처리
            for col in ['Ticker', 'Name', 'Market', 'Sector']:
                if col in ticker_df.columns:
                    ticker_df[col] = ticker_df[col].fillna('')
            # 저장
            output_file = os.path.join(self.processed_data_dir, "us_stock_tickers.pkl")
            ticker_df.to_pickle(output_file)
            print(f"미국 S&P 500 티커 데이터 처리 완료: {len(ticker_df)}개, 저장 경로: {output_file}")
            return ticker_df
        except Exception as e:
            print(f"미국 S&P 500 티커 데이터 처리 실패: {e}")
            return None
    
    def _add_technical_indicators(self, df):
        """주가 데이터에 기술적 지표 추가"""
        try:
            # 종목별로 그룹화하여 처리
            result_dfs = []
            
            for ticker, group in df.groupby('종목코드'):
                group = group.sort_values('날짜')
                
                # 이동평균선
                group['MA5'] = group['종가'].rolling(window=5).mean()
                group['MA20'] = group['종가'].rolling(window=20).mean()
                group['MA60'] = group['종가'].rolling(window=60).mean()
                
                # MACD
                ema_12 = group['종가'].ewm(span=12, adjust=False).mean()
                ema_26 = group['종가'].ewm(span=26, adjust=False).mean()
                group['MACD'] = ema_12 - ema_26
                group['MACD_signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
                
                # RSI
                delta = group['종가'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                group['RSI'] = 100 - (100 / (1 + rs))
                
                # 볼린저 밴드
                group['BB_mid'] = group['종가'].rolling(window=20).mean()
                std = group['종가'].rolling(window=20).std()
                group['BB_upper'] = group['BB_mid'] + 2 * std
                group['BB_lower'] = group['BB_mid'] - 2 * std
                
                # 결과 추가
                result_dfs.append(group)
            
            # 모든 데이터 결합
            if result_dfs:
                return pd.concat(result_dfs, ignore_index=True)
            return df
        except Exception as e:
            print(f"기술적 지표 추가 실패: {e}")
            return df
    
    def process_financial_statements(self):
        """재무제표 데이터 처리"""
        try:
            print("재무제표 데이터 처리 중...")
            
            # 최신 재무제표 데이터 파일 찾기
            fs_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.startswith('kor_fs_')], reverse=True)
            if not fs_files:
                print("재무제표 데이터 파일이 없습니다.")
                return None
            
            fs_file = os.path.join(self.raw_data_dir, fs_files[0])
            fs_df = pd.read_csv(fs_file)
            
            # 데이터 전처리
            # 종목코드 6자리로 포맷팅
            fs_df['종목코드'] = fs_df['종목코드'].astype(str).str.zfill(6)
            
            # 결측치 처리
            for col in fs_df.columns:
                if fs_df[col].dtype in [np.float64, np.int64]:
                    fs_df[col] = fs_df[col].fillna(0)
            
            # 재무비율 계산
            fs_df = self._calculate_financial_ratios(fs_df)
            
            # 처리된 데이터 저장
            output_file = os.path.join(self.processed_data_dir, "financial_statements.pkl")
            fs_df.to_pickle(output_file)
            
            print(f"재무제표 데이터 처리 완료: {len(fs_df)}개 레코드, 저장 경로: {output_file}")
            return fs_df
        except Exception as e:
            print(f"재무제표 데이터 처리 실패: {e}")
            return None
    
    def _calculate_financial_ratios(self, df):
        """재무제표 데이터로부터 재무비율 계산"""
        try:
            # 필요한 컬럼이 있는지 확인
            required_columns = ['매출액', '영업이익', '당기순이익', '자산총계', '부채총계', '자본총계']
            for col in required_columns:
                if col not in df.columns:
                    print(f"재무비율 계산에 필요한 컬럼이 없습니다: {col}")
                    return df
            
            # 영업이익률
            df['영업이익률'] = (df['영업이익'] / df['매출액']) * 100
            
            # 순이익률
            df['순이익률'] = (df['당기순이익'] / df['매출액']) * 100
            
            # 부채비율
            df['부채비율'] = (df['부채총계'] / df['자본총계']) * 100
            
            # 자기자본비율
            df['자기자본비율'] = (df['자본총계'] / df['자산총계']) * 100
            
            return df
        except Exception as e:
            print(f"재무비율 계산 실패: {e}")
            return df
    
    def process_valuation_metrics(self):
        """가치평가 지표 처리"""
        try:
            print("가치평가 지표 처리 중...")
            
            # 최신 가치평가 지표 파일 찾기
            value_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.startswith('kor_value_')], reverse=True)
            if not value_files:
                print("가치평가 지표 파일이 없습니다.")
                return None
            
            value_file = os.path.join(self.raw_data_dir, value_files[0])
            value_df = pd.read_csv(value_file)
            
            # 데이터 전처리
            # 종목코드 6자리로 포맷팅
            value_df['종목코드'] = value_df['종목코드'].astype(str).str.zfill(6)
            
            # 결측치 처리
            for col in value_df.columns:
                if value_df[col].dtype in [np.float64, np.int64]:
                    value_df[col] = value_df[col].fillna(value_df[col].median())
            
            # 처리된 데이터 저장
            output_file = os.path.join(self.processed_data_dir, "valuation_metrics.pkl")
            value_df.to_pickle(output_file)
            
            print(f"가치평가 지표 처리 완료: {len(value_df)}개 레코드, 저장 경로: {output_file}")
            return value_df
        except Exception as e:
            print(f"가치평가 지표 처리 실패: {e}")
            return None
    
    def build_vector_db(self):
        """벡터 DB 구축"""
        try:
            print("벡터 DB 구축 중...")
            
            if not self.embedding_model:
                print("임베딩 모델이 로드되지 않았습니다.")
                return False
            
            # 종목 정보 로드
            ticker_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.startswith('kor_ticker_')], reverse=True)
            if not ticker_files:
                print("종목 코드 파일이 없습니다.")
                return False
            
            ticker_file = os.path.join(self.raw_data_dir, ticker_files[0])
            ticker_df = pd.read_csv(ticker_file)
            
            # 섹터 정보 로드
            sector_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.startswith('kor_sector_')], reverse=True)
            if sector_files:
                sector_file = os.path.join(self.raw_data_dir, sector_files[0])
                sector_df = pd.read_csv(sector_file)
            else:
                sector_df = None
            
            # 가치평가 지표 로드
            value_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.startswith('kor_value_')], reverse=True)
            if value_files:
                value_file = os.path.join(self.raw_data_dir, value_files[0])
                value_df = pd.read_csv(value_file)
            else:
                value_df = None
            
            # 문서 생성
            documents = []
            
            # 종목 정보 문서
            for _, row in ticker_df.iterrows():
                ticker = row['종목코드']
                name = row['종목명']
                market = row.get('시장구분', '')
                
                # 섹터 정보 추가
                sector = ''
                if sector_df is not None:
                    sector_row = sector_df[sector_df['종목코드'] == ticker]
                    if not sector_row.empty:
                        sector = sector_row.iloc[0].get('sector', '')
                
                # 가치평가 지표 추가
                per, pbr = '', ''
                if value_df is not None:
                    value_row = value_df[value_df['종목코드'] == ticker]
                    if not value_row.empty:
                        per = value_row.iloc[0].get('PER', '')
                        pbr = value_row.iloc[0].get('PBR', '')
                
                # 문서 생성
                doc = f"{name}({ticker})는 {market} 시장에 상장된 {sector} 섹터 기업입니다."
                if per and pbr:
                    doc += f" PER은 {per}, PBR은 {pbr}입니다."
                
                documents.append({
                    'content': doc,
                    'metadata': {
                        'ticker': ticker,
                        'name': name,
                        'market': market,
                        'sector': sector
                    }
                })
            
            # 문서 임베딩 생성
            embeddings = []
            for doc in documents:
                embedding = self.embedding_model.encode(doc['content'])
                embeddings.append(embedding)
            
            # FAISS 인덱스 생성
            embeddings_np = np.array(embeddings).astype('float32')
            dimension = embeddings_np.shape[1]
            self.vector_db_index = faiss.IndexFlatL2(dimension)
            self.vector_db_index.add(embeddings_np)
            
            # 문서 저장
            self.document_store = documents
            
            # 벡터 DB 저장
            index_file = os.path.join(self.processed_data_dir, "vector_db_index.bin")
            faiss.write_index(self.vector_db_index, index_file)
            
            # 문서 저장
            docs_file = os.path.join(self.processed_data_dir, "vector_db_documents.json")
            with open(docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_store, f, ensure_ascii=False, indent=2)
            
            print(f"벡터 DB 구축 완료: {len(documents)}개 문서, 저장 경로: {index_file}")
            return True
        except Exception as e:
            print(f"벡터 DB 구축 실패: {e}")
            return False
    
    def search_vector_db(self, query, top_k=5):
        """벡터 DB 검색"""
        try:
            if not self.embedding_model or self.vector_db_index is None:
                # 벡터 DB 로드 시도
                index_file = os.path.join(self.processed_data_dir, "vector_db_index.bin")
                docs_file = os.path.join(self.processed_data_dir, "vector_db_documents.json")
                
                if os.path.exists(index_file) and os.path.exists(docs_file):
                    self.vector_db_index = faiss.read_index(index_file)
                    with open(docs_file, 'r', encoding='utf-8') as f:
                        self.document_store = json.load(f)
                else:
                    print("벡터 DB가 구축되지 않았습니다.")
                    return []
            
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode(query)
            query_embedding_np = np.array([query_embedding]).astype('float32')
            
            # 검색
            distances, indices = self.vector_db_index.search(query_embedding_np, top_k)
            
            # 결과 반환
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.document_store):
                    doc = self.document_store[idx]
                    results.append({
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'distance': float(distances[0][i])
                    })
            
            return results
        except Exception as e:
            print(f"벡터 DB 검색 실패: {e}")
            return []
    
    def run_all_processors(self):
        """모든 데이터 처리 함수 실행"""
        print("모든 데이터 처리 시작...")
        
        # 종목 코드 데이터 처리
        self.process_stock_tickers()
        
        # 주식 가격 데이터 처리
        self.process_stock_prices()
        
        # 재무제표 데이터 처리
        self.process_financial_statements()
        
        # 가치평가 지표 처리
        self.process_valuation_metrics()
        
        # 벡터 DB 구축
        self.build_vector_db()
        
        print("모든 데이터 처리 완료")

# 모듈 테스트용 코드
if __name__ == "__main__":
    processor = DataProcessor()
    processor.run_all_processors() 