#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
import re
import pickle
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """검색 결과"""
    종목명: str
    종목코드: str
    현재가: float
    시가총액: float
    매출성장률: Optional[float]
    순이익률: Optional[float]
    부채비율: Optional[float]
    PER: Optional[float]
    PBR: Optional[float]
    평가점수: int
    종합평가: str
    유사도: float

class StockSearchEngine:
    """주식 검색 엔진"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        self.stock_data = None
        self.idf_dict = None
        self.vectors = None
        
    def load_evaluation_data(self, file_path: str = None) -> pd.DataFrame:
        """평가 데이터 로드"""
        try:
            if file_path is None:
                file_path = os.path.join(self.data_dir, 'stock_evaluation_results.csv')
            
            df = pd.read_csv(file_path)
            df = df[df['평가점수'] != 0]  # 평가점수가 0인 항목 제거
            df.reset_index(drop=True, inplace=True)
            
            logger.info(f"평가 데이터 로드 완료: {len(df)}개 종목")
            return df
            
        except Exception as e:
            logger.error(f"평가 데이터 로드 실패: {e}")
            raise
    
    def create_text_for_embedding(self, row: pd.Series) -> str:
        """임베딩을 위한 텍스트 생성"""
        text = f"종목명: {row['종목명']}, 종목코드: {row['종목코드']}, "
        text += f"현재가: {row['현재가']}, 시가총액: {row['시가총액']}, "
        text += f"매출성장률: {row['매출성장률']}, 순이익률: {row['순이익률']}, "
        text += f"부채비율: {row['부채비율']}, PER: {row['PER']}, PBR: {row['PBR']}, "
        text += f"평가점수: {row['평가점수']}, 종합평가: {row['종합평가']}, "
        text += f"평가이유: {row['평가이유']}"
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화"""
        return re.findall(r'\w+', text.lower())
    
    def compute_tf(self, text: str) -> Dict[str, float]:
        """TF (Term Frequency) 계산"""
        tokens = self.tokenize(text)
        if not tokens:
            return {}
        
        tf_dict = Counter(tokens)
        total_tokens = len(tokens)
        
        for word in tf_dict:
            tf_dict[word] = tf_dict[word] / float(total_tokens)
        
        return tf_dict
    
    def compute_idf(self, corpus: List[str]) -> Dict[str, float]:
        """IDF (Inverse Document Frequency) 계산"""
        idf_dict = {}
        N = len(corpus)
        
        # 모든 고유 단어 수집
        all_words = set()
        for text in corpus:
            tokens = self.tokenize(text)
            all_words.update(tokens)
        
        # IDF 계산
        for word in all_words:
            count = sum(1 for text in corpus if word in self.tokenize(text))
            idf_dict[word] = math.log(N / float(count)) if count > 0 else 0
        
        return idf_dict
    
    def compute_tfidf(self, tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
        """TF-IDF 계산"""
        tfidf = {}
        for word, tf_value in tf.items():
            tfidf[word] = tf_value * idf.get(word, 0)
        return tfidf
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """코사인 유사도 계산"""
        intersection = set(vec1.keys()) & set(vec2.keys())
        
        if not intersection:
            return 0.0
        
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        
        sum1 = sum(vec1[x]**2 for x in vec1.keys())
        sum2 = sum(vec2[x]**2 for x in vec2.keys())
        
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
        if denominator == 0:
            return 0.0
        
        return float(numerator) / denominator
    
    def build_search_index(self, df: pd.DataFrame) -> None:
        """검색 인덱스 구축"""
        try:
            # 텍스트 데이터 생성
            df['text'] = df.apply(self.create_text_for_embedding, axis=1)
            corpus = df['text'].tolist()
            
            # IDF 계산
            self.idf_dict = self.compute_idf(corpus)
            
            # 벡터 생성
            self.vectors = []
            for text in corpus:
                tf = self.compute_tf(text)
                tfidf = self.compute_tfidf(tf, self.idf_dict)
                self.vectors.append(tfidf)
            
            # 데이터 저장
            self.stock_data = df.copy()
            self.stock_data['vector'] = self.vectors
            
            # 파일로 저장
            stock_data_file = os.path.join(self.data_dir, 'stock_data.pkl')
            idf_file = os.path.join(self.data_dir, 'idf.pkl')
            
            self.stock_data.to_pickle(stock_data_file)
            with open(idf_file, 'wb') as f:
                pickle.dump(self.idf_dict, f)
            
            logger.info(f"검색 인덱스 구축 완료: {len(self.stock_data)}개 종목")
            
        except Exception as e:
            logger.error(f"검색 인덱스 구축 실패: {e}")
            raise
    
    def load_search_index(self) -> None:
        """검색 인덱스 로드"""
        try:
            stock_data_file = os.path.join(self.data_dir, 'stock_data.pkl')
            idf_file = os.path.join(self.data_dir, 'idf.pkl')
            
            if not os.path.exists(stock_data_file) or not os.path.exists(idf_file):
                logger.warning("검색 인덱스 파일이 없습니다. 새로 구축합니다.")
                df = self.load_evaluation_data()
                self.build_search_index(df)
                return
            
            self.stock_data = pd.read_pickle(stock_data_file)
            with open(idf_file, 'rb') as f:
                self.idf_dict = pickle.load(f)
            
            self.vectors = self.stock_data['vector'].tolist()
            
            logger.info(f"검색 인덱스 로드 완료: {len(self.stock_data)}개 종목")
            
        except Exception as e:
            logger.error(f"검색 인덱스 로드 실패: {e}")
            raise
    
    def search_stocks(self, query: str, n_results: int = 10, 
                     min_score: int = 0, max_per: float = None, 
                     max_pbr: float = None) -> List[SearchResult]:
        """주식 검색"""
        try:
            if self.stock_data is None or self.idf_dict is None:
                self.load_search_index()
            
            # 쿼리 벡터화
            query_tf = self.compute_tf(query)
            query_vector = self.compute_tfidf(query_tf, self.idf_dict)
            
            # 유사도 계산
            similarities = []
            for i, row in self.stock_data.iterrows():
                similarity = self.cosine_similarity(query_vector, row['vector'])
                similarities.append((i, similarity))
            
            # 유사도로 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 필터링 및 결과 생성
            results = []
            for idx, similarity in similarities[:n_results * 2]:  # 더 많은 후보를 가져와서 필터링
                stock_info = self.stock_data.iloc[idx]
                
                # 기본 필터링
                if stock_info['평가점수'] < min_score:
                    continue
                
                # PER 필터링
                if max_per is not None and stock_info['PER'] is not None:
                    if stock_info['PER'] > max_per:
                        continue
                
                # PBR 필터링
                if max_pbr is not None and stock_info['PBR'] is not None:
                    if stock_info['PBR'] > max_pbr:
                        continue
                
                result = SearchResult(
                    종목명=stock_info['종목명'],
                    종목코드=stock_info['종목코드'],
                    현재가=stock_info['현재가'],
                    시가총액=stock_info['시가총액'],
                    매출성장률=stock_info['매출성장률'],
                    순이익률=stock_info['순이익률'],
                    부채비율=stock_info['부채비율'],
                    PER=stock_info['PER'],
                    PBR=stock_info['PBR'],
                    평가점수=stock_info['평가점수'],
                    종합평가=stock_info['종합평가'],
                    유사도=similarity
                )
                
                results.append(result)
                
                if len(results) >= n_results:
                    break
            
            logger.info(f"검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"주식 검색 실패: {e}")
            raise
    
    def search_by_criteria(self, criteria: Dict[str, any]) -> List[SearchResult]:
        """조건별 검색"""
        try:
            if self.stock_data is None:
                self.load_search_index()
            
            # 기본 필터링
            filtered_data = self.stock_data.copy()
            
            # 평가점수 필터링
            if 'min_score' in criteria:
                filtered_data = filtered_data[filtered_data['평가점수'] >= criteria['min_score']]
            
            # PER 필터링
            if 'max_per' in criteria:
                filtered_data = filtered_data[
                    (filtered_data['PER'].isna()) | (filtered_data['PER'] <= criteria['max_per'])
                ]
            
            # PBR 필터링
            if 'max_pbr' in criteria:
                filtered_data = filtered_data[
                    (filtered_data['PBR'].isna()) | (filtered_data['PBR'] <= criteria['max_pbr'])
                ]
            
            # 매출성장률 필터링
            if 'min_revenue_growth' in criteria:
                filtered_data = filtered_data[
                    (filtered_data['매출성장률'].isna()) | 
                    (filtered_data['매출성장률'] >= criteria['min_revenue_growth'])
                ]
            
            # 순이익률 필터링
            if 'min_profit_margin' in criteria:
                filtered_data = filtered_data[
                    (filtered_data['순이익률'].isna()) | 
                    (filtered_data['순이익률'] >= criteria['min_profit_margin'])
                ]
            
            # 부채비율 필터링
            if 'max_debt_ratio' in criteria:
                filtered_data = filtered_data[
                    (filtered_data['부채비율'].isna()) | 
                    (filtered_data['부채비율'] <= criteria['max_debt_ratio'])
                ]
            
            # 정렬
            sort_by = criteria.get('sort_by', '평가점수')
            ascending = criteria.get('ascending', False)
            filtered_data = filtered_data.sort_values(sort_by, ascending=ascending)
            
            # 결과 생성
            results = []
            for _, row in filtered_data.head(criteria.get('limit', 10)).iterrows():
                result = SearchResult(
                    종목명=row['종목명'],
                    종목코드=row['종목코드'],
                    현재가=row['현재가'],
                    시가총액=row['시가총액'],
                    매출성장률=row['매출성장률'],
                    순이익률=row['순이익률'],
                    부채비율=row['부채비율'],
                    PER=row['PER'],
                    PBR=row['PBR'],
                    평가점수=row['평가점수'],
                    종합평가=row['종합평가'],
                    유사도=1.0  # 조건 검색에서는 유사도 의미 없음
                )
                results.append(result)
            
            logger.info(f"조건 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"조건 검색 실패: {e}")
            raise
    
    def get_top_stocks(self, n_results: int = 10, criteria: str = '평가점수') -> List[SearchResult]:
        """상위 종목 조회"""
        try:
            if self.stock_data is None:
                self.load_search_index()
            
            top_data = self.stock_data.nlargest(n_results, criteria)
            
            results = []
            for _, row in top_data.iterrows():
                result = SearchResult(
                    종목명=row['종목명'],
                    종목코드=row['종목코드'],
                    현재가=row['현재가'],
                    시가총액=row['시가총액'],
                    매출성장률=row['매출성장률'],
                    순이익률=row['순이익률'],
                    부채비율=row['부채비율'],
                    PER=row['PER'],
                    PBR=row['PBR'],
                    평가점수=row['평가점수'],
                    종합평가=row['종합평가'],
                    유사도=1.0
                )
                results.append(result)
            
            logger.info(f"상위 종목 조회 완료: {len(results)}개")
            return results
            
        except Exception as e:
            logger.error(f"상위 종목 조회 실패: {e}")
            raise

def main():
    """메인 실행 함수"""
    # 검색 엔진 초기화
    search_engine = StockSearchEngine()
    
    try:
        # 검색 인덱스 로드 또는 구축
        search_engine.load_search_index()
        
        # 검색 예시
        print("=== 키워드 검색 ===")
        results = search_engine.search_stocks("성장성이 좋고 수익성이 높은 주식", n_results=5)
        for result in results:
            print(f"{result.종목명} ({result.종목코드}): {result.평가점수}점 - 유사도: {result.유사도:.3f}")
        
        print("\n=== 조건 검색 ===")
        criteria = {
            'min_score': 60,
            'max_per': 15,
            'max_pbr': 2.0,
            'min_revenue_growth': 5,
            'sort_by': '평가점수',
            'ascending': False,
            'limit': 5
        }
        results = search_engine.search_by_criteria(criteria)
        for result in results:
            print(f"{result.종목명} ({result.종목코드}): {result.평가점수}점")
        
        print("\n=== 상위 종목 ===")
        top_stocks = search_engine.get_top_stocks(n_results=5)
        for result in top_stocks:
            print(f"{result.종목명} ({result.종목코드}): {result.평가점수}점")
            
    except Exception as e:
        print(f"검색 엔진 실행 실패: {e}")

if __name__ == "__main__":
    main() 