#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.korean_stock_data_processor import KoreanStockDataProcessor, DataConfig
from src.stock_evaluator import StockEvaluator
from src.stock_search_engine import StockSearchEngine

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('korean_stock_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KoreanStockAnalysis:
    """국내주식 데이터 분석 통합 클래스"""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.data_processor = KoreanStockDataProcessor(config)
        self.evaluator = StockEvaluator()
        self.search_engine = StockSearchEngine()
        
        logger.info("KoreanStockAnalysis 초기화 완료")
    
    def collect_data(self) -> Dict[str, any]:
        """데이터 수집"""
        try:
            logger.info("데이터 수집 시작")
            results = self.data_processor.process_all_data()
            logger.info("데이터 수집 완료")
            return results
        except Exception as e:
            logger.error(f"데이터 수집 실패: {e}")
            raise
    
    def evaluate_stocks(self, data_results: Dict[str, any] = None) -> pd.DataFrame:
        """주식 평가"""
        try:
            logger.info("주식 평가 시작")
            
            if data_results is None:
                # 기존 데이터 로드
                data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
                fs_data = pd.read_csv(os.path.join(data_dir, 'kor_fs_20250721.csv'), dtype={'종목코드': str})
                price_data = pd.read_csv(os.path.join(data_dir, 'kor_price_20250721.csv'), dtype={'종목코드': str})
                ticker_data = pd.read_csv(os.path.join(data_dir, 'kor_ticker_20250721.csv'), dtype={'종목코드': str})
                value_data = pd.read_csv(os.path.join(data_dir, 'kor_value_20250721.csv'), dtype={'종목코드': str})
                
                # 데이터 전처리
                for df in [fs_data, price_data, ticker_data, value_data]:
                    df['종목코드'] = df['종목코드'].str.zfill(6)
            else:
                fs_data = data_results['financial_statements']
                price_data = data_results['price']
                ticker_data = data_results['ticker']
                value_data = data_results['valuation_metrics']
            
            # 평가 실행
            results = self.evaluator.evaluate_all_stocks(fs_data, price_data, ticker_data, value_data)
            
            # 결과 저장
            output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'stock_evaluation_results.csv')
            results.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"주식 평가 완료: {len(results)}개 종목")
            return results
            
        except Exception as e:
            logger.error(f"주식 평가 실패: {e}")
            raise
    
    def build_search_index(self, evaluation_results: pd.DataFrame = None) -> None:
        """검색 인덱스 구축"""
        try:
            logger.info("검색 인덱스 구축 시작")
            
            if evaluation_results is None:
                # 기존 평가 결과 로드
                eval_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'stock_evaluation_results.csv')
                evaluation_results = pd.read_csv(eval_file)
            
            self.search_engine.build_search_index(evaluation_results)
            logger.info("검색 인덱스 구축 완료")
            
        except Exception as e:
            logger.error(f"검색 인덱스 구축 실패: {e}")
            raise
    
    def search_stocks(self, query: str, n_results: int = 10, **kwargs) -> List:
        """주식 검색"""
        try:
            logger.info(f"주식 검색 시작: {query}")
            results = self.search_engine.search_stocks(query, n_results, **kwargs)
            logger.info(f"주식 검색 완료: {len(results)}개 결과")
            return results
        except Exception as e:
            logger.error(f"주식 검색 실패: {e}")
            raise
    
    def search_by_criteria(self, criteria: Dict[str, any]) -> List:
        """조건별 검색"""
        try:
            logger.info("조건별 검색 시작")
            results = self.search_engine.search_by_criteria(criteria)
            logger.info(f"조건별 검색 완료: {len(results)}개 결과")
            return results
        except Exception as e:
            logger.error(f"조건별 검색 실패: {e}")
            raise
    
    def get_top_stocks(self, n_results: int = 10, criteria: str = '평가점수') -> List:
        """상위 종목 조회"""
        try:
            logger.info(f"상위 종목 조회 시작: {criteria} 기준")
            results = self.search_engine.get_top_stocks(n_results, criteria)
            logger.info(f"상위 종목 조회 완료: {len(results)}개")
            return results
        except Exception as e:
            logger.error(f"상위 종목 조회 실패: {e}")
            raise
    
    def run_full_analysis(self) -> None:
        """전체 분석 실행"""
        try:
            logger.info("전체 분석 시작")
            
            # 1. 데이터 수집
            data_results = self.collect_data()
            
            # 2. 주식 평가
            evaluation_results = self.evaluate_stocks(data_results)
            
            # 3. 검색 인덱스 구축
            self.build_search_index(evaluation_results)
            
            # 4. 결과 요약 출력
            self.print_analysis_summary(evaluation_results)
            
            logger.info("전체 분석 완료")
            
        except Exception as e:
            logger.error(f"전체 분석 실패: {e}")
            raise
    
    def print_analysis_summary(self, evaluation_results: pd.DataFrame) -> None:
        """분석 결과 요약 출력"""
        try:
            print("\n" + "="*50)
            print("국내주식 데이터 분석 결과 요약")
            print("="*50)
            
            print(f"\n총 평가 종목 수: {len(evaluation_results)}개")
            
            # 평가 등급별 분포
            grade_counts = evaluation_results['종합평가'].value_counts()
            print("\n평가 등급별 분포:")
            for grade, count in grade_counts.items():
                print(f"  {grade}: {count}개 ({count/len(evaluation_results)*100:.1f}%)")
            
            # 상위 10개 종목
            top_stocks = evaluation_results.nlargest(10, '평가점수')
            print("\n상위 10개 종목:")
            for _, stock in top_stocks.iterrows():
                print(f"  {stock['종목명']} ({stock['종목코드']}): {stock['평가점수']}점 - {stock['종합평가']}")
            
            # 평균 지표
            print(f"\n평균 지표:")
            print(f"  평균 평가점수: {evaluation_results['평가점수'].mean():.1f}점")
            print(f"  평균 PER: {evaluation_results['PER'].mean():.2f}")
            print(f"  평균 PBR: {evaluation_results['PBR'].mean():.2f}")
            print(f"  평균 매출성장률: {evaluation_results['매출성장률'].mean():.2f}%")
            print(f"  평균 순이익률: {evaluation_results['순이익률'].mean():.2f}%")
            
            print("\n" + "="*50)
            
        except Exception as e:
            logger.error(f"분석 결과 요약 출력 실패: {e}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='국내주식 데이터 분석')
    parser.add_argument('--mode', choices=['collect', 'evaluate', 'search', 'full'], 
                       default='full', help='실행 모드')
    parser.add_argument('--test', action='store_true', help='테스트 모드')
    parser.add_argument('--query', type=str, help='검색 쿼리')
    parser.add_argument('--limit', type=int, default=10, help='결과 개수 제한')
    
    args = parser.parse_args()
    
    # 설정
    config = DataConfig(test_mode=args.test)
    analysis = KoreanStockAnalysis(config)
    
    try:
        if args.mode == 'collect':
            # 데이터 수집만
            analysis.collect_data()
            print("데이터 수집 완료")
            
        elif args.mode == 'evaluate':
            # 평가만
            results = analysis.evaluate_stocks()
            print(f"주식 평가 완료: {len(results)}개 종목")
            
        elif args.mode == 'search':
            # 검색만
            if not args.query:
                print("검색 쿼리를 입력해주세요 (--query 옵션)")
                return
            
            results = analysis.search_stocks(args.query, args.limit)
            print(f"\n검색 결과 ({len(results)}개):")
            for result in results:
                print(f"  {result.종목명} ({result.종목코드}): {result.평가점수}점 - 유사도: {result.유사도:.3f}")
            
        elif args.mode == 'full':
            # 전체 분석
            analysis.run_full_analysis()
            
        # 검색 인덱스 구축 (평가 후)
        if args.mode in ['evaluate', 'full']:
            analysis.build_search_index()
            print("검색 인덱스 구축 완료")
            
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main() 