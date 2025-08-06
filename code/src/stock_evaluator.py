#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
주식 평가 시스템
- docs/data_processing.ipynb 기반 구현
- 성장성, 수익성, 안정성, 가치평가 지표 종합 평가
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StockEvaluationResult:
    """주식 평가 결과"""
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
    평가이유: str
    성장성점수: int
    수익성점수: int
    안정성점수: int
    투자지표점수: int

class StockEvaluator:
    """주식 평가 클래스"""
    
    def __init__(self):
        self.evaluation_criteria = {
            '성장성': {
                '매출성장률_우수': 20,
                '매출성장률_양호': 10,
                '매출성장률_기준': 10
            },
            '수익성': {
                '순이익률_우수': 20,
                '순이익률_양호': 10,
                '순이익률_기준': 5
            },
            '안정성': {
                '부채비율_안정': 20,
                '부채비율_보통': 10,
                '부채비율_기준': 100
            },
            '투자지표': {
                'PER_적정': 20,
                'PER_저평가': 10,
                'PER_범위': (5, 15),
                'PBR_적정': 20,
                'PBR_저평가': 10,
                'PBR_범위': (0.5, 2.0)
            }
        }
    
    def evaluate_growth(self, fs_data: pd.DataFrame, stock_code: str) -> Tuple[int, Optional[float], str]:
        """성장성 평가"""
        try:
            revenue_data = fs_data[
                (fs_data['종목코드'] == stock_code) & 
                (fs_data['공시구분'] == 'y') & 
                (fs_data['계정'] == '매출액')
            ].sort_values('기준일', ascending=False)
            
            if len(revenue_data) < 2:
                return 0, None, "매출 데이터 부족"
            
            revenue_current = revenue_data['값'].iloc[0]
            revenue_previous = revenue_data['값'].iloc[1]
            
            if revenue_previous == 0:
                return 0, None, "전년 매출 데이터 없음"
            
            revenue_growth = (revenue_current - revenue_previous) / revenue_previous * 100
            
            if revenue_growth > self.evaluation_criteria['성장성']['매출성장률_우수']:
                score = 20
                reason = f"매출 성장률이 {revenue_growth:.2f}%로 우수함"
            elif revenue_growth > self.evaluation_criteria['성장성']['매출성장률_양호']:
                score = 10
                reason = f"매출 성장률이 {revenue_growth:.2f}%로 양호함"
            else:
                score = 0
                reason = f"매출 성장률이 {revenue_growth:.2f}%로 저조함"
            
            return score, revenue_growth, reason
            
        except Exception as e:
            logger.error(f"성장성 평가 오류 ({stock_code}): {e}")
            return 0, None, f"성장성 평가 오류: {e}"
    
    def evaluate_profitability(self, fs_data: pd.DataFrame, stock_code: str) -> Tuple[int, Optional[float], str]:
        """수익성 평가"""
        try:
            fs_filtered = fs_data[
                (fs_data['종목코드'] == stock_code) & 
                (fs_data['공시구분'] == 'y')
            ].sort_values('기준일', ascending=False)
            
            net_profit_data = fs_filtered[fs_filtered['계정'] == '당기순이익']
            revenue_data = fs_filtered[fs_filtered['계정'] == '매출액']
            
            if len(net_profit_data) == 0 or len(revenue_data) == 0:
                return 0, None, "재무 데이터 부족"
            
            net_profit = net_profit_data['값'].iloc[0]
            revenue = revenue_data['값'].iloc[0]
            
            if revenue == 0:
                return 0, None, "매출 데이터 없음"
            
            net_profit_margin = net_profit / revenue * 100
            
            if net_profit_margin > self.evaluation_criteria['수익성']['순이익률_우수']:
                score = 20
                reason = f"순이익률이 {net_profit_margin:.2f}%로 우수함"
            elif net_profit_margin > self.evaluation_criteria['수익성']['순이익률_양호']:
                score = 10
                reason = f"순이익률이 {net_profit_margin:.2f}%로 양호함"
            else:
                score = 0
                reason = f"순이익률이 {net_profit_margin:.2f}%로 개선 필요"
            
            return score, net_profit_margin, reason
            
        except Exception as e:
            logger.error(f"수익성 평가 오류 ({stock_code}): {e}")
            return 0, None, f"수익성 평가 오류: {e}"
    
    def evaluate_stability(self, fs_data: pd.DataFrame, stock_code: str) -> Tuple[int, Optional[float], str]:
        """재무 안정성 평가"""
        try:
            fs_filtered = fs_data[
                (fs_data['종목코드'] == stock_code) & 
                (fs_data['공시구분'] == 'y')
            ].sort_values('기준일', ascending=False)
            
            debt_data = fs_filtered[fs_filtered['계정'] == '부채']
            equity_data = fs_filtered[fs_filtered['계정'] == '자본']
            
            if len(debt_data) == 0 or len(equity_data) == 0:
                return 0, None, "재무 데이터 부족"
            
            debt = debt_data['값'].iloc[0]
            equity = equity_data['값'].iloc[0]
            
            if equity == 0:
                return 0, None, "자본 데이터 없음"
            
            debt_ratio = debt / equity * 100
            
            if debt_ratio < self.evaluation_criteria['안정성']['부채비율_안정']:
                score = 20
                reason = f"부채비율이 {debt_ratio:.2f}%로 안정적임"
            elif debt_ratio < self.evaluation_criteria['안정성']['부채비율_보통']:
                score = 10
                reason = f"부채비율이 {debt_ratio:.2f}%로 보통 수준"
            else:
                score = 0
                reason = f"부채비율이 {debt_ratio:.2f}%로 높음"
            
            return score, debt_ratio, reason
            
        except Exception as e:
            logger.error(f"안정성 평가 오류 ({stock_code}): {e}")
            return 0, None, f"안정성 평가 오류: {e}"
    
    def evaluate_investment_metrics(self, value_data: pd.DataFrame, stock_code: str) -> Tuple[int, Optional[float], Optional[float], str]:
        """투자 지표 평가"""
        try:
            stock_value = value_data[value_data['종목코드'] == stock_code]
            
            per_data = stock_value[stock_value['지표'] == 'PER']
            pbr_data = stock_value[stock_value['지표'] == 'PBR']
            
            per = float(per_data['값'].iloc[0]) if len(per_data) > 0 else None
            pbr = float(pbr_data['값'].iloc[0]) if len(pbr_data) > 0 else None
            
            score = 0
            reasons = []
            
            # PER 평가
            if per is not None:
                per_low, per_high = self.evaluation_criteria['투자지표']['PER_범위']
                if per_low < per < per_high:
                    score += 20
                    reasons.append(f"PER이 {per:.2f}로 적정 수준")
                elif per <= per_low:
                    score += 10
                    reasons.append(f"PER이 {per:.2f}로 저평가 가능성")
                else:
                    reasons.append(f"PER이 {per:.2f}로 고평가 가능성")
            else:
                reasons.append("PER 데이터 없음")
            
            # PBR 평가
            if pbr is not None:
                pbr_low, pbr_high = self.evaluation_criteria['투자지표']['PBR_범위']
                if pbr_low < pbr < pbr_high:
                    score += 20
                    reasons.append(f"PBR이 {pbr:.2f}로 적정 수준")
                elif pbr <= pbr_low:
                    score += 10
                    reasons.append(f"PBR이 {pbr:.2f}로 저평가 가능성")
                else:
                    reasons.append(f"PBR이 {pbr:.2f}로 고평가 가능성")
            else:
                reasons.append("PBR 데이터 없음")
            
            return score, per, pbr, '; '.join(reasons)
            
        except Exception as e:
            logger.error(f"투자 지표 평가 오류 ({stock_code}): {e}")
            return 0, None, None, f"투자 지표 평가 오류: {e}"
    
    def evaluate_stock(self, stock_code: str, fs_data: pd.DataFrame, 
                      price_data: pd.DataFrame, ticker_data: pd.DataFrame, 
                      value_data: pd.DataFrame) -> StockEvaluationResult:
        """개별 종목 평가"""
        try:
            # 종목 정보 추출
            stock_info = ticker_data[ticker_data['종목코드'] == stock_code]
            if len(stock_info) == 0:
                raise ValueError(f"종목 정보를 찾을 수 없습니다: {stock_code}")
            
            stock_info = stock_info.iloc[0]
            
            # 각 항목별 평가
            growth_score, revenue_growth, growth_reason = self.evaluate_growth(fs_data, stock_code)
            profit_score, net_profit_margin, profit_reason = self.evaluate_profitability(fs_data, stock_code)
            stability_score, debt_ratio, stability_reason = self.evaluate_stability(fs_data, stock_code)
            investment_score, per, pbr, investment_reason = self.evaluate_investment_metrics(value_data, stock_code)
            
            # 총점 계산
            total_score = growth_score + profit_score + stability_score + investment_score
            
            # 종합 평가
            if total_score >= 80:
                evaluation = "매우 좋음"
            elif total_score >= 60:
                evaluation = "좋음"
            elif total_score >= 40:
                evaluation = "보통"
            else:
                evaluation = "주의 필요"
            
            # 평가 이유 통합
            all_reasons = [r for r in [growth_reason, profit_reason, stability_reason, investment_reason] if r]
            evaluation_reason = '; '.join(all_reasons)
            
            return StockEvaluationResult(
                종목명=stock_info['종목명'],
                종목코드=stock_code,
                현재가=stock_info['종가'],
                시가총액=stock_info['시가총액'],
                매출성장률=revenue_growth,
                순이익률=net_profit_margin,
                부채비율=debt_ratio,
                PER=per,
                PBR=pbr,
                평가점수=total_score,
                종합평가=evaluation,
                평가이유=evaluation_reason,
                성장성점수=growth_score,
                수익성점수=profit_score,
                안정성점수=stability_score,
                투자지표점수=investment_score
            )
            
        except Exception as e:
            logger.error(f"종목 평가 실패 ({stock_code}): {e}")
            raise
    
    def evaluate_all_stocks(self, fs_data: pd.DataFrame, price_data: pd.DataFrame,
                          ticker_data: pd.DataFrame, value_data: pd.DataFrame) -> pd.DataFrame:
        """전체 종목 평가"""
        try:
            results = []
            stock_codes = ticker_data['종목코드'].unique()
            
            logger.info(f"전체 {len(stock_codes)}개 종목 평가 시작")
            
            for stock_code in stock_codes:
                try:
                    result = self.evaluate_stock(stock_code, fs_data, price_data, ticker_data, value_data)
                    results.append(result)
                    
                    if len(results) % 100 == 0:
                        logger.info(f"평가 진행률: {len(results)}/{len(stock_codes)}")
                        
                except Exception as e:
                    logger.error(f"종목 {stock_code} 평가 실패: {e}")
                    continue
            
            # 결과를 DataFrame으로 변환
            results_df = pd.DataFrame([vars(result) for result in results])
            
            # 평가점수가 0인 항목 제거
            results_df = results_df[results_df['평가점수'] != 0].reset_index(drop=True)
            
            logger.info(f"평가 완료: {len(results_df)}개 종목")
            
            return results_df
            
        except Exception as e:
            logger.error(f"전체 종목 평가 실패: {e}")
            raise

def main():
    """메인 실행 함수"""
    import os
    
    # 데이터 로드
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    
    fs_data = pd.read_csv(os.path.join(data_dir, 'kor_fs_20250721.csv'), dtype={'종목코드': str})
    fs_data['종목코드'] = fs_data['종목코드'].str.zfill(6)
    
    price_data = pd.read_csv(os.path.join(data_dir, 'kor_price_20250721.csv'), dtype={'종목코드': str})
    price_data['종목코드'] = price_data['종목코드'].str.zfill(6)
    
    ticker_data = pd.read_csv(os.path.join(data_dir, 'kor_ticker_20250721.csv'), dtype={'종목코드': str})
    ticker_data['종목코드'] = ticker_data['종목코드'].str.zfill(6)
    
    value_data = pd.read_csv(os.path.join(data_dir, 'kor_value_20250721.csv'), dtype={'종목코드': str})
    value_data['종목코드'] = value_data['종목코드'].str.zfill(6)
    
    # 평가 실행
    evaluator = StockEvaluator()
    results = evaluator.evaluate_all_stocks(fs_data, price_data, ticker_data, value_data)
    
    # 결과 저장
    output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'stock_evaluation_results.csv')
    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"평가 완료: {len(results)}개 종목")
    print(f"결과 저장: {output_file}")
    
    # 상위 10개 종목 출력
    top_stocks = results.nlargest(10, '평가점수')
    print("\n=== 상위 10개 종목 ===")
    for _, stock in top_stocks.iterrows():
        print(f"{stock['종목명']} ({stock['종목코드']}): {stock['평가점수']}점 - {stock['종합평가']}")

if __name__ == "__main__":
    main() 