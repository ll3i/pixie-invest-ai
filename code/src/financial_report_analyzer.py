#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
금융 보고서 및 데이터 자동 수집/분석 시스템
- 재무제표 자동 수집 및 분석
- 실적 보고서 핵심 지표 추출
- 위험 신호 감지 알고리즘
- 투자 인사이트 생성
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import json
import pickle
import yfinance as yf
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """위험 수준 정의"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskAlert:
    """위험 경고 데이터 구조"""
    ticker: str
    stock_name: str
    risk_level: RiskLevel
    alert_type: str
    title: str
    description: str
    analysis: str
    indicators: Dict[str, Any]
    detected_at: datetime
    recommendations: List[str]

@dataclass
class FinancialInsight:
    """금융 인사이트 데이터 구조"""
    ticker: str
    stock_name: str
    insight_type: str
    title: str
    summary: str
    key_metrics: Dict[str, Any]
    trend_analysis: str
    created_at: datetime

class FinancialReportAnalyzer:
    """금융 보고서 자동 수집 및 분석 시스템"""
    
    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.raw_data_path = os.path.join(self.data_path, 'raw')
        self.processed_data_path = os.path.join(self.data_path, 'processed')
        
        # 위험 감지 임계값 설정
        self.risk_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_spike': 2.0,  # 평균 거래량의 2배
            'price_drop': -0.05,  # 5% 하락
            'price_spike': 0.05,  # 5% 상승
            'per_high': 30,  # PER 30 이상
            'debt_ratio_high': 2.0,  # 부채비율 200% 이상
            'roe_low': 0.05,  # ROE 5% 이하
            'revenue_decline': -0.1,  # 매출 10% 감소
        }
        
        # 평가 데이터 로드
        self.evaluation_data = self._load_evaluation_data()
        self.stock_data_pkl = self._load_stock_data_pkl()
        
        logger.info("FinancialReportAnalyzer 초기화 완료")
    
    def analyze_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """
        재무제표 데이터 수집 및 분석
        
        Args:
            ticker: 종목 코드
            
        Returns:
            분석된 재무 데이터
        """
        try:
            # 최신 재무제표 데이터 로드
            fs_data = self._load_financial_statements(ticker)
            if not fs_data:
                return self._analyze_with_yfinance(ticker)
            
            # 주요 재무 지표 계산
            metrics = self._calculate_financial_metrics(fs_data)
            
            # 재무 건전성 평가
            health_score = self._evaluate_financial_health(metrics)
            
            # 성장성 분석
            growth_analysis = self._analyze_growth_trends(fs_data)
            
            # 수익성 분석
            profitability = self._analyze_profitability(metrics)
            
            return {
                'ticker': ticker,
                'latest_data': fs_data.get('latest', {}),
                'key_metrics': metrics,
                'health_score': health_score,
                'growth_analysis': growth_analysis,
                'profitability': profitability,
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"재무제표 분석 실패 ({ticker}): {e}")
            return {}
    
    def detect_risk_signals(self, ticker: str, stock_data: Optional[Dict] = None) -> List[RiskAlert]:
        """
        위험 신호 감지
        
        Args:
            ticker: 종목 코드
            stock_data: 주가 데이터 (선택적)
            
        Returns:
            감지된 위험 신호 리스트
        """
        alerts = []
        
        try:
            # 주가 데이터가 없으면 로드
            if not stock_data:
                stock_data = self._load_stock_data(ticker)
            
            if not stock_data:
                return alerts
            
            stock_name = stock_data.get('name', ticker)
            
            # 1. RSI 과매도/과매수 체크
            rsi_alert = self._check_rsi_signal(ticker, stock_name, stock_data)
            if rsi_alert:
                alerts.append(rsi_alert)
            
            # 2. 거래량 급증 체크
            volume_alert = self._check_volume_spike(ticker, stock_name, stock_data)
            if volume_alert:
                alerts.append(volume_alert)
            
            # 3. 급락/급등 체크
            price_alert = self._check_price_movement(ticker, stock_name, stock_data)
            if price_alert:
                alerts.append(price_alert)
            
            # 4. 재무 위험 체크
            financial_alerts = self._check_financial_risks(ticker, stock_name)
            alerts.extend(financial_alerts)
            
            # 5. 밸류에이션 위험 체크
            valuation_alert = self._check_valuation_risk(ticker, stock_name, stock_data)
            if valuation_alert:
                alerts.append(valuation_alert)
            
        except Exception as e:
            logger.error(f"위험 신호 감지 실패 ({ticker}): {e}")
        
        return alerts
    
    def generate_insights(self, ticker: str) -> List[FinancialInsight]:
        """
        핵심 투자 인사이트 생성
        
        Args:
            ticker: 종목 코드
            
        Returns:
            생성된 인사이트 리스트
        """
        insights = []
        
        try:
            # 재무제표 분석
            financial_analysis = self.analyze_financial_statements(ticker)
            stock_data = self._load_stock_data(ticker)
            stock_name = stock_data.get('name', ticker) if stock_data else ticker
            
            # 1. 성장성 인사이트
            growth_insight = self._generate_growth_insight(
                ticker, stock_name, financial_analysis
            )
            if growth_insight:
                insights.append(growth_insight)
            
            # 2. 수익성 인사이트
            profitability_insight = self._generate_profitability_insight(
                ticker, stock_name, financial_analysis
            )
            if profitability_insight:
                insights.append(profitability_insight)
            
            # 3. 밸류에이션 인사이트
            valuation_insight = self._generate_valuation_insight(
                ticker, stock_name, stock_data, financial_analysis
            )
            if valuation_insight:
                insights.append(valuation_insight)
            
            # 4. 기술적 분석 인사이트
            technical_insight = self._generate_technical_insight(
                ticker, stock_name, stock_data
            )
            if technical_insight:
                insights.append(technical_insight)
            
        except Exception as e:
            logger.error(f"인사이트 생성 실패 ({ticker}): {e}")
        
        return insights
    
    def _load_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """재무제표 데이터 로드"""
        try:
            # 최신 재무제표 파일 찾기
            fs_files = [f for f in os.listdir(self.raw_data_path) 
                       if f.startswith('kor_fs_') and f.endswith('.csv')]
            
            if not fs_files:
                return {}
            
            # 가장 최신 파일 선택
            latest_file = sorted(fs_files)[-1]
            df = pd.read_csv(os.path.join(self.raw_data_path, latest_file))
            
            # 해당 종목 데이터 추출
            ticker_data = df[df['종목코드'] == ticker]
            if ticker_data.empty:
                return {}
            
            return {
                'latest': ticker_data.iloc[0].to_dict(),
                'historical': ticker_data.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"재무제표 로드 실패 ({ticker}): {e}")
            return {}
    
    def _load_stock_data(self, ticker: str) -> Dict[str, Any]:
        """주가 데이터 로드"""
        try:
            # 최신 주가 파일 찾기
            price_files = [f for f in os.listdir(self.raw_data_path) 
                          if f.startswith('kor_price_') and f.endswith('.csv')]
            
            if not price_files:
                return {}
            
            # 가장 최신 파일 선택
            latest_file = sorted(price_files)[-1]
            df = pd.read_csv(os.path.join(self.raw_data_path, latest_file))
            
            # 해당 종목 데이터 추출
            ticker_data = df[df['종목코드'] == ticker]
            if ticker_data.empty:
                return {}
            
            # 과거 데이터도 로드하여 추세 분석
            historical_data = []
            for price_file in sorted(price_files)[-20:]:  # 최근 20일
                try:
                    hist_df = pd.read_csv(os.path.join(self.raw_data_path, price_file))
                    hist_ticker = hist_df[hist_df['종목코드'] == ticker]
                    if not hist_ticker.empty:
                        historical_data.append(hist_ticker.iloc[0].to_dict())
                except:
                    continue
            
            latest = ticker_data.iloc[0].to_dict()
            return {
                'ticker': ticker,
                'name': latest.get('종목명', ticker),
                'latest': latest,
                'historical': historical_data
            }
            
        except Exception as e:
            logger.error(f"주가 데이터 로드 실패 ({ticker}): {e}")
            return {}
    
    def _calculate_financial_metrics(self, fs_data: Dict[str, Any]) -> Dict[str, float]:
        """주요 재무 지표 계산"""
        latest = fs_data.get('latest', {})
        
        metrics = {}
        
        try:
            # ROE (자기자본이익률)
            if latest.get('당기순이익') and latest.get('자본총계'):
                metrics['roe'] = latest['당기순이익'] / latest['자본총계']
            
            # ROA (총자산이익률)
            if latest.get('당기순이익') and latest.get('자산총계'):
                metrics['roa'] = latest['당기순이익'] / latest['자산총계']
            
            # 부채비율
            if latest.get('부채총계') and latest.get('자본총계'):
                metrics['debt_ratio'] = latest['부채총계'] / latest['자본총계']
            
            # 유동비율
            if latest.get('유동자산') and latest.get('유동부채'):
                metrics['current_ratio'] = latest['유동자산'] / latest['유동부채']
            
            # 매출총이익률
            if latest.get('매출총이익') and latest.get('매출액'):
                metrics['gross_margin'] = latest['매출총이익'] / latest['매출액']
            
            # 영업이익률
            if latest.get('영업이익') and latest.get('매출액'):
                metrics['operating_margin'] = latest['영업이익'] / latest['매출액']
            
        except Exception as e:
            logger.error(f"재무 지표 계산 실패: {e}")
        
        return metrics
    
    def _evaluate_financial_health(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """재무 건전성 평가"""
        score = 100
        issues = []
        
        # ROE 평가
        if 'roe' in metrics:
            if metrics['roe'] < 0.05:
                score -= 20
                issues.append("ROE가 5% 미만으로 수익성이 낮음")
            elif metrics['roe'] > 0.15:
                score += 10
        
        # 부채비율 평가
        if 'debt_ratio' in metrics:
            if metrics['debt_ratio'] > 2.0:
                score -= 30
                issues.append("부채비율이 200% 초과로 재무 리스크 높음")
            elif metrics['debt_ratio'] < 0.5:
                score += 10
        
        # 유동비율 평가
        if 'current_ratio' in metrics:
            if metrics['current_ratio'] < 1.0:
                score -= 20
                issues.append("유동비율이 100% 미만으로 단기 지급능력 우려")
            elif metrics['current_ratio'] > 2.0:
                score += 5
        
        # 등급 판정
        if score >= 80:
            grade = "우수"
        elif score >= 60:
            grade = "양호"
        elif score >= 40:
            grade = "보통"
        else:
            grade = "주의"
        
        return {
            'score': max(0, min(100, score)),
            'grade': grade,
            'issues': issues
        }
    
    def _analyze_growth_trends(self, fs_data: Dict[str, Any]) -> Dict[str, Any]:
        """성장성 분석"""
        historical = fs_data.get('historical', [])
        
        if len(historical) < 2:
            return {'status': '데이터 부족'}
        
        try:
            # 최근 데이터와 이전 데이터 비교
            recent = historical[-1]
            previous = historical[-2]
            
            growth_metrics = {}
            
            # 매출 성장률
            if recent.get('매출액') and previous.get('매출액'):
                revenue_growth = (recent['매출액'] - previous['매출액']) / previous['매출액']
                growth_metrics['revenue_growth'] = revenue_growth
            
            # 영업이익 성장률
            if recent.get('영업이익') and previous.get('영업이익'):
                operating_growth = (recent['영업이익'] - previous['영업이익']) / abs(previous['영업이익'])
                growth_metrics['operating_profit_growth'] = operating_growth
            
            # 순이익 성장률
            if recent.get('당기순이익') and previous.get('당기순이익'):
                net_income_growth = (recent['당기순이익'] - previous['당기순이익']) / abs(previous['당기순이익'])
                growth_metrics['net_income_growth'] = net_income_growth
            
            # 성장성 평가
            growth_score = 0
            if growth_metrics.get('revenue_growth', 0) > 0.1:
                growth_score += 30
            if growth_metrics.get('operating_profit_growth', 0) > 0.15:
                growth_score += 35
            if growth_metrics.get('net_income_growth', 0) > 0.2:
                growth_score += 35
            
            return {
                'metrics': growth_metrics,
                'score': growth_score,
                'trend': self._classify_growth_trend(growth_metrics)
            }
            
        except Exception as e:
            logger.error(f"성장성 분석 실패: {e}")
            return {'status': '분석 실패'}
    
    def _analyze_profitability(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """수익성 분석"""
        profitability_score = 0
        analysis = []
        
        # ROE 분석
        if 'roe' in metrics:
            roe = metrics['roe']
            if roe > 0.15:
                profitability_score += 40
                analysis.append("ROE가 15% 이상으로 우수한 수익성")
            elif roe > 0.1:
                profitability_score += 25
                analysis.append("ROE가 10% 이상으로 양호한 수익성")
            else:
                analysis.append(f"ROE가 {roe*100:.1f}%로 개선 필요")
        
        # 영업이익률 분석
        if 'operating_margin' in metrics:
            margin = metrics['operating_margin']
            if margin > 0.15:
                profitability_score += 30
                analysis.append("영업이익률이 15% 이상으로 우수")
            elif margin > 0.08:
                profitability_score += 20
                analysis.append("영업이익률이 양호한 수준")
            else:
                analysis.append(f"영업이익률이 {margin*100:.1f}%로 낮음")
        
        # 매출총이익률 분석
        if 'gross_margin' in metrics:
            margin = metrics['gross_margin']
            if margin > 0.3:
                profitability_score += 30
                analysis.append("매출총이익률이 30% 이상으로 경쟁력 있음")
        
        return {
            'score': min(100, profitability_score),
            'analysis': analysis
        }
    
    def _check_rsi_signal(self, ticker: str, stock_name: str, stock_data: Dict) -> Optional[RiskAlert]:
        """RSI 기반 위험 신호 체크"""
        try:
            historical = stock_data.get('historical', [])
            if len(historical) < 14:
                return None
            
            # 간단한 RSI 계산 (14일 기준)
            prices = [h.get('종가', 0) for h in historical[-14:]]
            if not all(prices):
                return None
            
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # RSI 과매도/과매수 체크
            if rsi <= self.risk_thresholds['rsi_oversold']:
                return RiskAlert(
                    ticker=ticker,
                    stock_name=stock_name,
                    risk_level=RiskLevel.MEDIUM,
                    alert_type="technical",
                    title=f"{stock_name} RSI 과매도 신호",
                    description=f"RSI가 {rsi:.0f}로 과매도 구간에 진입했습니다.",
                    analysis="단기적으로 반등 가능성이 있으나, 추가 하락 위험도 존재합니다.",
                    indicators={'rsi': rsi},
                    detected_at=datetime.now(),
                    recommendations=[
                        "추가 하락에 대비한 리스크 관리 필요",
                        "반등 시점을 노린 분할 매수 고려",
                        "기업 펀더멘털 재확인 권장"
                    ]
                )
            elif rsi >= self.risk_thresholds['rsi_overbought']:
                return RiskAlert(
                    ticker=ticker,
                    stock_name=stock_name,
                    risk_level=RiskLevel.MEDIUM,
                    alert_type="technical",
                    title=f"{stock_name} RSI 과매수 신호",
                    description=f"RSI가 {rsi:.0f}로 과매수 구간에 진입했습니다.",
                    analysis="단기 조정 가능성이 높아 주의가 필요합니다.",
                    indicators={'rsi': rsi},
                    detected_at=datetime.now(),
                    recommendations=[
                        "일부 차익 실현 고려",
                        "추가 매수는 조정 후 검토",
                        "손절 라인 설정 권장"
                    ]
                )
            
        except Exception as e:
            logger.error(f"RSI 체크 실패: {e}")
        
        return None
    
    def _check_volume_spike(self, ticker: str, stock_name: str, stock_data: Dict) -> Optional[RiskAlert]:
        """거래량 급증 체크"""
        try:
            latest = stock_data.get('latest', {})
            historical = stock_data.get('historical', [])
            
            if len(historical) < 20:
                return None
            
            current_volume = latest.get('거래량', 0)
            # 20일 평균 거래량
            avg_volume = sum(h.get('거래량', 0) for h in historical[-20:]) / 20
            
            if avg_volume > 0 and current_volume / avg_volume >= self.risk_thresholds['volume_spike']:
                volume_ratio = current_volume / avg_volume
                
                return RiskAlert(
                    ticker=ticker,
                    stock_name=stock_name,
                    risk_level=RiskLevel.HIGH,
                    alert_type="volume",
                    title=f"{stock_name} 거래량 급증",
                    description=f"거래량이 평균 대비 {volume_ratio:.1f}배 급증했습니다.",
                    analysis="대량 매매 주체의 움직임이 감지되었습니다. 주가 변동성이 확대될 수 있습니다.",
                    indicators={
                        'current_volume': current_volume,
                        'avg_volume': avg_volume,
                        'volume_ratio': volume_ratio
                    },
                    detected_at=datetime.now(),
                    recommendations=[
                        "거래량 급증 원인 파악 필요",
                        "뉴스 및 공시 확인 권장",
                        "변동성 확대에 대비한 리스크 관리"
                    ]
                )
            
        except Exception as e:
            logger.error(f"거래량 체크 실패: {e}")
        
        return None
    
    def _check_price_movement(self, ticker: str, stock_name: str, stock_data: Dict) -> Optional[RiskAlert]:
        """급락/급등 체크"""
        try:
            latest = stock_data.get('latest', {})
            
            # 등락률 확인
            change_rate = latest.get('등락률', 0)
            if isinstance(change_rate, str):
                change_rate = float(change_rate.replace('%', '')) / 100
            
            # 급락 체크
            if change_rate <= self.risk_thresholds['price_drop']:
                return RiskAlert(
                    ticker=ticker,
                    stock_name=stock_name,
                    risk_level=RiskLevel.HIGH,
                    alert_type="price",
                    title=f"{stock_name} 주가 급락",
                    description=f"주가가 {abs(change_rate)*100:.1f}% 하락했습니다.",
                    analysis="단기간 큰 폭의 하락이 발생했습니다. 투자 심리 악화 및 추가 하락 위험이 있습니다.",
                    indicators={
                        'change_rate': change_rate,
                        'current_price': latest.get('종가', 0)
                    },
                    detected_at=datetime.now(),
                    recommendations=[
                        "하락 원인 분석 필요",
                        "손절 라인 재검토",
                        "추가 하락 시 대응 전략 수립"
                    ]
                )
            
            # 급등 체크
            elif change_rate >= self.risk_thresholds['price_spike']:
                return RiskAlert(
                    ticker=ticker,
                    stock_name=stock_name,
                    risk_level=RiskLevel.MEDIUM,
                    alert_type="price",
                    title=f"{stock_name} 주가 급등",
                    description=f"주가가 {change_rate*100:.1f}% 상승했습니다.",
                    analysis="단기 급등으로 조정 가능성이 있습니다. 과열 여부 점검이 필요합니다.",
                    indicators={
                        'change_rate': change_rate,
                        'current_price': latest.get('종가', 0)
                    },
                    detected_at=datetime.now(),
                    recommendations=[
                        "차익 실현 시점 검토",
                        "급등 원인 확인",
                        "조정 시 재진입 전략 수립"
                    ]
                )
            
        except Exception as e:
            logger.error(f"가격 움직임 체크 실패: {e}")
        
        return None
    
    def _check_financial_risks(self, ticker: str, stock_name: str) -> List[RiskAlert]:
        """재무적 위험 체크"""
        alerts = []
        
        try:
            fs_data = self._load_financial_statements(ticker)
            if not fs_data:
                return alerts
            
            metrics = self._calculate_financial_metrics(fs_data)
            
            # 부채비율 체크
            if 'debt_ratio' in metrics and metrics['debt_ratio'] >= self.risk_thresholds['debt_ratio_high']:
                alerts.append(RiskAlert(
                    ticker=ticker,
                    stock_name=stock_name,
                    risk_level=RiskLevel.HIGH,
                    alert_type="financial",
                    title=f"{stock_name} 높은 부채비율",
                    description=f"부채비율이 {metrics['debt_ratio']*100:.0f}%로 매우 높습니다.",
                    analysis="재무 안정성이 우려되는 수준입니다. 금리 상승 시 위험이 가중될 수 있습니다.",
                    indicators={'debt_ratio': metrics['debt_ratio']},
                    detected_at=datetime.now(),
                    recommendations=[
                        "재무구조 개선 계획 확인",
                        "현금흐름 안정성 점검",
                        "업종 평균 대비 비교 분석"
                    ]
                ))
            
            # ROE 체크
            if 'roe' in metrics and metrics['roe'] <= self.risk_thresholds['roe_low']:
                alerts.append(RiskAlert(
                    ticker=ticker,
                    stock_name=stock_name,
                    risk_level=RiskLevel.MEDIUM,
                    alert_type="financial",
                    title=f"{stock_name} 낮은 수익성",
                    description=f"ROE가 {metrics['roe']*100:.1f}%로 매우 낮습니다.",
                    analysis="자본 대비 수익성이 부진합니다. 경영 효율성 개선이 필요합니다.",
                    indicators={'roe': metrics['roe']},
                    detected_at=datetime.now(),
                    recommendations=[
                        "수익성 개선 전략 확인",
                        "동종업계 대비 분석",
                        "장기 투자 적합성 재검토"
                    ]
                ))
            
        except Exception as e:
            logger.error(f"재무 위험 체크 실패: {e}")
        
        return alerts
    
    def _check_valuation_risk(self, ticker: str, stock_name: str, stock_data: Dict) -> Optional[RiskAlert]:
        """밸류에이션 위험 체크"""
        try:
            # 밸류에이션 데이터 로드
            value_files = [f for f in os.listdir(self.raw_data_path) 
                          if f.startswith('kor_value_') and f.endswith('.csv')]
            
            if not value_files:
                return None
            
            latest_file = sorted(value_files)[-1]
            df = pd.read_csv(os.path.join(self.raw_data_path, latest_file))
            
            ticker_data = df[df['종목코드'] == ticker]
            if ticker_data.empty:
                return None
            
            per = ticker_data.iloc[0].get('PER', 0)
            
            if per >= self.risk_thresholds['per_high']:
                return RiskAlert(
                    ticker=ticker,
                    stock_name=stock_name,
                    risk_level=RiskLevel.MEDIUM,
                    alert_type="valuation",
                    title=f"{stock_name} 높은 밸류에이션",
                    description=f"PER이 {per:.1f}배로 높은 수준입니다.",
                    analysis="현재 주가가 이익 대비 고평가 되어 있을 수 있습니다.",
                    indicators={'per': per},
                    detected_at=datetime.now(),
                    recommendations=[
                        "성장성 대비 밸류에이션 적정성 검토",
                        "업종 평균 PER과 비교",
                        "조정 시 매수 기회 활용"
                    ]
                )
            
        except Exception as e:
            logger.error(f"밸류에이션 체크 실패: {e}")
        
        return None
    
    def _classify_growth_trend(self, metrics: Dict[str, float]) -> str:
        """성장 추세 분류"""
        revenue_growth = metrics.get('revenue_growth', 0)
        profit_growth = metrics.get('operating_profit_growth', 0)
        
        if revenue_growth > 0.2 and profit_growth > 0.2:
            return "고성장"
        elif revenue_growth > 0.1 and profit_growth > 0.1:
            return "안정성장"
        elif revenue_growth > 0 and profit_growth > 0:
            return "저성장"
        elif revenue_growth < -0.1 or profit_growth < -0.1:
            return "역성장"
        else:
            return "정체"
    
    def _generate_growth_insight(self, ticker: str, stock_name: str, 
                                financial_analysis: Dict) -> Optional[FinancialInsight]:
        """성장성 인사이트 생성"""
        growth = financial_analysis.get('growth_analysis', {})
        metrics = growth.get('metrics', {})
        
        if not metrics:
            return None
        
        # 인사이트 내용 구성
        revenue_growth = metrics.get('revenue_growth', 0)
        profit_growth = metrics.get('operating_profit_growth', 0)
        
        if revenue_growth > 0.15:
            title = f"{stock_name} 높은 매출 성장세 지속"
            summary = f"매출이 전년 대비 {revenue_growth*100:.1f}% 성장하며 견조한 성장세를 보이고 있습니다."
        elif revenue_growth < -0.1:
            title = f"{stock_name} 매출 성장 둔화 주의"
            summary = f"매출이 전년 대비 {abs(revenue_growth)*100:.1f}% 감소하여 성장 동력 회복이 필요합니다."
        else:
            return None
        
        trend_analysis = self._analyze_growth_trend_detail(metrics)
        
        return FinancialInsight(
            ticker=ticker,
            stock_name=stock_name,
            insight_type="growth",
            title=title,
            summary=summary,
            key_metrics={
                '매출성장률': f"{revenue_growth*100:.1f}%",
                '영업이익성장률': f"{profit_growth*100:.1f}%"
            },
            trend_analysis=trend_analysis,
            created_at=datetime.now()
        )
    
    def _generate_profitability_insight(self, ticker: str, stock_name: str,
                                      financial_analysis: Dict) -> Optional[FinancialInsight]:
        """수익성 인사이트 생성"""
        metrics = financial_analysis.get('key_metrics', {})
        profitability = financial_analysis.get('profitability', {})
        
        if not metrics:
            return None
        
        roe = metrics.get('roe', 0)
        operating_margin = metrics.get('operating_margin', 0)
        
        if roe > 0.15:
            title = f"{stock_name} 우수한 수익성 유지"
            summary = f"ROE {roe*100:.1f}%로 업계 평균을 상회하는 수익성을 보이고 있습니다."
            trend_analysis = "높은 자본 효율성을 바탕으로 주주가치 창출이 지속되고 있습니다."
        elif operating_margin > 0.15:
            title = f"{stock_name} 견조한 영업 수익성"
            summary = f"영업이익률 {operating_margin*100:.1f}%로 안정적인 수익구조를 유지하고 있습니다."
            trend_analysis = "원가 관리와 운영 효율성이 우수한 수준입니다."
        else:
            return None
        
        return FinancialInsight(
            ticker=ticker,
            stock_name=stock_name,
            insight_type="profitability",
            title=title,
            summary=summary,
            key_metrics={
                'ROE': f"{roe*100:.1f}%",
                '영업이익률': f"{operating_margin*100:.1f}%"
            },
            trend_analysis=trend_analysis,
            created_at=datetime.now()
        )
    
    def _generate_valuation_insight(self, ticker: str, stock_name: str,
                                  stock_data: Dict, financial_analysis: Dict) -> Optional[FinancialInsight]:
        """밸류에이션 인사이트 생성"""
        try:
            # 밸류에이션 데이터 로드
            value_files = [f for f in os.listdir(self.raw_data_path) 
                          if f.startswith('kor_value_') and f.endswith('.csv')]
            
            if not value_files:
                return None
            
            latest_file = sorted(value_files)[-1]
            df = pd.read_csv(os.path.join(self.raw_data_path, latest_file))
            
            ticker_data = df[df['종목코드'] == ticker]
            if ticker_data.empty:
                return None
            
            per = ticker_data.iloc[0].get('PER', 0)
            pbr = ticker_data.iloc[0].get('PBR', 0)
            
            if per > 0 and per < 15:
                title = f"{stock_name} 저평가 매력 부각"
                summary = f"PER {per:.1f}배로 업종 평균 대비 저평가 구간에 있습니다."
                trend_analysis = "실적 대비 주가가 저평가되어 있어 투자 매력이 높습니다."
            elif pbr > 0 and pbr < 1:
                title = f"{stock_name} 자산가치 대비 저평가"
                summary = f"PBR {pbr:.2f}배로 청산가치 이하에서 거래되고 있습니다."
                trend_analysis = "자산가치 대비 시장 평가가 낮아 안전마진이 존재합니다."
            else:
                return None
            
            return FinancialInsight(
                ticker=ticker,
                stock_name=stock_name,
                insight_type="valuation",
                title=title,
                summary=summary,
                key_metrics={
                    'PER': f"{per:.1f}배",
                    'PBR': f"{pbr:.2f}배"
                },
                trend_analysis=trend_analysis,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"밸류에이션 인사이트 생성 실패: {e}")
            return None
    
    def _generate_technical_insight(self, ticker: str, stock_name: str,
                                  stock_data: Dict) -> Optional[FinancialInsight]:
        """기술적 분석 인사이트 생성"""
        historical = stock_data.get('historical', [])
        
        if len(historical) < 20:
            return None
        
        # 20일 이동평균선 계산
        prices = [h.get('종가', 0) for h in historical[-20:]]
        ma20 = sum(prices) / len(prices) if prices else 0
        
        current_price = stock_data.get('latest', {}).get('종가', 0)
        
        if current_price > ma20 * 1.05:
            title = f"{stock_name} 단기 상승 추세 강화"
            summary = f"주가가 20일 이동평균선을 {((current_price/ma20-1)*100):.1f}% 상회하며 상승세를 보이고 있습니다."
            trend_analysis = "단기 모멘텀이 강화되고 있으나 과열 여부 점검이 필요합니다."
        elif current_price < ma20 * 0.95:
            title = f"{stock_name} 단기 조정 국면 진입"
            summary = f"주가가 20일 이동평균선을 {((1-current_price/ma20)*100):.1f}% 하회하며 조정을 받고 있습니다."
            trend_analysis = "단기 조정이 진행 중이며 지지선 확인이 필요합니다."
        else:
            return None
        
        return FinancialInsight(
            ticker=ticker,
            stock_name=stock_name,
            insight_type="technical",
            title=title,
            summary=summary,
            key_metrics={
                '현재가': f"{current_price:,}원",
                '20일 이동평균': f"{ma20:,.0f}원"
            },
            trend_analysis=trend_analysis,
            created_at=datetime.now()
        )
    
    def _analyze_growth_trend_detail(self, metrics: Dict[str, float]) -> str:
        """성장 추세 상세 분석"""
        revenue_growth = metrics.get('revenue_growth', 0)
        profit_growth = metrics.get('operating_profit_growth', 0)
        
        if revenue_growth > profit_growth and profit_growth > 0:
            return "매출 성장이 이익 성장을 상회하고 있어 향후 이익률 개선 여지가 있습니다."
        elif profit_growth > revenue_growth and revenue_growth > 0:
            return "이익 성장이 매출 성장을 상회하며 수익성 개선이 진행되고 있습니다."
        elif revenue_growth > 0 and profit_growth < 0:
            return "매출은 성장하나 이익이 감소하여 수익성 관리가 필요합니다."
        else:
            return "성장성과 수익성 모두 개선이 필요한 상황입니다."
    
    def _analyze_with_yfinance(self, ticker: str) -> Dict[str, Any]:
        """yfinance를 통한 대체 분석 (한국 주식 미지원 시)"""
        logger.info(f"로컬 데이터 없음, yfinance 시도: {ticker}")
        return {
            'ticker': ticker,
            'status': 'no_local_data',
            'message': '재무제표 데이터가 없습니다.'
        }
    
    def _load_evaluation_data(self) -> pd.DataFrame:
        """평가 데이터 로드"""
        try:
            eval_file = os.path.join(self.processed_data_path, 'stock_evaluation_results.csv')
            if os.path.exists(eval_file):
                return pd.read_csv(eval_file, encoding='utf-8-sig')
            else:
                logger.warning("평가 데이터 파일이 없습니다.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"평가 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def _load_stock_data_pkl(self) -> Dict[str, Any]:
        """pickle 파일에서 주식 데이터 로드"""
        try:
            pkl_file = os.path.join(self.processed_data_path, 'stock_data.pkl')
            if os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as f:
                    return pickle.load(f)
            else:
                logger.warning("stock_data.pkl 파일이 없습니다.")
                return {}
        except Exception as e:
            logger.error(f"pickle 데이터 로드 실패: {e}")
            return {}
    
    def get_evaluation_insights(self, ticker: str = None) -> List[Dict[str, Any]]:
        """평가 데이터 기반 인사이트 생성"""
        insights = []
        
        try:
            if self.evaluation_data.empty:
                return insights
            
            # 특정 종목 또는 상위 종목들
            if ticker:
                stock_data = self.evaluation_data[self.evaluation_data['종목코드'] == ticker]
            else:
                # 평가점수 상위 10개 종목
                stock_data = self.evaluation_data.nlargest(10, '평가점수')
            
            for _, row in stock_data.iterrows():
                insight = {
                    '종목코드': row['종목코드'],
                    '종목명': row['종목명'],
                    '현재가': f"{int(row['현재가']):,}원",
                    '시가총액': f"{int(row['시가총액']/100000000):,}억원",
                    '매출성장률': f"{row['매출성장률']:.2f}%",
                    '순이익률': f"{row['순이익률']:.2f}%",
                    '부채비율': f"{row['부채비율']:.2f}%",
                    'PER': f"{row['PER']:.2f}",
                    'PBR': f"{row['PBR']:.2f}",
                    '평가점수': row['평가점수'],
                    '종합평가': row['종합평가'],
                    '평가이유': row['평가이유']
                }
                insights.append(insight)
                
        except Exception as e:
            logger.error(f"평가 인사이트 생성 실패: {e}")
        
        return insights
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """전체 시장 종합 분석"""
        try:
            if self.evaluation_data.empty:
                return {}
            
            # 전체 통계
            total_stocks = len(self.evaluation_data)
            avg_score = self.evaluation_data['평가점수'].mean()
            
            # 등급별 분포
            grade_distribution = self.evaluation_data['종합평가'].value_counts().to_dict()
            
            # 상위/하위 종목
            top_stocks = self.evaluation_data.nlargest(5, '평가점수')[['종목명', '평가점수']].to_dict('records')
            bottom_stocks = self.evaluation_data.nsmallest(5, '평가점수')[['종목명', '평가점수']].to_dict('records')
            
            # 평균 지표
            avg_metrics = {
                '평균_매출성장률': f"{self.evaluation_data['매출성장률'].mean():.2f}%",
                '평균_순이익률': f"{self.evaluation_data['순이익률'].mean():.2f}%",
                '평균_부채비율': f"{self.evaluation_data['부채비율'].mean():.2f}%",
                '평균_PER': f"{self.evaluation_data['PER'].mean():.2f}",
                '평균_PBR': f"{self.evaluation_data['PBR'].mean():.2f}"
            }
            
            return {
                '분석일시': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '총_평가종목수': total_stocks,
                '평균_평가점수': f"{avg_score:.1f}",
                '등급별_분포': grade_distribution,
                '상위_5개_종목': top_stocks,
                '하위_5개_종목': bottom_stocks,
                '시장_평균지표': avg_metrics
            }
            
        except Exception as e:
            logger.error(f"종합 분석 실패: {e}")
            return {}