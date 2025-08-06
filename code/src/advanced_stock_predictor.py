"""
고도화된 주가 예측 모델
ARIMA-X 모델과 뉴스 감정 분석을 결합한 예측 시스템
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedStockPredictor:
    """ARIMA-X 모델과 뉴스 감정 분석을 결합한 고도화된 주가 예측기"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.sentiment_weight = 0.3  # 감정 분석 가중치
        
    def predict_with_sentiment(
        self,
        stock_code: str,
        price_data: pd.DataFrame,
        sentiment_data: Dict[str, Any],
        days: int = 30
    ) -> Dict[str, Any]:
        """
        뉴스 감정 분석을 반영한 주가 예측
        
        Args:
            stock_code: 종목 코드
            price_data: 가격 데이터 (Date, Close 컬럼 필요)
            sentiment_data: 뉴스 감정 분석 데이터
            days: 예측 일수
            
        Returns:
            예측 결과 딕셔너리
        """
        try:
            # 데이터 전처리
            price_data = price_data.copy()
            price_data['Date'] = pd.to_datetime(price_data['Date'])
            price_data.set_index('Date', inplace=True)
            price_data.sort_index(inplace=True)
            
            # 로그 변환 (안정성 향상)
            price_data['log_close'] = np.log(price_data['Close'])
            
            # 감정 점수 시계열 생성
            sentiment_series = self._create_sentiment_series(
                price_data.index,
                sentiment_data
            )
            
            # ARIMA-X 모델 학습
            model, predictions = self._fit_arimax_model(
                price_data['log_close'],
                sentiment_series,
                days
            )
            
            # 예측값 역변환
            predictions_exp = np.exp(predictions)
            
            # 감정 분석 기반 조정
            adjusted_predictions = self._adjust_predictions_by_sentiment(
                predictions_exp,
                sentiment_data
            )
            
            # 신뢰 구간 계산
            confidence_intervals = self._calculate_confidence_intervals(
                adjusted_predictions,
                sentiment_data
            )
            
            # 기술적 지표 계산
            technical_indicators = self._calculate_technical_indicators(
                price_data['Close'].values,
                adjusted_predictions
            )
            
            # 투자 시그널 생성
            investment_signals = self._generate_investment_signals(
                price_data['Close'].values[-1],
                adjusted_predictions,
                sentiment_data
            )
            
            return {
                'stock_code': stock_code,
                'current_price': float(price_data['Close'].values[-1]),
                'predictions': adjusted_predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'sentiment_impact': self._calculate_sentiment_impact(sentiment_data),
                'technical_indicators': technical_indicators,
                'investment_signals': investment_signals,
                'prediction_dates': [
                    (price_data.index[-1] + timedelta(days=i+1)).strftime('%Y-%m-%d')
                    for i in range(days)
                ]
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {stock_code}: {e}")
            return self._get_fallback_prediction(stock_code, price_data, days)
    
    def _create_sentiment_series(
        self,
        dates: pd.DatetimeIndex,
        sentiment_data: Dict[str, Any]
    ) -> pd.Series:
        """감정 점수 시계열 생성"""
        # 기본 감정 점수
        base_sentiment = sentiment_data.get('average_score', 0.5)
        
        # 시간에 따른 감정 변화 시뮬레이션
        sentiment_series = pd.Series(index=dates, dtype=float)
        
        # 최근 30일 감정 트렌드 반영
        recent_days = min(30, len(dates))
        for i in range(recent_days):
            # 시간이 지날수록 감정의 영향력 감소
            decay_factor = 0.95 ** i
            sentiment_series.iloc[-(i+1)] = base_sentiment * decay_factor + 0.5 * (1 - decay_factor)
        
        # 나머지 기간은 중립값
        sentiment_series.fillna(0.5, inplace=True)
        
        return sentiment_series
    
    def _fit_arimax_model(
        self,
        price_series: pd.Series,
        sentiment_series: pd.Series,
        forecast_steps: int
    ) -> Tuple[Any, np.ndarray]:
        """ARIMA-X 모델 학습 및 예측"""
        try:
            # 정상성 검정
            adf_result = adfuller(price_series.dropna())
            p_value = adf_result[1]
            
            # 차분 필요 여부 결정
            d = 0 if p_value < 0.05 else 1
            
            # ARIMA-X 모델 생성 (외생변수로 감정 점수 사용)
            model = ARIMA(
                price_series,
                exog=sentiment_series,
                order=(2, d, 1),  # (p, d, q)
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # 모델 학습
            fitted_model = model.fit(disp=False)
            
            # 미래 감정 점수 예측 (간단한 방법)
            future_sentiment = np.full(forecast_steps, sentiment_series.iloc[-1])
            
            # 예측
            forecast = fitted_model.forecast(
                steps=forecast_steps,
                exog=future_sentiment
            )
            
            return fitted_model, forecast
            
        except Exception as e:
            logger.warning(f"ARIMA-X failed, using simple ARIMA: {e}")
            # 폴백: 단순 ARIMA
            model = ARIMA(price_series, order=(1, 1, 1))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=forecast_steps)
            return fitted_model, forecast
    
    def _adjust_predictions_by_sentiment(
        self,
        predictions: np.ndarray,
        sentiment_data: Dict[str, Any]
    ) -> np.ndarray:
        """감정 분석 기반 예측값 조정"""
        sentiment_score = sentiment_data.get('average_score', 0.5)
        overall_sentiment = sentiment_data.get('overall_sentiment', '중립적')
        
        # 감정에 따른 조정 계수
        if overall_sentiment == '긍정적':
            adjustment_factor = 1 + (sentiment_score - 0.5) * self.sentiment_weight
        elif overall_sentiment == '부정적':
            adjustment_factor = 1 - (0.5 - sentiment_score) * self.sentiment_weight
        else:
            adjustment_factor = 1.0
        
        # 점진적 조정 (시간이 지날수록 감정의 영향 감소)
        adjusted_predictions = predictions.copy()
        for i in range(len(predictions)):
            decay = 0.95 ** i  # 매일 5%씩 감정 영향력 감소
            current_adjustment = 1 + (adjustment_factor - 1) * decay
            adjusted_predictions[i] *= current_adjustment
        
        return adjusted_predictions
    
    def _calculate_confidence_intervals(
        self,
        predictions: np.ndarray,
        sentiment_data: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """신뢰 구간 계산"""
        # 감정 불확실성을 반영한 신뢰 구간
        sentiment_uncertainty = abs(sentiment_data.get('average_score', 0.5) - 0.5)
        
        # 기본 표준편차 (예측값의 2%)
        base_std = predictions * 0.02
        
        # 감정 불확실성 반영
        adjusted_std = base_std * (1 + sentiment_uncertainty)
        
        # 시간이 지날수록 불확실성 증가
        for i in range(len(predictions)):
            adjusted_std[i] *= (1 + 0.01 * i)  # 매일 1%씩 불확실성 증가
        
        return {
            'upper_95': (predictions + 1.96 * adjusted_std).tolist(),
            'lower_95': (predictions - 1.96 * adjusted_std).tolist(),
            'upper_68': (predictions + adjusted_std).tolist(),
            'lower_68': (predictions - adjusted_std).tolist()
        }
    
    def _calculate_technical_indicators(
        self,
        historical_prices: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """기술적 지표 계산"""
        # 이동평균
        ma20 = np.mean(historical_prices[-20:]) if len(historical_prices) >= 20 else np.mean(historical_prices)
        ma60 = np.mean(historical_prices[-60:]) if len(historical_prices) >= 60 else ma20
        
        # 예측 가격의 이동평균 대비 위치
        predicted_ma_position = {
            'vs_ma20': (predictions[-1] / ma20 - 1) * 100,
            'vs_ma60': (predictions[-1] / ma60 - 1) * 100
        }
        
        # RSI (간단한 버전)
        price_changes = np.diff(historical_prices[-14:])
        gains = price_changes[price_changes > 0]
        losses = -price_changes[price_changes < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1
        
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'ma20': float(ma20),
            'ma60': float(ma60),
            'predicted_ma_position': predicted_ma_position,
            'rsi': float(rsi),
            'trend': 'bullish' if predictions[-1] > predictions[0] else 'bearish'
        }
    
    def _generate_investment_signals(
        self,
        current_price: float,
        predictions: np.ndarray,
        sentiment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """투자 시그널 생성"""
        # 예상 수익률
        expected_return = (predictions[-1] / current_price - 1) * 100
        
        # 감정 점수
        sentiment_score = sentiment_data.get('average_score', 0.5)
        
        # 시그널 강도 계산
        signal_strength = 0
        
        # 가격 상승 예상
        if expected_return > 0:
            signal_strength += min(expected_return / 10, 3)  # 최대 3점
        
        # 긍정적 감정
        if sentiment_score > 0.6:
            signal_strength += (sentiment_score - 0.6) * 5  # 최대 2점
        
        # 시그널 결정
        if signal_strength >= 4:
            signal = 'strong_buy'
            confidence = 'high'
        elif signal_strength >= 2:
            signal = 'buy'
            confidence = 'medium'
        elif signal_strength >= -1:
            signal = 'hold'
            confidence = 'medium'
        elif signal_strength >= -3:
            signal = 'sell'
            confidence = 'medium'
        else:
            signal = 'strong_sell'
            confidence = 'high'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'expected_return': float(expected_return),
            'signal_strength': float(signal_strength),
            'factors': {
                'price_trend': 'positive' if expected_return > 0 else 'negative',
                'sentiment': sentiment_data.get('overall_sentiment', '중립적'),
                'sentiment_score': float(sentiment_score)
            }
        }
    
    def _calculate_sentiment_impact(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """감정 분석의 영향도 계산"""
        sentiment_score = sentiment_data.get('average_score', 0.5)
        
        # 감정의 방향성
        if sentiment_score > 0.6:
            direction = 'positive'
            magnitude = (sentiment_score - 0.6) / 0.4
        elif sentiment_score < 0.4:
            direction = 'negative'
            magnitude = (0.4 - sentiment_score) / 0.4
        else:
            direction = 'neutral'
            magnitude = 0
        
        return {
            'direction': direction,
            'magnitude': float(magnitude),
            'score': float(sentiment_score),
            'impact_percentage': float(magnitude * self.sentiment_weight * 100)
        }
    
    def _get_fallback_prediction(
        self,
        stock_code: str,
        price_data: pd.DataFrame,
        days: int
    ) -> Dict[str, Any]:
        """폴백 예측 (단순 추세 기반)"""
        try:
            prices = price_data['Close'].values
            
            # 최근 추세 계산
            recent_returns = np.diff(prices[-20:]) / prices[-21:-1]
            avg_return = np.mean(recent_returns)
            
            # 단순 예측
            predictions = []
            last_price = prices[-1]
            
            for i in range(days):
                # 변동성 감소 적용
                daily_return = avg_return * (0.95 ** i)
                last_price = last_price * (1 + daily_return)
                predictions.append(last_price)
            
            return {
                'stock_code': stock_code,
                'current_price': float(prices[-1]),
                'predictions': predictions,
                'method': 'simple_trend',
                'warning': 'Advanced prediction failed, using simple trend'
            }
            
        except Exception as e:
            logger.error(f"Fallback prediction also failed: {e}")
            return {
                'stock_code': stock_code,
                'error': 'Prediction failed',
                'message': str(e)
            }