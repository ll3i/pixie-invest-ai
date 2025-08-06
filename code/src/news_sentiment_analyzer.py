"""
뉴스 감정 분석기
뉴스 데이터에서 긍정/부정 감정을 분석하고 시장 분위기를 파악
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter

class NewsSentimentAnalyzer:
    """뉴스 감정 분석 및 시장 분위기 분석"""
    
    def __init__(self):
        # 감정 분석용 키워드 사전
        self.positive_keywords = {
            # 긍정적 시장 용어
            '상승': 3, '급등': 4, '강세': 3, '호조': 3, '회복': 2,
            '신고가': 4, '최고치': 4, '돌파': 3, '상향': 3, '개선': 2,
            '증가': 2, '성장': 3, '호재': 3, '기대': 2, '긍정': 2,
            '흑자': 3, '수익': 2, '실적개선': 3, '매수': 2, '순매수': 3,
            '투자': 2, '유입': 2, '확대': 2, '호전': 3, '반등': 2,
            '상승세': 3, '강보합': 2, '양호': 2, '견조': 2, '활황': 3,
            
            # 긍정적 기업 실적
            '최대실적': 4, '어닝서프라이즈': 4, '실적호조': 3, '매출증가': 3,
            '이익증가': 3, '성과': 2, '달성': 2, '초과': 3,
            
            # 긍정적 경제 지표
            '경기회복': 3, '수출증가': 3, '고용개선': 3, '소비증가': 3,
            '금리인하': 2, '안정': 2, '완화': 2
        }
        
        self.negative_keywords = {
            # 부정적 시장 용어
            '하락': -3, '급락': -4, '약세': -3, '부진': -3, '침체': -3,
            '저점': -3, '최저치': -4, '이탈': -3, '하향': -3, '악화': -3,
            '감소': -2, '둔화': -2, '우려': -2, '불안': -3, '부정': -2,
            '적자': -3, '손실': -3, '실적악화': -3, '매도': -2, '순매도': -3,
            '이탈': -2, '유출': -2, '축소': -2, '위축': -3, '반락': -2,
            '하락세': -3, '약보합': -2, '저조': -2, '위기': -4, '불황': -4,
            
            # 부정적 기업 실적
            '실적부진': -3, '어닝쇼크': -4, '실적하향': -3, '매출감소': -3,
            '이익감소': -3, '손실': -3, '미달': -2, '하회': -3,
            
            # 부정적 경제 지표
            '경기침체': -4, '수출감소': -3, '고용악화': -3, '소비위축': -3,
            '금리인상': -2, '불안정': -3, '긴축': -2,
            
            # 리스크 관련
            '리스크': -2, '위험': -2, '불확실': -2, '변동성': -2
        }
        
        # 종목명 패턴
        self.stock_patterns = {
            '삼성전자': ['삼성전자', '삼전', 'Samsung Electronics'],
            'SK하이닉스': ['SK하이닉스', 'SK Hynix', '하이닉스'],
            'LG에너지솔루션': ['LG에너지솔루션', 'LG에너지', 'LGES'],
            'NAVER': ['네이버', 'NAVER', 'Naver'],
            '카카오': ['카카오', 'Kakao'],
            '현대차': ['현대차', '현대자동차', 'Hyundai Motor'],
            '기아': ['기아', 'KIA', 'Kia'],
            'POSCO홀딩스': ['포스코', 'POSCO', '포스코홀딩스'],
            'LG화학': ['LG화학', 'LG Chem'],
            'KB금융': ['KB금융', '국민은행', 'KB'],
            '신한지주': ['신한지주', '신한은행', 'Shinhan'],
            '삼성바이오로직스': ['삼성바이오로직스', '삼성바이오'],
            '셀트리온': ['셀트리온', 'Celltrion'],
            '카카오뱅크': ['카카오뱅크', 'KakaoBank'],
            '삼성SDI': ['삼성SDI', 'Samsung SDI']
        }
        
    def analyze_sentiment(self, text: str) -> Tuple[float, Dict[str, int]]:
        """
        텍스트의 감정 점수 계산
        Returns:
            - sentiment_score: -1(매우부정) ~ 1(매우긍정)
            - keyword_counts: 키워드별 출현 횟수
        """
        if not text:
            return 0.0, {}
            
        # 텍스트 전처리
        text = text.lower()
        
        # 긍정/부정 점수 계산
        positive_score = 0
        negative_score = 0
        keyword_counts = {'positive': {}, 'negative': {}}
        
        # 긍정 키워드 검색
        for keyword, weight in self.positive_keywords.items():
            count = text.count(keyword.lower())
            if count > 0:
                positive_score += weight * count
                keyword_counts['positive'][keyword] = count
                
        # 부정 키워드 검색
        for keyword, weight in self.negative_keywords.items():
            count = text.count(keyword.lower())
            if count > 0:
                negative_score += abs(weight) * count
                keyword_counts['negative'][keyword] = count
        
        # 종합 점수 계산 (-1 ~ 1)
        total_score = positive_score - negative_score
        max_score = max(positive_score + negative_score, 1)
        sentiment_score = total_score / (max_score * 10)  # 정규화
        sentiment_score = max(-1, min(1, sentiment_score))  # -1 ~ 1 범위로 제한
        
        return sentiment_score, keyword_counts
    
    def analyze_news_sentiment(self, news_df: pd.DataFrame) -> Dict:
        """
        뉴스 데이터프레임 전체의 감정 분석
        """
        if news_df.empty:
            return {
                'overall_sentiment': 0.0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'top_positive_keywords': [],
                'top_negative_keywords': [],
                'stock_sentiments': {},
                'daily_sentiment': {}
            }
        
        results = []
        all_positive_keywords = Counter()
        all_negative_keywords = Counter()
        stock_sentiments = {stock: [] for stock in self.stock_patterns.keys()}
        
        # 각 뉴스 분석
        for _, news in news_df.iterrows():
            # 제목과 요약 합쳐서 분석
            text = f"{news.get('title', '')} {news.get('summary', '')}"
            sentiment_score, keyword_counts = self.analyze_sentiment(text)
            
            # 키워드 집계
            for keyword, count in keyword_counts.get('positive', {}).items():
                all_positive_keywords[keyword] += count
            for keyword, count in keyword_counts.get('negative', {}).items():
                all_negative_keywords[keyword] += count
            
            # 종목별 감정 분석
            for stock, patterns in self.stock_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in text.lower():
                        stock_sentiments[stock].append(sentiment_score)
                        break
            
            results.append({
                'title': news.get('title', ''),
                'sentiment_score': sentiment_score,
                'published': news.get('published', '')
            })
        
        # 전체 감정 점수
        sentiment_scores = [r['sentiment_score'] for r in results]
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # 감정 분포
        positive_count = sum(1 for s in sentiment_scores if s > 0.1)
        negative_count = sum(1 for s in sentiment_scores if s < -0.1)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        # 종목별 평균 감정
        stock_sentiment_avg = {}
        for stock, scores in stock_sentiments.items():
            if scores:
                stock_sentiment_avg[stock] = {
                    'sentiment': np.mean(scores),
                    'count': len(scores),
                    'trend': 'positive' if np.mean(scores) > 0.1 else 'negative' if np.mean(scores) < -0.1 else 'neutral'
                }
        
        # 일별 감정 추이
        daily_sentiment = {}
        for result in results:
            try:
                date = pd.to_datetime(result['published']).date()
                if date not in daily_sentiment:
                    daily_sentiment[date] = []
                daily_sentiment[date].append(result['sentiment_score'])
            except:
                continue
                
        daily_sentiment_avg = {
            str(date): np.mean(scores) 
            for date, scores in daily_sentiment.items()
        }
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'sentiment_distribution': {
                'positive': positive_count,
                'neutral': neutral_count,
                'negative': negative_count
            },
            'top_positive_keywords': all_positive_keywords.most_common(10),
            'top_negative_keywords': all_negative_keywords.most_common(10),
            'stock_sentiments': stock_sentiment_avg,
            'daily_sentiment': daily_sentiment_avg,
            'market_mood': self._get_market_mood(overall_sentiment),
            'analysis_time': datetime.now().isoformat()
        }
    
    def _get_market_mood(self, sentiment_score: float) -> Dict[str, str]:
        """시장 분위기 판단"""
        if sentiment_score >= 0.5:
            mood = "매우 긍정적"
            description = "시장이 강한 상승세를 보이고 있으며, 투자 심리가 매우 긍정적입니다."
            recommendation = "적극적인 투자를 고려할 수 있지만, 과열 여부도 점검하세요."
        elif sentiment_score >= 0.2:
            mood = "긍정적"
            description = "시장이 안정적인 상승세를 유지하고 있습니다."
            recommendation = "점진적인 투자를 고려하되, 리스크 관리에 유의하세요."
        elif sentiment_score >= -0.2:
            mood = "중립적"
            description = "시장이 방향성을 정하지 못하고 있습니다."
            recommendation = "관망세를 유지하며 시장 변화를 주시하세요."
        elif sentiment_score >= -0.5:
            mood = "부정적"
            description = "시장에 우려가 확산되고 있습니다."
            recommendation = "보수적인 접근이 필요하며, 손실 관리에 집중하세요."
        else:
            mood = "매우 부정적"
            description = "시장이 큰 하락 압력을 받고 있습니다."
            recommendation = "현금 비중을 늘리고 리스크를 최소화하세요."
            
        return {
            'mood': mood,
            'description': description,
            'recommendation': recommendation,
            'score': sentiment_score
        }
    
    def get_investment_signals(self, sentiment_analysis: Dict) -> Dict[str, any]:
        """감정 분석 결과를 기반으로 투자 신호 생성"""
        signals = {
            'market_signal': 'hold',
            'confidence': 0.5,
            'top_stocks': [],
            'avoid_stocks': [],
            'sector_preference': []
        }
        
        # 전체 시장 신호
        overall = sentiment_analysis['overall_sentiment']
        if overall >= 0.3:
            signals['market_signal'] = 'buy'
            signals['confidence'] = min(0.9, 0.5 + overall)
        elif overall <= -0.3:
            signals['market_signal'] = 'sell'
            signals['confidence'] = min(0.9, 0.5 - overall)
        
        # 종목별 추천
        stock_sentiments = sentiment_analysis.get('stock_sentiments', {})
        sorted_stocks = sorted(
            stock_sentiments.items(), 
            key=lambda x: x[1]['sentiment'] if isinstance(x[1], dict) else 0,
            reverse=True
        )
        
        # 상위/하위 종목
        if sorted_stocks:
            signals['top_stocks'] = [
                (stock, data) for stock, data in sorted_stocks[:3] 
                if isinstance(data, dict) and data['sentiment'] > 0.2
            ]
            signals['avoid_stocks'] = [
                (stock, data) for stock, data in sorted_stocks[-3:] 
                if isinstance(data, dict) and data['sentiment'] < -0.2
            ]
        
        return signals


def main():
    """테스트용 메인 함수"""
    analyzer = NewsSentimentAnalyzer()
    
    # 샘플 뉴스 데이터
    sample_news = pd.DataFrame([
        {
            'title': '코스피 3,200선 돌파...외국인 매수세 지속',
            'summary': '코스피가 외국인 순매수에 힘입어 3,200선을 돌파했다. 반도체주 강세가 지속되고 있다.',
            'published': datetime.now().isoformat()
        },
        {
            'title': '금리 인상 우려...증시 하락 압력',
            'summary': '미국 연준의 금리 인상 우려로 증시가 하락 압력을 받고 있다. 투자자들의 불안감이 확산되고 있다.',
            'published': datetime.now().isoformat()
        },
        {
            'title': '삼성전자 실적 개선...영업이익 급증',
            'summary': '삼성전자가 역대 최대 실적을 기록했다. 반도체 부문 호조가 이어지고 있다.',
            'published': datetime.now().isoformat()
        }
    ])
    
    # 감정 분석
    result = analyzer.analyze_news_sentiment(sample_news)
    print("감정 분석 결과:")
    print(f"전체 감정 점수: {result['overall_sentiment']}")
    print(f"시장 분위기: {result['market_mood']['mood']}")
    print(f"긍정 키워드: {result['top_positive_keywords']}")
    print(f"부정 키워드: {result['top_negative_keywords']}")
    
    # 투자 신호
    signals = analyzer.get_investment_signals(result)
    print(f"\n투자 신호: {signals['market_signal']} (신뢰도: {signals['confidence']})")


if __name__ == "__main__":
    main()