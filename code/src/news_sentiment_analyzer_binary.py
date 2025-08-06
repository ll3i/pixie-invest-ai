"""
이진 분류 뉴스 감정 분석기
중립 제거, 긍정/부정만 분류
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter

class BinaryNewsSentimentAnalyzer:
    """이진 분류 뉴스 감정 분석 (긍정/부정만)"""
    
    def __init__(self):
        # 확장된 감정 분석용 키워드 사전 (가중치 조정)
        self.positive_keywords = {
            # 강한 긍정 (5-6점)
            '급등': 6, '폭등': 6, '대박': 6, '최고치': 6, '신고가': 6, '사상최대': 6,
            '급상승': 5, '급증': 5, '대폭상승': 5, '역대최고': 5, '최대실적': 5,
            '호황': 5, '활황': 5, '강력': 5, '훌륭': 5,
            
            # 중간 긍정 (3-4점)
            '상승': 4, '오름': 4, '올라': 4, '증가': 4, '늘어': 4, '확대': 4,
            '호조': 4, '호전': 4, '개선': 4, '회복': 4, '반등': 4, '상향': 4,
            '강세': 3, '양호': 3, '긍정': 3, '매수': 3, '순매수': 3, '유입': 3,
            '성장': 3, '흑자': 3, '이익': 3, '수익': 3, '실적개선': 3, '돌파': 3,
            '신기록': 4, '최고': 4, '성공': 3, '달성': 3, '완료': 3,
            
            # 약한 긍정 (1-2점)
            '안정': 2, '견조': 2, '유지': 2, '지속': 2, '기대': 2, '관심': 2,
            '투자': 1, '상승세': 2, '강보합': 1, '보합': 1, '착한': 2,
            '선방': 2, '버티': 2, '방어': 2,
            
            # 실적 관련
            '어닝서프라이즈': 5, '실적호조': 4, '매출증가': 3, '이익증가': 3,
            '영업이익증가': 4, '순이익증가': 4, '흑자전환': 5, '적자축소': 3,
            '실적상향': 4, '컨센서스상회': 4, '기대이상': 4,
            
            # 경제 지표
            '경기회복': 4, '경기개선': 3, '수출증가': 3, '내수회복': 3,
            '고용개선': 3, '실업률하락': 3, '소비증가': 3, '생산증가': 3,
            '금리인하': 3, '유동성확대': 3, '부양책': 3, '경기부양': 3,
            '물가안정': 3, '성장률상승': 4,
            
            # 산업/섹터별
            '반도체호황': 5, '수주증가': 4, '판매호조': 4, '시장점유율확대': 4,
            '신약승인': 5, '기술혁신': 4, '신제품출시': 3, '수출호조': 4,
            '계약체결': 3, '인수': 3, '합병': 3, 'IPO': 3,
            
            # 투자 긍정
            '배당': 3, '자사주매입': 4, '증자': 2, '투자유치': 4,
            '목표가상향': 4, '매수추천': 4, '비중확대': 3
        }
        
        self.negative_keywords = {
            # 강한 부정 (-6 ~ -5점)
            '급락': -6, '폭락': -6, '대폭하락': -6, '최저치': -6, '신저가': -6,
            '급감': -5, '급하락': -5, '대폭감소': -5, '최악': -5, '파산': -6,
            '상장폐지': -6, '부도': -6, '파산': -6, '폐업': -6, '붕괴': -6,
            '충격': -5, '공포': -5, '패닉': -5,
            
            # 중간 부정 (-4 ~ -3점)
            '하락': -4, '내림': -4, '내려': -4, '감소': -4, '줄어': -4, '축소': -4,
            '부진': -4, '악화': -4, '둔화': -4, '위축': -4, '침체': -4, '하향': -4,
            '약세': -3, '우려': -3, '불안': -3, '매도': -3, '순매도': -3, '이탈': -3,
            '적자': -3, '손실': -3, '실적악화': -3, '저조': -3, '위기': -4,
            '실패': -3, '중단': -3, '포기': -3, '철수': -3,
            
            # 약한 부정 (-2 ~ -1점)
            '조정': -2, '차익실현': -2, '관망': -2, '불확실': -2, '변동성': -2,
            '압력': -2, '부담': -2, '하락세': -2, '약보합': -1, '혼조': -1,
            '경계': -2, '주의': -2, '보류': -2,
            
            # 실적 관련
            '어닝쇼크': -5, '실적부진': -4, '매출감소': -3, '이익감소': -3,
            '영업이익감소': -4, '순이익감소': -4, '적자전환': -5, '적자확대': -4,
            '실적하향': -4, '가이던스하향': -4, '컨센서스하회': -4, '기대미달': -4,
            
            # 경제 지표
            '경기침체': -5, '경기둔화': -4, '불황': -5, '수출감소': -3,
            '내수부진': -3, '고용악화': -3, '실업률상승': -3, '소비위축': -3,
            '금리인상': -3, '긴축': -3, '유동성축소': -3, '테이퍼링': -3,
            '인플레이션': -3, '스태그플레이션': -4, '디플레이션': -3,
            
            # 리스크 관련
            '리스크': -2, '위험': -3, '공포': -4, '패닉': -5, '불확실성': -3,
            '규제': -2, '제재': -3, '갈등': -3, '분쟁': -3, '전쟁': -5,
            '소송': -3, '과징금': -3, '벌금': -3, '처벌': -3,
            
            # 산업/섹터별
            '반도체불황': -5, '수주감소': -4, '판매부진': -4, '재고증가': -3,
            '리콜': -4, '결함': -3, '사고': -4, '스캔들': -4,
            '계약취소': -4, '거래중단': -4, '공급중단': -3,
            
            # 투자 부정
            '감자': -4, '상장폐지위기': -5, '거래정지': -4, '투자주의': -3,
            '목표가하향': -4, '매도추천': -4, '비중축소': -3
        }
        
        # 강조 패턴 (문맥에 따라 가중치 증폭)
        self.emphasis_patterns = {
            '매우': 1.5, '정말': 1.5, '너무': 1.5, '굉장히': 1.5, '상당히': 1.3,
            '크게': 1.3, '대폭': 1.5, '소폭': 0.7, '약간': 0.5, '다소': 0.6,
            '잠시': 0.5, '일시적': 0.5, '지속적': 1.3, '계속': 1.2,
            '확실히': 1.4, '분명히': 1.4, '역시': 1.2
        }
        
        # 부정어 패턴
        self.negation_patterns = ['안', '못', '없', '불', '미', '비', '아니']
        
        # 종목명 패턴 (기존과 동일)
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
    
    def check_negation(self, text: str, keyword_position: int, window: int = 20) -> bool:
        """키워드 주변에 부정어가 있는지 확인"""
        start = max(0, keyword_position - window)
        context = text[start:keyword_position]
        
        for neg in self.negation_patterns:
            if neg in context:
                return True
        return False
    
    def apply_emphasis(self, text: str, base_score: float) -> float:
        """강조 표현에 따른 점수 조정"""
        for emphasis, multiplier in self.emphasis_patterns.items():
            if emphasis in text:
                return base_score * multiplier
        return base_score
    
    def analyze_sentiment(self, text: str) -> Tuple[float, Dict[str, any], str]:
        """
        텍스트의 감정 점수 계산 (이진 분류)
        Returns:
            - sentiment_score: -1(매우부정) ~ 1(매우긍정)
            - details: 상세 분석 정보
            - sentiment_label: '긍정' 또는 '부정'만
        """
        if not text:
            return 0.0, {}, '긍정'  # 기본값은 긍정
            
        # 텍스트 전처리
        text_lower = text.lower()
        
        # 긍정/부정 점수 계산
        positive_score = 0
        negative_score = 0
        positive_keywords = []
        negative_keywords = []
        
        # 긍정 키워드 검색
        for keyword, weight in self.positive_keywords.items():
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                # 키워드 위치 찾기
                positions = [m.start() for m in re.finditer(re.escape(keyword_lower), text_lower)]
                for pos in positions:
                    # 부정어 체크
                    if not self.check_negation(text_lower, pos):
                        # 강조 표현 적용
                        adjusted_weight = self.apply_emphasis(text_lower[max(0, pos-10):pos+len(keyword)+10], weight)
                        positive_score += adjusted_weight
                        positive_keywords.append((keyword, adjusted_weight))
                    else:
                        # 부정어가 있으면 반대로
                        negative_score += abs(weight) * 0.8
                        negative_keywords.append((f"not_{keyword}", -weight * 0.8))
                
        # 부정 키워드 검색
        for keyword, weight in self.negative_keywords.items():
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                positions = [m.start() for m in re.finditer(re.escape(keyword_lower), text_lower)]
                for pos in positions:
                    if not self.check_negation(text_lower, pos):
                        adjusted_weight = self.apply_emphasis(text_lower[max(0, pos-10):pos+len(keyword)+10], abs(weight))
                        negative_score += adjusted_weight
                        negative_keywords.append((keyword, -adjusted_weight))
                    else:
                        # 부정어가 있으면 반대로
                        positive_score += abs(weight) * 0.8
                        positive_keywords.append((f"not_{keyword}", abs(weight) * 0.8))
        
        # 종합 점수 계산
        total_positive = positive_score
        total_negative = negative_score
        
        # 점수 차이를 기반으로 감정 결정
        score_diff = total_positive - total_negative
        total_score = total_positive + total_negative
        
        # 감정 점수 계산
        if total_score == 0:
            # 키워드가 없는 경우 긍정으로 분류
            sentiment_score = 0.01
            sentiment_label = '긍정'
        else:
            # 정규화
            sentiment_score = score_diff / max(total_score * 0.5, 10)
            sentiment_score = max(-1, min(1, sentiment_score))
            
            # 이진 분류 (중립 제거)
            if sentiment_score >= 0:
                sentiment_label = '긍정'
                # 긍정이지만 점수가 낮은 경우 최소값 보장
                if sentiment_score < 0.01:
                    sentiment_score = 0.01
            else:
                sentiment_label = '부정'
                # 부정이지만 점수가 높은 경우 최대값 보장
                if sentiment_score > -0.01:
                    sentiment_score = -0.01
        
        # 키워드가 전혀 없는 경우에도 판단
        if len(positive_keywords) == 0 and len(negative_keywords) == 0:
            # 특정 단어나 표현으로 추가 판단
            if any(word in text_lower for word in ['하지만', '그러나', '우려', '걱정', '문제']):
                sentiment_label = '부정'
                sentiment_score = -0.1
            else:
                sentiment_label = '긍정'
                sentiment_score = 0.1
        
        # 상세 정보
        details = {
            'positive_score': round(total_positive, 2),
            'negative_score': round(total_negative, 2),
            'positive_keywords': sorted(positive_keywords, key=lambda x: x[1], reverse=True)[:5],
            'negative_keywords': sorted(negative_keywords, key=lambda x: x[1], reverse=True)[:5],
            'total_keywords': len(positive_keywords) + len(negative_keywords),
            'confidence': min(1.0, total_score / 20) if total_score > 0 else 0.5  # 신뢰도
        }
        
        return round(sentiment_score, 3), details, sentiment_label
    
    def analyze_news_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """뉴스 데이터프레임에 감정 분석 결과 추가 (이진 분류)"""
        results = []
        
        for idx, row in df.iterrows():
            # 제목과 요약 결합하여 분석
            text = f"{row.get('title', '')} {row.get('summary', '')}"
            
            # 감정 분석
            sentiment_score, details, sentiment_label = self.analyze_sentiment(text)
            
            # 기존 데이터에 감정 분석 결과 추가
            result_row = row.to_dict()
            result_row.update({
                'sentiment': sentiment_label,  # 긍정 또는 부정만
                'sentiment_score': sentiment_score,
                'confidence_score': details['confidence'],
                'positive_score': details['positive_score'],
                'negative_score': details['negative_score'],
                'positive_keywords': [k[0] for k in details['positive_keywords']],
                'negative_keywords': [k[0] for k in details['negative_keywords']],
                'total_keywords': details['total_keywords']
            })
            
            # 관련 종목 찾기
            related_tickers = []
            for stock, patterns in self.stock_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in text.lower():
                        related_tickers.append(stock)
                        break
            result_row['related_tickers'] = related_tickers
            
            # 중요도 점수 (감정 강도 + 키워드 수 + 신뢰도)
            importance_score = min(100, int((abs(sentiment_score) * 40 + 
                                            details['total_keywords'] * 3 + 
                                            details['confidence'] * 20)))
            result_row['importance_score'] = importance_score
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """감정 분석 요약 통계 (이진 분류)"""
        if df.empty:
            return {
                'total_count': 0,
                'sentiment_distribution': {'긍정': 0, '부정': 0},
                'average_sentiment_score': 0.0,
                'strong_positive_count': 0,
                'strong_negative_count': 0,
                'top_positive_news': [],
                'top_negative_news': [],
                'positive_ratio': 0.0,
                'negative_ratio': 0.0
            }
        
        # 감정 분포
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        total = len(df)
        
        # 강한 긍정/부정 (점수 기준)
        strong_positive = df[df['sentiment_score'] >= 0.5]
        strong_negative = df[df['sentiment_score'] <= -0.5]
        
        # 상위 긍정/부정 뉴스
        top_positive = df[df['sentiment'] == '긍정'].nlargest(5, 'sentiment_score')[['title', 'sentiment_score', 'positive_keywords']].to_dict('records')
        top_negative = df[df['sentiment'] == '부정'].nsmallest(5, 'sentiment_score')[['title', 'sentiment_score', 'negative_keywords']].to_dict('records')
        
        # 비율 계산
        positive_ratio = sentiment_counts.get('긍정', 0) / total if total > 0 else 0
        negative_ratio = sentiment_counts.get('부정', 0) / total if total > 0 else 0
        
        return {
            'total_count': total,
            'sentiment_distribution': {
                '긍정': sentiment_counts.get('긍정', 0),
                '부정': sentiment_counts.get('부정', 0)
            },
            'average_sentiment_score': round(df['sentiment_score'].mean(), 3),
            'strong_positive_count': len(strong_positive),
            'strong_negative_count': len(strong_negative),
            'top_positive_news': top_positive,
            'top_negative_news': top_negative,
            'positive_ratio': round(positive_ratio, 3),
            'negative_ratio': round(negative_ratio, 3),
            'market_signal': self._get_market_signal(df['sentiment_score'].mean(), positive_ratio, negative_ratio)
        }
    
    def _get_market_signal(self, avg_score: float, positive_ratio: float, negative_ratio: float) -> str:
        """시장 신호 판단 (이진 분류)"""
        if avg_score >= 0.3 or positive_ratio >= 0.7:
            return "강한 매수 신호"
        elif avg_score >= 0.1 or positive_ratio >= 0.6:
            return "매수 신호"
        elif avg_score <= -0.3 or negative_ratio >= 0.7:
            return "강한 매도 신호"
        elif avg_score <= -0.1 or negative_ratio >= 0.6:
            return "매도 신호"
        else:
            return "약한 매수 신호"  # 중립 대신 약한 매수


def test_binary_analyzer():
    """이진 분류 분석기 테스트"""
    analyzer = BinaryNewsSentimentAnalyzer()
    
    # 테스트 케이스
    test_cases = [
        "코스피 급등하며 신고가 경신, 외국인 대규모 순매수",
        "경기 침체 우려에 주가 급락, 투자자들 패닉",
        "삼성전자 실적 소폭 개선되었으나 전망은 불확실",
        "금리 인상으로 부담 증가, 하지만 경제는 견조",
        "반도체 업황 회복세, 수출 호조 지속",
        "코로나 재확산 우려에 여행주 약세",
        "배터리 수주 대박, LG에너지솔루션 주가 상승",
        "실적 부진에 주가 하락, 전망도 어두워",
        "증시 혼조세 지속",  # 중립적 표현
        "오늘 주식시장 마감"  # 중립적 표현
    ]
    
    print("=== 이진 분류 감정 분석 테스트 ===\n")
    for text in test_cases:
        score, details, label = analyzer.analyze_sentiment(text)
        print(f"텍스트: {text}")
        print(f"감정: {label} (점수: {score})")
        print(f"긍정 키워드: {details['positive_keywords'][:3]}")
        print(f"부정 키워드: {details['negative_keywords'][:3]}")
        print(f"신뢰도: {details['confidence']:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    test_binary_analyzer()