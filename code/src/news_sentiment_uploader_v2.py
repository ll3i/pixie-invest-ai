"""
뉴스 감정 분석 및 Supabase 업로드 시스템 V2
제공된 테이블 구조에 맞춰 업데이트된 버전
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import re
import time

# dotenv 임포트를 try-except로 처리
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Using system environment variables.")

# supabase 임포트를 try-except로 처리
try:
    from supabase import create_client, Client
except ImportError:
    print("Warning: supabase not found. Supabase features will be disabled.")
    create_client = None
    Client = None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"news_sentiment_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수는 이미 위에서 로드됨

# 상위 디렉토리를 Python 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from news_sentiment_analyzer import NewsSentimentAnalyzer


class NewsSentimentUploaderV2:
    """뉴스 감정 분석 및 Supabase 업로드 클래스 V2"""
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.script_dir), "data", "raw")
        
        # Supabase 초기화
        self.supabase_url = os.environ.get('SUPABASE_URL')
        self.supabase_key = os.environ.get('SUPABASE_KEY')
        
        if self.supabase_url and self.supabase_key and create_client:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase 클라이언트 초기화 성공")
            except Exception as e:
                logger.error(f"Supabase 클라이언트 초기화 실패: {e}")
                self.supabase = None
        else:
            self.supabase = None
            if not create_client:
                logger.warning("Supabase 라이브러리가 설치되지 않았습니다.")
            else:
                logger.warning("Supabase URL 또는 KEY가 설정되지 않았습니다.")
        
        # 감정 분석기 초기화
        self.analyzer = NewsSentimentAnalyzer()
        
        # 종목 코드 매핑
        self.ticker_mapping = {
            '삼성전자': '005930',
            'SK하이닉스': '000660',
            'LG에너지솔루션': '373220',
            'NAVER': '035420',
            '네이버': '035420',
            '카카오': '035720',
            '현대차': '005380',
            '기아': '000270',
            'POSCO홀딩스': '005490',
            '포스코': '005490',
            'LG화학': '051910',
            'KB금융': '105560',
            '신한지주': '055550',
            '삼성바이오로직스': '207940',
            '셀트리온': '068270',
            '카카오뱅크': '323410',
            '삼성SDI': '006400',
            'SK이노베이션': '096770',
            'LG전자': '066570',
            '하나금융지주': '086790',
            '우리금융지주': '316140'
        }
        
        # 섹터 매핑
        self.sector_mapping = {
            '005930': 'IT/전자',
            '000660': 'IT/전자',
            '373220': '2차전지',
            '035420': 'IT/인터넷',
            '035720': 'IT/인터넷',
            '005380': '자동차',
            '000270': '자동차',
            '005490': '철강',
            '051910': '화학',
            '105560': '금융',
            '055550': '금융',
            '207940': '바이오',
            '068270': '바이오',
            '323410': '금융',
            '006400': '2차전지'
        }
        
        # 카테고리 매핑 (뉴스 유형)
        self.news_categories = {
            'tech': 'IT',
            'finance': '금융',
            'auto': '자동차',
            'bio': '바이오',
            'energy': '에너지',
            'materials': '소재',
            'consumer': '소비재',
            'market': '시장',
            'economy': '경제',
            'policy': '정책',
            'global': '글로벌'
        }
    
    def get_latest_news_file(self) -> Optional[str]:
        """가장 최근 뉴스 파일 가져오기"""
        try:
            files = [f for f in os.listdir(self.data_dir) if f.startswith('news_') and f.endswith('.csv')]
            if not files:
                logger.warning("뉴스 파일이 없습니다.")
                return None
            
            # 날짜순 정렬
            files.sort(reverse=True)
            latest_file = os.path.join(self.data_dir, files[0])
            logger.info(f"최신 뉴스 파일: {files[0]}")
            return latest_file
        except Exception as e:
            logger.error(f"뉴스 파일 조회 실패: {e}")
            return None
    
    def classify_sentiment(self, score: float) -> str:
        """감정 점수를 긍정/부정/중립으로 분류"""
        if score > 0.1:
            return '긍정'
        elif score < -0.1:
            return '부정'
        else:
            return '중립'
    
    def calculate_confidence(self, keyword_counts: Dict) -> float:
        """감정 분석 신뢰도 계산"""
        total_keywords = sum(len(keywords) for keywords in keyword_counts.values())
        if total_keywords == 0:
            return 0.5
        
        # 키워드 수에 따른 신뢰도 계산 (최대 1.0)
        confidence = min(1.0, 0.5 + (total_keywords * 0.1))
        return round(confidence, 3)
    
    def extract_related_tickers(self, text: str) -> List[str]:
        """텍스트에서 관련 종목 코드 추출"""
        related_tickers = []
        
        for company, ticker in self.ticker_mapping.items():
            if company.lower() in text.lower():
                if ticker not in related_tickers:
                    related_tickers.append(ticker)
        
        return related_tickers
    
    def calculate_importance_score(self, news_item: Dict) -> int:
        """뉴스 중요도 점수 계산 (0-100)"""
        score = 50  # 기본 점수
        
        # 제목에 주요 키워드 포함 여부
        important_keywords = ['급등', '급락', '돌파', '최고', '최저', '위기', '호재', '악재', '금리', '규제']
        title = news_item.get('title', '').lower()
        
        for keyword in important_keywords:
            if keyword in title:
                score += 10
        
        # 관련 종목 수
        related_tickers_count = len(news_item.get('related_tickers', []))
        score += min(20, related_tickers_count * 5)
        
        # 감정 극성 (극단적일수록 중요)
        sentiment_score = abs(news_item.get('sentiment_score', 0))
        if sentiment_score > 0.5:
            score += 20
        elif sentiment_score > 0.3:
            score += 10
        
        # 최대 100점으로 제한
        return min(100, score)
    
    def process_news_data(self, news_file: str) -> pd.DataFrame:
        """뉴스 데이터 처리 및 감정 분석"""
        try:
            # 뉴스 데이터 로드
            news_df = pd.read_csv(news_file, encoding='utf-8')
            logger.info(f"뉴스 데이터 로드 완료: {len(news_df)}개 기사")
            
            # 처리 시작 시간
            start_time = time.time()
            
            # 감정 분석 수행
            analyzed_news = []
            
            for idx, row in news_df.iterrows():
                try:
                    # 제목과 요약 합쳐서 분석
                    full_text = f"{row.get('title', '')} {row.get('summary', '')}"
                    
                    # 감정 분석
                    sentiment_score, keyword_counts = self.analyzer.analyze_sentiment(full_text)
                    sentiment_label = self.classify_sentiment(sentiment_score)
                    confidence_score = self.calculate_confidence(keyword_counts)
                    
                    # 키워드 추출
                    all_keywords = []
                    for keywords in keyword_counts.values():
                        all_keywords.extend(list(keywords.keys()))
                    
                    # 긍정/부정 키워드 분리
                    positive_keywords = list(keyword_counts.get('positive', {}).keys())
                    negative_keywords = list(keyword_counts.get('negative', {}).keys())
                    
                    # 관련 종목 추출
                    related_tickers = self.extract_related_tickers(full_text)
                    
                    # 중요도 계산
                    importance_item = {
                        'title': row.get('title', ''),
                        'related_tickers': related_tickers,
                        'sentiment_score': sentiment_score
                    }
                    importance_score = self.calculate_importance_score(importance_item)
                    
                    # 카테고리 결정
                    category = self.detect_category(full_text)
                    
                    # 처리 시간 계산 (밀리초)
                    processing_time = int((time.time() - start_time) * 1000 / (idx + 1))
                    
                    analyzed_item = {
                        # 기본 정보
                        'title': row.get('title', ''),
                        'summary': row.get('summary', '')[:500] if pd.notna(row.get('summary')) else None,
                        'content': None,  # 전체 내용은 별도 수집 필요
                        'url': row.get('link', ''),
                        'published_date': row.get('published', ''),
                        'source': row.get('source', ''),
                        
                        # 감정 분석 결과
                        'sentiment': sentiment_label,
                        'sentiment_score': round(sentiment_score, 3),
                        'confidence_score': confidence_score,
                        
                        # 키워드 및 관련 정보
                        'keywords': all_keywords[:10],  # 상위 10개
                        'related_tickers': related_tickers,
                        'importance_score': importance_score,
                        
                        # 메타데이터
                        'category': self.news_categories.get(category, '기타'),
                        'language': 'ko' if row.get('is_korean', True) else 'en',
                        
                        # 상세 감정 분석 (별도 테이블용)
                        'positive_keywords': positive_keywords[:5],
                        'negative_keywords': negative_keywords[:5],
                        'neutral_keywords': [],  # 중립 키워드는 별도 분석 필요
                        'positive_score': round(len(positive_keywords) / max(1, len(all_keywords)), 3),
                        'negative_score': round(len(negative_keywords) / max(1, len(all_keywords)), 3),
                        'processing_time_ms': processing_time
                    }
                    
                    analyzed_news.append(analyzed_item)
                    
                except Exception as e:
                    logger.error(f"뉴스 분석 오류 (idx: {idx}): {e}")
                    continue
            
            logger.info(f"감정 분석 완료: {len(analyzed_news)}개 기사")
            return pd.DataFrame(analyzed_news)
            
        except Exception as e:
            logger.error(f"뉴스 데이터 처리 실패: {e}")
            return pd.DataFrame()
    
    def detect_category(self, text: str) -> str:
        """뉴스 카테고리 자동 감지"""
        text_lower = text.lower()
        
        # 카테고리별 키워드
        category_keywords = {
            'tech': ['IT', '기술', '반도체', '소프트웨어', '인공지능', 'AI', '5G'],
            'finance': ['금융', '은행', '증권', '보험', '카드', '대출'],
            'auto': ['자동차', '전기차', '배터리', '모빌리티'],
            'bio': ['바이오', '제약', '헬스케어', '신약', '임상'],
            'energy': ['에너지', '전력', '신재생', '태양광', '풍력', '원자력'],
            'materials': ['소재', '철강', '화학', '석유화학'],
            'consumer': ['소비', '유통', '식품', '패션', '화장품'],
            'market': ['코스피', '코스닥', '증시', '주가'],
            'economy': ['경제', '금리', '환율', '인플레이션', 'GDP'],
            'policy': ['정책', '정부', '규제', '법안'],
            'global': ['미국', '중국', '유럽', '일본', '글로벌']
        }
        
        # 각 카테고리별 매칭 점수 계산
        scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > 0:
                scores[category] = score
        
        # 가장 높은 점수의 카테고리 반환
        if scores:
            return max(scores, key=scores.get)
        return 'general'
    
    def upload_to_supabase(self, analyzed_df: pd.DataFrame) -> bool:
        """분석된 뉴스 데이터를 Supabase에 업로드"""
        if self.supabase is None:
            logger.warning("Supabase 클라이언트가 초기화되지 않았습니다.")
            return False
        
        try:
            uploaded_count = 0
            
            for idx, row in analyzed_df.iterrows():
                try:
                    # 1. news_articles 테이블에 삽입
                    article_data = {
                        'title': row['title'],
                        'summary': row['summary'],
                        'content': row.get('content'),
                        'url': row['url'],
                        'published_date': row['published_date'],
                        'sentiment': row['sentiment'],
                        'sentiment_score': float(row['sentiment_score']),
                        'confidence_score': float(row['confidence_score']),
                        'keywords': row['keywords'],
                        'related_tickers': row['related_tickers'],
                        'importance_score': int(row['importance_score']),
                        'source': row['source'],
                        'category': row['category'],
                        'language': row['language']
                    }
                    
                    # 중복 확인 (URL 기준)
                    existing = self.supabase.table('news_articles').select('id').eq('url', row['url']).execute()
                    
                    if not existing.data:
                        # 새로운 뉴스 삽입
                        result = self.supabase.table('news_articles').insert(article_data).execute()
                        
                        if result.data:
                            article_id = result.data[0]['id']
                            uploaded_count += 1
                            
                            # 2. sentiment_details 테이블에 상세 정보 삽입
                            sentiment_detail = {
                                'article_id': article_id,
                                'positive_score': float(row['positive_score']),
                                'negative_score': float(row['negative_score']),
                                'neutral_score': float(1 - row['positive_score'] - row['negative_score']),
                                'positive_keywords': row['positive_keywords'],
                                'negative_keywords': row['negative_keywords'],
                                'neutral_keywords': row.get('neutral_keywords', []),
                                'analysis_method': 'keyword_based',
                                'model_version': 'v1.0',
                                'processing_time_ms': int(row['processing_time_ms'])
                            }
                            self.supabase.table('sentiment_details').insert(sentiment_detail).execute()
                            
                            # 3. ticker_impact 테이블에 종목별 영향 분석
                            for ticker in row['related_tickers']:
                                # 예상 가격 방향 결정
                                if row['sentiment_score'] > 0.3:
                                    expected_direction = '상승'
                                elif row['sentiment_score'] < -0.3:
                                    expected_direction = '하락'
                                else:
                                    expected_direction = '중립'
                                
                                ticker_impact = {
                                    'article_id': article_id,
                                    'ticker_symbol': ticker,
                                    'company_name': self.get_company_name(ticker),
                                    'impact_score': int(row['sentiment_score'] * 100),
                                    'relevance_score': 0.8,  # 기본값, 추후 정교화 필요
                                    'expected_price_direction': expected_direction,
                                    'sector': self.sector_mapping.get(ticker, '기타'),
                                    'market': 'KOSPI' if ticker in ['005930', '000660', '005380'] else 'KOSDAQ'
                                }
                                self.supabase.table('ticker_impact').insert(ticker_impact).execute()
                    
                    else:
                        # 기존 뉴스는 감정 분석 결과만 업데이트
                        update_data = {
                            'sentiment': row['sentiment'],
                            'sentiment_score': float(row['sentiment_score']),
                            'confidence_score': float(row['confidence_score']),
                            'keywords': row['keywords'],
                            'importance_score': int(row['importance_score'])
                        }
                        self.supabase.table('news_articles').update(update_data).eq('url', row['url']).execute()
                    
                except Exception as e:
                    logger.error(f"뉴스 업로드 오류 (idx: {idx}): {e}")
                    continue
            
            # 4. 일별 통계 업데이트
            self.update_daily_stats(analyzed_df)
            
            # 5. 키워드 트렌드 업데이트
            self.update_keyword_trends(analyzed_df)
            
            logger.info(f"Supabase 업로드 완료: {uploaded_count}개 신규 뉴스")
            return True
            
        except Exception as e:
            logger.error(f"Supabase 업로드 실패: {e}")
            return False
    
    def get_company_name(self, ticker: str) -> str:
        """종목 코드로 회사명 반환"""
        ticker_to_name = {v: k for k, v in self.ticker_mapping.items()}
        return ticker_to_name.get(ticker, ticker)
    
    def update_daily_stats(self, analyzed_df: pd.DataFrame):
        """일별 감정 통계 업데이트"""
        try:
            today = datetime.now().date()
            
            # 카테고리별 통계 계산
            for category in analyzed_df['category'].unique():
                cat_df = analyzed_df[analyzed_df['category'] == category]
                
                stats = {
                    'date': today.isoformat(),
                    'category': category,
                    'source': None,  # 전체 소스
                    'total_articles': len(cat_df),
                    'positive_count': len(cat_df[cat_df['sentiment'] == '긍정']),
                    'negative_count': len(cat_df[cat_df['sentiment'] == '부정']),
                    'neutral_count': len(cat_df[cat_df['sentiment'] == '중립']),
                    'avg_sentiment_score': float(cat_df['sentiment_score'].mean()),
                    'avg_importance_score': float(cat_df['importance_score'].mean())
                }
                
                # UPSERT
                existing = self.supabase.table('daily_sentiment_stats').select('id').eq('date', today.isoformat()).eq('category', category).is_('source', 'null').execute()
                
                if existing.data:
                    self.supabase.table('daily_sentiment_stats').update(stats).eq('id', existing.data[0]['id']).execute()
                else:
                    self.supabase.table('daily_sentiment_stats').insert(stats).execute()
                    
        except Exception as e:
            logger.error(f"일별 통계 업데이트 실패: {e}")
    
    def update_keyword_trends(self, analyzed_df: pd.DataFrame):
        """키워드 트렌드 업데이트"""
        try:
            today = datetime.now().date()
            
            # 모든 키워드 수집
            keyword_sentiments = {}
            
            for _, row in analyzed_df.iterrows():
                sentiment = row['sentiment']
                for keyword in row['keywords']:
                    if keyword not in keyword_sentiments:
                        keyword_sentiments[keyword] = {'긍정': 0, '부정': 0, '중립': 0}
                    keyword_sentiments[keyword][sentiment] += 1
            
            # 키워드별 트렌드 업데이트
            for keyword, counts in keyword_sentiments.items():
                total = sum(counts.values())
                sentiment_ratio = (counts['긍정'] - counts['부정']) / total if total > 0 else 0
                
                trend_data = {
                    'keyword': keyword,
                    'date': today.isoformat(),
                    'mention_count': total,
                    'positive_mentions': counts['긍정'],
                    'negative_mentions': counts['부정'],
                    'neutral_mentions': counts['중립'],
                    'sentiment_ratio': round(sentiment_ratio, 3)
                }
                
                # UPSERT
                existing = self.supabase.table('keyword_trends').select('id').eq('date', today.isoformat()).eq('keyword', keyword).execute()
                
                if existing.data:
                    self.supabase.table('keyword_trends').update(trend_data).eq('id', existing.data[0]['id']).execute()
                else:
                    self.supabase.table('keyword_trends').insert(trend_data).execute()
                    
        except Exception as e:
            logger.error(f"키워드 트렌드 업데이트 실패: {e}")
    
    def run(self):
        """뉴스 감정 분석 및 업로드 실행"""
        logger.info("=== 뉴스 감정 분석 및 업로드 시작 (V2) ===")
        
        # 1. 최신 뉴스 파일 가져오기
        news_file = self.get_latest_news_file()
        if not news_file:
            logger.error("처리할 뉴스 파일이 없습니다.")
            return False
        
        # 2. 뉴스 데이터 처리 및 감정 분석
        analyzed_df = self.process_news_data(news_file)
        if analyzed_df.empty:
            logger.error("분석된 뉴스 데이터가 없습니다.")
            return False
        
        # 3. 분석 결과 요약 출력
        sentiment_dist = analyzed_df['sentiment'].value_counts()
        logger.info(f"감정 분포: 긍정 {sentiment_dist.get('긍정', 0)}, 부정 {sentiment_dist.get('부정', 0)}, 중립 {sentiment_dist.get('중립', 0)}")
        logger.info(f"평균 감정 점수: {analyzed_df['sentiment_score'].mean():.3f}")
        logger.info(f"평균 중요도: {analyzed_df['importance_score'].mean():.1f}")
        
        # 4. Supabase 업로드
        if self.supabase:
            success = self.upload_to_supabase(analyzed_df)
            if success:
                logger.info("Supabase 업로드 성공")
            else:
                logger.error("Supabase 업로드 실패")
        
        # 5. 분석 결과 CSV 저장
        analyzed_file = news_file.replace('.csv', '_analyzed_v2.csv')
        analyzed_df.to_csv(analyzed_file, index=False, encoding='utf-8-sig')
        logger.info(f"분석 결과 저장: {analyzed_file}")
        
        logger.info("=== 뉴스 감정 분석 및 업로드 완료 (V2) ===")
        return True


def main():
    """메인 함수"""
    uploader = NewsSentimentUploaderV2()
    success = uploader.run()
    
    if success:
        print("✅ 뉴스 감정 분석 및 업로드가 완료되었습니다.")
    else:
        print("❌ 뉴스 감정 분석 중 오류가 발생했습니다.")


if __name__ == "__main__":
    main()