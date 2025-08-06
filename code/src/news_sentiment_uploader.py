"""
뉴스 감정 분석 및 Supabase 업로드 시스템
매일 수집된 뉴스를 분석하여 긍정/부정/중립으로 분류하고 Supabase에 업로드
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from supabase import create_client, Client
from dotenv import load_dotenv
import json
import re

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

# 환경 변수 로드
load_dotenv()

# 상위 디렉토리를 Python 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from news_sentiment_analyzer import NewsSentimentAnalyzer


class NewsSentimentUploader:
    """뉴스 감정 분석 및 Supabase 업로드 클래스"""
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.script_dir), "data", "raw")
        
        # Supabase 초기화
        self.supabase_url = os.environ.get('SUPABASE_URL')
        self.supabase_key = os.environ.get('SUPABASE_KEY')
        
        if self.supabase_url and self.supabase_key:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase 클라이언트 초기화 완료")
        else:
            self.supabase = None
            logger.warning("Supabase 환경 변수가 설정되지 않았습니다.")
        
        # 감정 분석기 초기화
        self.analyzer = NewsSentimentAnalyzer()
        
        # 카테고리 매핑
        self.category_mapping = {
            '삼성전자': 'tech',
            'SK하이닉스': 'tech',
            'LG': 'tech',
            'NAVER': 'tech',
            '카카오': 'tech',
            '현대차': 'auto',
            '기아': 'auto',
            'KB금융': 'finance',
            '신한': 'finance',
            '하나': 'finance',
            '우리': 'finance',
            'POSCO': 'materials',
            '포스코': 'materials',
            '바이오': 'healthcare',
            '제약': 'healthcare',
            '셀트리온': 'healthcare',
            '에너지': 'energy',
            '전력': 'energy',
            '석유': 'energy',
            '건설': 'construction',
            '부동산': 'realestate',
            '리츠': 'realestate',
            '소비': 'consumer',
            '유통': 'consumer',
            '식품': 'consumer'
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
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """텍스트에서 주요 키워드 추출"""
        # 불용어 제거
        stopwords = {'는', '은', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '와', '과', '하고', '및', '또는', '등', '년', '월', '일'}
        
        # 명사 추출 (간단한 방법)
        words = re.findall(r'[가-힣]+', text)
        words = [w for w in words if len(w) > 1 and w not in stopwords]
        
        # 빈도수 계산
        from collections import Counter
        word_counts = Counter(words)
        
        # 상위 키워드 반환
        return [word for word, count in word_counts.most_common(top_n)]
    
    def detect_category(self, text: str) -> str:
        """뉴스 카테고리 자동 감지"""
        text_lower = text.lower()
        
        for keyword, category in self.category_mapping.items():
            if keyword.lower() in text_lower:
                return category
        
        # 기본 카테고리
        if '시장' in text or '코스피' in text or '코스닥' in text:
            return 'market'
        elif '경제' in text or '금리' in text or '환율' in text:
            return 'economy'
        elif '정책' in text or '정부' in text or '규제' in text:
            return 'policy'
        else:
            return 'general'
    
    def process_news_data(self, news_file: str) -> pd.DataFrame:
        """뉴스 데이터 처리 및 감정 분석"""
        try:
            # 뉴스 데이터 로드
            news_df = pd.read_csv(news_file, encoding='utf-8')
            logger.info(f"뉴스 데이터 로드 완료: {len(news_df)}개 기사")
            
            # MCP 수집 뉴스인 경우 keyword 컬럼이 있을 수 있음
            has_keyword_column = 'keyword' in news_df.columns
            
            # 감정 분석 수행
            analyzed_news = []
            
            for idx, row in news_df.iterrows():
                try:
                    # 제목과 요약 합쳐서 분석
                    full_text = f"{row.get('title', '')} {row.get('summary', '')}"
                    
                    # 감정 분석
                    sentiment_score, keyword_counts = self.analyzer.analyze_sentiment(full_text)
                    sentiment_label = self.classify_sentiment(sentiment_score)
                    
                    # 키워드 추출
                    keywords = self.extract_keywords(full_text)
                    
                    # 카테고리 감지
                    category = self.detect_category(full_text)
                    
                    # 긍정/부정 주요 키워드
                    positive_keywords = list(keyword_counts.get('positive', {}).keys())[:3]
                    negative_keywords = list(keyword_counts.get('negative', {}).keys())[:3]
                    
                    analyzed_item = {
                        'title': row.get('title', ''),
                        'summary': row.get('summary', '')[:500] if pd.notna(row.get('summary')) else '',
                        'link': row.get('link', ''),
                        'published': row.get('published', ''),
                        'source': row.get('source', ''),
                        'sentiment_score': round(sentiment_score, 3),
                        'sentiment_label': sentiment_label,
                        'category': category,
                        'keywords': keywords,
                        'positive_keywords': positive_keywords,
                        'negative_keywords': negative_keywords,
                        'is_korean': row.get('is_korean', True),
                        'search_keyword': row.get('keyword', '') if has_keyword_column else ''  # MCP 검색 키워드
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
    
    def upload_to_supabase(self, analyzed_df: pd.DataFrame) -> bool:
        """분석된 뉴스 데이터를 Supabase에 업로드"""
        if self.supabase is None:
            logger.warning("Supabase 클라이언트가 초기화되지 않았습니다.")
            return False
        
        try:
            uploaded_count = 0
            
            for idx, row in analyzed_df.iterrows():
                try:
                    # Supabase에 맞는 데이터 형식으로 변환
                    news_data = {
                        'title': row['title'],
                        'content': row['summary'],
                        'url': row['link'],
                        'published_date': row['published'],
                        'source': row['source'],
                        'category': row['category'],
                        'sentiment_score': float(row['sentiment_score']),
                        'sentiment_label': row['sentiment_label'],
                        'keywords': row['keywords'],
                        'positive_keywords': row['positive_keywords'],
                        'negative_keywords': row['negative_keywords']
                    }
                    
                    # 중복 확인 (URL 기준)
                    existing = self.supabase.table('news').select('id').eq('url', row['link']).execute()
                    
                    if not existing.data:
                        # 새로운 뉴스 삽입
                        response = self.supabase.table('news').insert(news_data).execute()
                        uploaded_count += 1
                    else:
                        # 기존 뉴스 업데이트 (감정 분석 결과만)
                        update_data = {
                            'sentiment_score': float(row['sentiment_score']),
                            'sentiment_label': row['sentiment_label'],
                            'category': row['category'],
                            'keywords': row['keywords'],
                            'positive_keywords': row['positive_keywords'],
                            'negative_keywords': row['negative_keywords']
                        }
                        response = self.supabase.table('news').update(update_data).eq('url', row['link']).execute()
                    
                except Exception as e:
                    logger.error(f"뉴스 업로드 오류 (idx: {idx}): {e}")
                    continue
            
            logger.info(f"Supabase 업로드 완료: {uploaded_count}개 신규 뉴스")
            return True
            
        except Exception as e:
            logger.error(f"Supabase 업로드 실패: {e}")
            return False
    
    def generate_daily_summary(self, analyzed_df: pd.DataFrame) -> Dict:
        """일일 뉴스 감정 분석 요약 생성"""
        if analyzed_df.empty:
            return {}
        
        # 전체 감정 분포
        sentiment_counts = analyzed_df['sentiment_label'].value_counts().to_dict()
        
        # 카테고리별 감정 분석
        category_sentiment = {}
        for category in analyzed_df['category'].unique():
            cat_df = analyzed_df[analyzed_df['category'] == category]
            category_sentiment[category] = {
                'count': len(cat_df),
                'avg_sentiment': round(cat_df['sentiment_score'].mean(), 3),
                'sentiment_dist': cat_df['sentiment_label'].value_counts().to_dict()
            }
        
        # 주요 긍정/부정 키워드
        all_positive = []
        all_negative = []
        for _, row in analyzed_df.iterrows():
            all_positive.extend(row['positive_keywords'])
            all_negative.extend(row['negative_keywords'])
        
        from collections import Counter
        top_positive = Counter(all_positive).most_common(10)
        top_negative = Counter(all_negative).most_common(10)
        
        # 시장 종합 점수
        overall_sentiment = analyzed_df['sentiment_score'].mean()
        
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_news': len(analyzed_df),
            'overall_sentiment': round(overall_sentiment, 3),
            'sentiment_distribution': sentiment_counts,
            'category_analysis': category_sentiment,
            'top_positive_keywords': top_positive,
            'top_negative_keywords': top_negative,
            'market_mood': self.analyzer._get_market_mood(overall_sentiment)
        }
        
        return summary
    
    def save_summary_to_file(self, summary: Dict):
        """요약 정보를 파일로 저장"""
        try:
            summary_dir = os.path.join(os.path.dirname(self.script_dir), "data", "processed")
            os.makedirs(summary_dir, exist_ok=True)
            
            filename = f"news_sentiment_summary_{summary['date']}.json"
            filepath = os.path.join(summary_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"요약 파일 저장 완료: {filename}")
            
        except Exception as e:
            logger.error(f"요약 파일 저장 실패: {e}")
    
    def run(self):
        """뉴스 감정 분석 및 업로드 실행"""
        logger.info("=== 뉴스 감정 분석 및 업로드 시작 ===")
        
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
        
        # 3. 일일 요약 생성
        summary = self.generate_daily_summary(analyzed_df)
        logger.info(f"일일 요약 생성 완료: 전체 감정 점수 {summary['overall_sentiment']}")
        
        # 4. 요약 파일 저장
        self.save_summary_to_file(summary)
        
        # 5. Supabase 업로드
        if self.supabase:
            success = self.upload_to_supabase(analyzed_df)
            if success:
                logger.info("Supabase 업로드 성공")
            else:
                logger.error("Supabase 업로드 실패")
        
        # 6. 분석 결과 CSV 저장
        analyzed_file = news_file.replace('.csv', '_analyzed.csv')
        analyzed_df.to_csv(analyzed_file, index=False, encoding='utf-8-sig')
        logger.info(f"분석 결과 저장: {analyzed_file}")
        
        logger.info("=== 뉴스 감정 분석 및 업로드 완료 ===")
        return True


def main():
    """메인 함수"""
    uploader = NewsSentimentUploader()
    success = uploader.run()
    
    if success:
        print("✅ 뉴스 감정 분석 및 업로드가 완료되었습니다.")
    else:
        print("❌ 뉴스 감정 분석 중 오류가 발생했습니다.")


if __name__ == "__main__":
    main()