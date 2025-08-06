"""
뉴스 감정 분석 및 Supabase 업로드 V4
이진 분류 감정 분석기 사용 (긍정/부정만)
"""
import os
import sys
import logging
from datetime import datetime, date
import pandas as pd
import json
from typing import List, Dict, Optional
import time

# 환경 변수 로드
import os as _os
_env_file = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), '.env')
if _os.path.exists(_env_file):
    with open(_env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    _os.environ[key.strip()] = value.strip()

# Supabase 클라이언트
try:
    from supabase import create_client, Client
except ImportError:
    print("Warning: Supabase 패키지가 설치되지 않았습니다. 일부 기능이 제한될 수 있습니다.")
    create_client = None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"news_sentiment_v4_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 상위 디렉토리를 Python 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from news_sentiment_analyzer_binary import BinaryNewsSentimentAnalyzer


class NewsSentimentUploaderV4:
    """이진 분류 뉴스 감정 분석 및 Supabase 업로드 클래스"""
    
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
            logger.warning("Supabase 연결 정보가 없습니다.")
        
        # 이진 분류 감정 분석기 초기화
        self.analyzer = BinaryNewsSentimentAnalyzer()
        logger.info("이진 분류 감정 분석기 초기화 완료")
    
    def analyze_and_save_news(self, news_file: str = None) -> pd.DataFrame:
        """뉴스 파일을 분석하고 결과 저장"""
        try:
            # 파일 경로 설정
            if news_file is None:
                today = date.today().strftime('%Y%m%d')
                news_file = os.path.join(self.data_dir, f'news_{today}.csv')
            
            if not os.path.exists(news_file):
                logger.error(f"뉴스 파일을 찾을 수 없습니다: {news_file}")
                return pd.DataFrame()
            
            # 뉴스 데이터 로드
            logger.info(f"뉴스 파일 로드: {news_file}")
            news_df = pd.read_csv(news_file, encoding='utf-8-sig')
            
            # 필수 컬럼 확인
            required_columns = ['title', 'summary', 'link', 'published']
            for col in required_columns:
                if col not in news_df.columns:
                    news_df[col] = ''
            
            # URL 컬럼 이름 통일
            if 'link' in news_df.columns and 'url' not in news_df.columns:
                news_df['url'] = news_df['link']
            
            # published_date 컬럼 이름 통일
            if 'published' in news_df.columns and 'published_date' not in news_df.columns:
                news_df['published_date'] = news_df['published']
            
            # 이진 분류 감정 분석 수행
            logger.info(f"{len(news_df)}개 뉴스 이진 분류 감정 분석 시작...")
            analyzed_df = self.analyzer.analyze_news_dataframe(news_df)
            
            # 추가 필드 설정
            analyzed_df['language'] = 'ko'
            analyzed_df['analyzed_at'] = datetime.now().isoformat()
            
            # 카테고리가 없으면 기본값 설정
            if 'category' not in analyzed_df.columns:
                analyzed_df['category'] = '경제'
            
            # 분석 결과 저장
            processed_dir = os.path.join(os.path.dirname(self.data_dir), "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            output_file = os.path.join(processed_dir, f'analyzed_news_binary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            analyzed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"분석 결과 저장: {output_file}")
            
            # 감정 분석 요약
            summary = self.analyzer.get_sentiment_summary(analyzed_df)
            logger.info(f"이진 분류 완료 - 긍정: {summary['sentiment_distribution']['긍정']} ({summary['positive_ratio']*100:.1f}%), "
                       f"부정: {summary['sentiment_distribution']['부정']} ({summary['negative_ratio']*100:.1f}%)")
            logger.info(f"평균 감정 점수: {summary['average_sentiment_score']}")
            logger.info(f"시장 신호: {summary['market_signal']}")
            
            # 요약 정보도 저장
            summary_file = os.path.join(processed_dir, f'sentiment_summary_binary_{datetime.now().strftime("%Y%m%d")}.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            return analyzed_df
            
        except Exception as e:
            logger.error(f"뉴스 분석 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def clear_today_data(self):
        """오늘 날짜의 기존 데이터 삭제"""
        if not self.supabase:
            return False
        
        try:
            today = date.today().isoformat()
            logger.info(f"오늘({today}) 기존 데이터 삭제 중...")
            
            # news_articles 테이블에서 오늘 데이터 삭제
            result = self.supabase.table('news_articles').delete().gte('created_at', f"{today}T00:00:00").execute()
            logger.info(f"기존 뉴스 데이터 삭제 완료")
            
            return True
        except Exception as e:
            logger.error(f"데이터 삭제 중 오류: {e}")
            return False
    
    def upload_to_supabase(self, analyzed_df: pd.DataFrame, clear_existing: bool = True) -> bool:
        """분석된 뉴스를 Supabase에 업로드"""
        if self.supabase is None:
            logger.warning("Supabase 클라이언트가 초기화되지 않았습니다.")
            return False
        
        if analyzed_df.empty:
            logger.warning("업로드할 데이터가 없습니다.")
            return False
        
        # 기존 데이터 삭제 옵션
        if clear_existing:
            self.clear_today_data()
        
        try:
            uploaded_count = 0
            error_count = 0
            
            for idx, row in analyzed_df.iterrows():
                try:
                    # 데이터 준비
                    article_data = {
                        'title': str(row['title'])[:500],
                        'summary': str(row['summary'])[:1000],
                        'content': str(row.get('content', ''))[:2000],
                        'url': str(row['url'])[:500],
                        'published_date': str(row['published_date']),
                        'sentiment': str(row['sentiment']),  # 긍정 또는 부정만
                        'sentiment_score': float(row['sentiment_score']),
                        'confidence_score': float(row['confidence_score']),
                        'keywords': row.get('positive_keywords', []) + row.get('negative_keywords', []),
                        'related_tickers': row.get('related_tickers', []),
                        'importance_score': int(row.get('importance_score', 50)),
                        'source': str(row.get('source', 'Unknown'))[:100],
                        'category': str(row.get('category', '경제'))[:50],
                        'language': 'ko'
                    }
                    
                    # 중복 확인
                    existing = self.supabase.table('news_articles').select('id').eq('url', article_data['url']).execute()
                    
                    if not existing.data:
                        # 새 기사 삽입
                        result = self.supabase.table('news_articles').insert(article_data).execute()
                        
                        if result.data:
                            article_id = result.data[0]['id']
                            uploaded_count += 1
                            
                            # sentiment_details 테이블에 상세 정보 삽입
                            sentiment_detail = {
                                'article_id': article_id,
                                'positive_score': float(row.get('positive_score', 0)),
                                'negative_score': float(row.get('negative_score', 0)),
                                'neutral_score': 0.0,  # 이진 분류이므로 중립은 0
                                'positive_keywords': row.get('positive_keywords', []),
                                'negative_keywords': row.get('negative_keywords', []),
                                'analysis_method': 'binary_keyword_based',
                                'model_version': 'v4.0',
                                'confidence_score': float(row['confidence_score'])
                            }
                            
                            self.supabase.table('sentiment_details').insert(sentiment_detail).execute()
                            
                            # 진행 상황 표시
                            if uploaded_count % 10 == 0:
                                logger.info(f"진행: {uploaded_count}개 업로드 완료")
                    else:
                        logger.debug(f"중복 기사 스킵: {row['title'][:50]}...")
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"기사 업로드 오류 ({idx}): {str(e)[:100]}")
                    if error_count > 10:
                        logger.error("오류가 너무 많아 업로드를 중단합니다.")
                        break
            
            logger.info(f"Supabase 업로드 완료 - 성공: {uploaded_count}개, 실패: {error_count}개")
            
            # 통계 업데이트
            if uploaded_count > 0:
                self.update_statistics(analyzed_df)
            
            return uploaded_count > 0
            
        except Exception as e:
            logger.error(f"Supabase 업로드 중 오류: {e}")
            return False
    
    def update_statistics(self, df: pd.DataFrame):
        """일별 통계 업데이트"""
        try:
            today = date.today().isoformat()
            
            # 감정별 통계
            sentiment_counts = df['sentiment'].value_counts().to_dict()
            
            # daily_sentiment_stats 업데이트
            stats_data = {
                'date': today,
                'positive_count': sentiment_counts.get('긍정', 0),
                'negative_count': sentiment_counts.get('부정', 0),
                'neutral_count': 0,  # 이진 분류이므로 중립은 0
                'total_count': len(df),
                'avg_sentiment_score': float(df['sentiment_score'].mean()),
                'strong_positive_count': len(df[df['sentiment_score'] >= 0.5]),
                'strong_negative_count': len(df[df['sentiment_score'] <= -0.5])
            }
            
            # 기존 레코드 확인
            existing = self.supabase.table('daily_sentiment_stats').select('id').eq('date', today).eq('category', None).eq('source', None).execute()
            
            if existing.data:
                # 업데이트
                self.supabase.table('daily_sentiment_stats').update(stats_data).eq('id', existing.data[0]['id']).execute()
            else:
                # 삽입
                self.supabase.table('daily_sentiment_stats').insert(stats_data).execute()
            
            logger.info("일별 통계 업데이트 완료")
            
        except Exception as e:
            logger.error(f"통계 업데이트 오류: {e}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print(f"이진 분류 뉴스 감정 분석 V4 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("(중립 제거, 긍정/부정만 분류)")
    print("=" * 60)
    
    uploader = NewsSentimentUploaderV4()
    
    # 뉴스 분석
    analyzed_df = uploader.analyze_and_save_news()
    
    if not analyzed_df.empty:
        # Supabase 업로드
        if uploader.supabase:
            # 기존 데이터 삭제 여부 확인
            print("\n기존 데이터를 삭제하고 새로 업로드하시겠습니까?")
            print("(기존 데이터를 유지하려면 'n'을 입력하세요)")
            response = input("삭제 후 업로드? (Y/n): ").strip().lower()
            
            clear_existing = response != 'n'
            
            success = uploader.upload_to_supabase(analyzed_df, clear_existing=clear_existing)
            if success:
                print("\n✅ 모든 작업 완료!")
            else:
                print("\n⚠️ 업로드 중 일부 오류 발생")
        else:
            print("\n⚠️ Supabase 연결 없이 분석만 완료")
    else:
        print("\n❌ 뉴스 분석 실패")


if __name__ == "__main__":
    main()
    input("\nEnter 키를 눌러 종료...")