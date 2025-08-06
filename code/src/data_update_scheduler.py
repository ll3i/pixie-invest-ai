#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터 업데이트 스케줄러 모듈
- 주기적으로 금융 데이터 수집 및 처리
- 백그라운드 작업 관리
"""

import os
import time
import schedule
import threading
from datetime import datetime, timedelta
import logging

# 데이터 수집 및 처리 모듈 임포트
from data_collector import DataCollector
from data_processor import DataProcessor
from db_client import get_supabase_client
import pandas as pd
import glob

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataUpdateScheduler")

class DataUpdateScheduler:
    """데이터 업데이트 스케줄러 클래스"""
    def __init__(self):
        self.collector = DataCollector()
        self.processor = DataProcessor()
        self.is_running = False
        self.scheduler_thread = None
    
    def update_stock_tickers(self):
        """주식 종목 코드 업데이트"""
        try:
            logger.info("주식 종목 코드 업데이트 시작")
            df = self.collector.collect_stock_tickers()
            if df is not None:
                self.upload_csv_to_supabase(self.get_latest_file('kor_ticker'), 'kor_ticker')
            logger.info("주식 종목 코드 업데이트 완료")
        except Exception as e:
            logger.error(f"주식 종목 코드 업데이트 실패: {e}")
    
    def update_kor_stock_prices(self, price_df):
        supabase = get_supabase_client()
        df = price_df.copy()
        if '날짜' in df.columns:
            df['날짜'] = pd.to_datetime(df['날짜']).dt.strftime('%Y-%m-%d')
        data = df.to_dict(orient='records')
        for chunk in [data[i:i+500] for i in range(0, len(data), 500)]:
            supabase.table("kor_stock_prices").upsert(chunk).execute()
        print(f"kor_stock_prices 업로드 완료: {len(df)} rows")

    def update_us_stock_prices(self, price_df):
        supabase = get_supabase_client()
        df = price_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        data = df.to_dict(orient='records')
        for chunk in [data[i:i+500] for i in range(0, len(data), 500)]:
            supabase.table("us_stock_prices").upsert(chunk).execute()
        print(f"us_stock_prices 업로드 완료: {len(df)} rows")

    def update_stock_prices(self):
        """주식 가격 데이터 업데이트"""
        try:
            logger.info("주식 가격 데이터 업데이트 시작")
            df = self.collector.collect_stock_prices()
            if df is not None:
                self.upload_csv_to_supabase(self.get_latest_file('kor_price'), 'kor_stock_prices')
            logger.info("주식 가격 데이터 업데이트 완료")
        except Exception as e:
            logger.error(f"주식 가격 데이터 업데이트 실패: {e}")
    
    def update_financial_statements(self):
        """재무제표 데이터 업데이트"""
        try:
            logger.info("재무제표 데이터 업데이트 시작")
            df = self.collector.collect_financial_statements()
            if df is not None:
                self.upload_csv_to_supabase(self.get_latest_file('kor_fs'), 'kor_fs')
            logger.info("재무제표 데이터 업데이트 완료")
        except Exception as e:
            logger.error(f"재무제표 데이터 업데이트 실패: {e}")
    
    def update_valuation_metrics(self):
        """가치평가 지표 업데이트"""
        try:
            logger.info("가치평가 지표 업데이트 시작")
            df = self.collector.collect_valuation_metrics()
            if df is not None:
                self.upload_csv_to_supabase(self.get_latest_file('kor_value'), 'kor_value')
            logger.info("가치평가 지표 업데이트 완료")
        except Exception as e:
            logger.error(f"가치평가 지표 업데이트 실패: {e}")
    
    def update_sector_data(self):
        """섹터별 종목 데이터 업데이트"""
        try:
            logger.info("섹터별 종목 데이터 업데이트 시작")
            df = self.collector.collect_sector_data()
            if df is not None:
                self.upload_csv_to_supabase(self.get_latest_file('kor_sector'), 'kor_sector')
            logger.info("섹터별 종목 데이터 업데이트 완료")
        except Exception as e:
            logger.error(f"섹터별 종목 데이터 업데이트 실패: {e}")
    
    def update_us_stock_prices(self):
        """
        미국주식 가격 데이터 업데이트 (수집 + 전처리)
        - DataCollector().collect_us_stock_prices()로 raw 데이터 수집
        - DataProcessor().process_us_stock_prices()로 전처리/저장
        """
        try:
            logger.info("미국주식 가격 데이터 업데이트 시작")
            if hasattr(self.collector, 'collect_us_stock_prices'):
                price_df = self.collector.collect_us_stock_prices(days=1)
            else:
                logger.warning("DataCollector에 collect_us_stock_prices 함수가 없습니다.")
                price_df = None
            if price_df is not None:
                self.processor.process_us_stock_prices()
                self.upload_us_stock_prices(price_df)
            logger.info("미국주식 가격 데이터 업데이트 완료")
        except Exception as e:
            logger.error(f"미국주식 가격 데이터 업데이트 실패: {e}")
    
    def rebuild_vector_db(self):
        """벡터 DB 재구축"""
        try:
            logger.info("벡터 DB 재구축 시작")
            success = self.processor.build_vector_db()
            if success:
                logger.info("벡터 DB 재구축 완료")
            else:
                logger.error("벡터 DB 재구축 실패")
        except Exception as e:
            logger.error(f"벡터 DB 재구축 실패: {e}")
    
    def update_all_data(self):
        """
        모든 데이터 업데이트 (국내+미국)
        """
        try:
            logger.info("모든 데이터 업데이트 시작")
            # 국내주식
            self.update_stock_tickers()
            self.update_stock_prices()
            self.update_financial_statements()
            self.update_valuation_metrics()
            self.update_sector_data()
            # 미국주식
            self.update_us_stock_prices()
            # 벡터 DB 재구축
            self.rebuild_vector_db()
            logger.info("모든 데이터 업데이트 완료")
        except Exception as e:
            logger.error(f"모든 데이터 업데이트 실패: {e}")
    
    def schedule_jobs(self):
        """매일 자동 데이터 업데이트 스케줄 설정"""
        # 매일 오전 6시: 전날 주가 데이터 수집
        schedule.every().day.at("06:00").do(self.daily_price_update)
        
        # 매일 오전 7시: 뉴스 데이터 수집
        schedule.every().day.at("07:00").do(self.daily_news_update)
        
        # 매일 오전 8시: 재무지표 업데이트 (평일만)
        schedule.every().monday.at("08:00").do(self.update_financial_statements)
        schedule.every().wednesday.at("08:00").do(self.update_valuation_metrics)
        schedule.every().friday.at("08:00").do(self.update_sector_data)
        
        # 매주 월요일: 종목 코드 업데이트
        schedule.every().monday.at("05:00").do(self.update_stock_tickers)
        
        print("일일 자동 업데이트 스케줄 설정 완료")
        print("- 06:00: 전날 주가 데이터 수집")
        print("- 07:00: 뉴스 데이터 수집")
        print("- 08:00: 재무지표 업데이트 (주중)")
        print("- 05:00 (월요일): 종목 코드 업데이트")
    
    def daily_price_update(self):
        """매일 전날 주가 데이터 수집"""
        try:
            logger.info("일일 주가 데이터 업데이트 시작")
            
            # 국내주식 전날 데이터
            success_kor = self.collector.collect_stock_prices(days=1)
            if success_kor is not None:
                self.upload_csv_to_supabase(self.get_latest_file('kor_price'), 'kor_stock_prices')
            
            # 미국주식 전날 데이터  
            success_us = self.collector.collect_us_stock_prices(days=1)
            if success_us is not None:
                self.upload_csv_to_supabase(self.get_latest_file('us_price'), 'us_stock_prices')
            
            logger.info("일일 주가 데이터 업데이트 완료")
        except Exception as e:
            logger.error(f"일일 주가 데이터 업데이트 실패: {e}")
    
    def daily_news_update(self):
        """매일 뉴스 데이터 수집 및 감정 분석"""
        try:
            logger.info("일일 뉴스 데이터 업데이트 시작")
            
            # 최근 3일간 뉴스 수집
            news_df = self.collector.collect_news_data(days=3)
            if news_df is not None:
                self.upload_csv_to_supabase(self.get_latest_file('news'), 'news_data')
                
                # 뉴스 감정 분석 실행 (V2)
                try:
                    from news_sentiment_uploader_v2 import NewsSentimentUploaderV2
                    sentiment_uploader = NewsSentimentUploaderV2()
                    sentiment_uploader.run()
                    logger.info("뉴스 감정 분석 완료 (V2)")
                except Exception as e:
                    logger.error(f"뉴스 감정 분석 실패: {e}")
            
            logger.info("일일 뉴스 데이터 업데이트 완료")
        except Exception as e:
            logger.error(f"일일 뉴스 데이터 업데이트 실패: {e}")
    
    def run_immediate_update(self, update_type='all'):
        """즉시 데이터 업데이트 실행"""
        if update_type == 'all':
            threading.Thread(target=self.update_all_data).start()
            logger.info("모든 데이터 업데이트가 백그라운드에서 실행 중입니다.")
        elif update_type == 'prices':
            threading.Thread(target=self.daily_price_update).start()
            logger.info("주가 데이터 업데이트가 백그라운드에서 실행 중입니다.")
        elif update_type == 'news':
            threading.Thread(target=self.daily_news_update).start()
            logger.info("뉴스 데이터 업데이트가 백그라운드에서 실행 중입니다.")
        elif update_type == 'financials':
            threading.Thread(target=self.update_financial_statements).start()
            logger.info("재무제표 데이터 업데이트가 백그라운드에서 실행 중입니다.")
        elif update_type == 'valuations':
            threading.Thread(target=self.update_valuation_metrics).start()
            logger.info("가치평가 지표 업데이트가 백그라운드에서 실행 중입니다.")
        elif update_type == 'tickers':
            threading.Thread(target=self.update_stock_tickers).start()
            logger.info("주식 종목 코드 업데이트가 백그라운드에서 실행 중입니다.")
        elif update_type == 'sectors':
            threading.Thread(target=self.update_sector_data).start()
            logger.info("섹터별 종목 데이터 업데이트가 백그라운드에서 실행 중입니다.")
        elif update_type == 'vector_db':
            threading.Thread(target=self.rebuild_vector_db).start()
            logger.info("벡터 DB 재구축이 백그라운드에서 실행 중입니다.")
        elif update_type == 'historical_setup':
            threading.Thread(target=self.run_immediate_historical_setup).start()
            logger.info("3년간 과거 데이터 수집이 백그라운드에서 실행 중입니다.")
        else:
            logger.error(f"알 수 없는 업데이트 유형: {update_type}")
            return False
        
        return True
    
    def run_scheduler(self):
        """스케줄러 실행"""
        logger.info("스케줄러 시작")
        self.is_running = True
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 스케줄 확인
    
    def start(self):
        """스케줄러 시작"""
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            # 작업 스케줄링
            self.schedule_jobs()
            
            # 스케줄러 스레드 시작
            self.scheduler_thread = threading.Thread(target=self.run_scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            
            logger.info("스케줄러가 백그라운드에서 실행 중입니다.")
            return True
        else:
            logger.warning("스케줄러가 이미 실행 중입니다.")
            return False
    
    def stop(self):
        """스케줄러 중지"""
        if self.scheduler_thread is not None and self.scheduler_thread.is_alive():
            self.is_running = False
            self.scheduler_thread.join(timeout=5)
            logger.info("스케줄러가 중지되었습니다.")
            return True
        else:
            logger.warning("실행 중인 스케줄러가 없습니다.")
            return False

    def run_immediate_historical_setup(self):
        """3년간 과거 데이터 즉시 수집 (초기 설정용)"""
        try:
            logger.info("과거 3년간 데이터 수집 시작")
            
            # 초기 설정 실행
            success = self.collector.run_initial_data_setup()
            
            if success:
                # 수집된 데이터를 데이터베이스에 업로드
                self.upload_all_collected_data()
                logger.info("과거 3년간 데이터 수집 및 업로드 완료")
            else:
                logger.error("과거 데이터 수집 실패")
            
            return success
        except Exception as e:
            logger.error(f"과거 데이터 수집 실패: {e}")
            return False
    
    def upload_all_collected_data(self):
        """수집된 모든 데이터를 데이터베이스에 업로드"""
        try:
            # 국내 주가 데이터
            kor_price_file = self.get_latest_file('kor_price')
            if kor_price_file:
                self.upload_csv_to_supabase(kor_price_file, 'kor_stock_prices')
            
            # 미국 주가 데이터
            us_price_file = self.get_latest_file('us_price')
            if us_price_file:
                self.upload_csv_to_supabase(us_price_file, 'us_stock_prices')
            
            # 종목 정보
            kor_ticker_file = self.get_latest_file('kor_ticker')
            if kor_ticker_file:
                self.upload_csv_to_supabase(kor_ticker_file, 'kor_tickers')
            
            us_ticker_file = self.get_latest_file('us_ticker')
            if us_ticker_file:
                self.upload_csv_to_supabase(us_ticker_file, 'us_tickers')
            
            # 재무제표 데이터
            fs_file = self.get_latest_file('kor_fs')
            if fs_file:
                self.upload_csv_to_supabase(fs_file, 'kor_financials')
            
            # 뉴스 데이터
            news_file = self.get_latest_file('news')
            if news_file:
                self.upload_csv_to_supabase(news_file, 'news_data')
            
            logger.info("모든 수집 데이터 업로드 완료")
        except Exception as e:
            logger.error(f"데이터 업로드 실패: {e}")

    def get_latest_file(self, prefix):
        files = sorted(glob.glob(f'data/raw/{prefix}_*.csv'), reverse=True)
        return files[0] if files else None

    def upload_csv_to_supabase(self, csv_path, table_name):
        if not csv_path:
            print(f"{table_name} csv 파일이 없습니다.")
            return
        supabase = get_supabase_client()
        df = pd.read_csv(csv_path)
        for col in ['날짜', '기준일']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
        data = df.to_dict(orient='records')
        for chunk in [data[i:i+500] for i in range(0, len(data), 500)]:
            supabase.table(table_name).upsert(chunk).execute()
        print(f"{table_name} 업로드 완료: {len(df)} rows")

# 모듈 테스트용 코드
if __name__ == "__main__":
    scheduler = DataUpdateScheduler()
    
    # 테스트: 즉시 가격 데이터 업데이트 실행
    scheduler.run_immediate_update('prices')
    
    # 테스트: 스케줄러 시작
    scheduler.start()
    
    try:
        # 메인 스레드 유지
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # 스케줄러 중지
        scheduler.stop()
        print("프로그램이 종료되었습니다.") 