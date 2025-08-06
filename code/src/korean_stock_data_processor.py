#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
import time
import re
import csv
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import requests as rq
from bs4 import BeautifulSoup
from io import BytesIO
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.db_client import get_supabase_client

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('korean_stock_data.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """데이터 처리 설정"""
    raw_dir: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    processed_dir: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    request_delay: float = 2.0
    max_retries: int = 3
    test_mode: bool = False  # 테스트 모드 (일부 종목만 처리)
    test_count: int = 30

class KoreanStockDataProcessor:
    """국내주식 데이터 처리 클래스"""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.supabase = get_supabase_client()
        
        # 디렉토리 생성
        os.makedirs(self.config.raw_dir, exist_ok=True)
        os.makedirs(self.config.processed_dir, exist_ok=True)
        
        # 세션 설정
        self.session = rq.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        logger.info("KoreanStockDataProcessor 초기화 완료")
    
    def fetch_business_day(self) -> str:
        """영업일 가져오기"""
        try:
            url = 'https://finance.naver.com/sise/sise_deposit.nhn'
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            parse_day = soup.select_one('div.subtop_sise_graph2 > ul.subtop_chart_note > li > span.tah').text
            biz_day = ''.join(re.findall('[0-9]+', parse_day))
            
            logger.info(f"영업일 수집 완료: {biz_day}")
            return biz_day
            
        except Exception as e:
            logger.error(f"영업일 수집 실패: {e}")
            # 폴백: 오늘 날짜 사용
            return date.today().strftime("%Y%m%d")
    
    def fetch_ticker_list(self, biz_day: str) -> pd.DataFrame:
        """종목 리스트 수집"""
        try:
            gen_otp_url = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
            headers = {
                'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader',
                'User-Agent': 'Mozilla/5.0'
            }
            
            # 코스피 종목 수집
            gen_otp_stk = {
                'mktId': 'STK', 'trdDd': biz_day, 'money': '1',
                'csvxls_isNo': 'false', 'name': 'fileDown',
                'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
            }
            
            otp_stk = self.session.post(gen_otp_url, gen_otp_stk, headers=headers).text
            time.sleep(1)
            
            down_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
            down_sector_stk = self.session.post(down_url, {'code': otp_stk}, headers=headers)
            time.sleep(1)
            
            sector_stk = pd.read_csv(BytesIO(down_sector_stk.content), encoding='EUC-KR')
            
            # 코스닥 종목 수집
            gen_otp_ksq = {
                'mktId': 'KSQ', 'trdDd': biz_day, 'money': '1',
                'csvxls_isNo': 'false', 'name': 'fileDown',
                'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
            }
            
            otp_ksq = self.session.post(gen_otp_url, gen_otp_ksq, headers=headers).text
            time.sleep(1)
            
            down_sector_ksq = self.session.post(down_url, {'code': otp_ksq}, headers=headers)
            time.sleep(1)
            
            sector_ksq = pd.read_csv(BytesIO(down_sector_ksq.content), encoding='EUC-KR')
            
            # 데이터 병합 및 전처리
            krx_sector = pd.concat([sector_stk, sector_ksq]).reset_index(drop=True)
            krx_sector['종목명'] = krx_sector['종목명'].str.strip()
            krx_sector['기준일'] = biz_day
            
            # 산업별 현황 수집
            gen_otp_data = {
                'searchType': '1', 'mktId': 'ALL', 'trdDd': biz_day,
                'csvxls_isNo': 'false', 'name': 'fileDown',
                'url': 'dbms/MDC/STAT/standard/MDCSTAT03501'
            }
            
            otp = self.session.post(gen_otp_url, gen_otp_data, headers=headers).text
            time.sleep(1)
            
            krx_ind = self.session.post(down_url, {'code': otp}, headers=headers)
            time.sleep(1)
            
            krx_ind = pd.read_csv(BytesIO(krx_ind.content), encoding='EUC-KR')
            krx_ind['종목명'] = krx_ind['종목명'].str.strip()
            krx_ind['기준일'] = biz_day
            
            # 데이터 병합
            diff = list(set(krx_sector['종목명']).symmetric_difference(set(krx_ind['종목명'])))
            kor_ticker = pd.merge(
                krx_sector, krx_ind,
                on=krx_sector.columns.intersection(krx_ind.columns).tolist(),
                how='outer'
            )
            
            # 종목 구분 설정
            kor_ticker['종목구분'] = np.where(
                kor_ticker['종목명'].str.contains('스팩|제[0-9]+호'), '스팩',
                np.where(
                    kor_ticker['종목코드'].str[-1:] != '0', '우선주',
                    np.where(
                        kor_ticker['종목명'].str.endswith('리츠'), '리츠',
                        np.where(kor_ticker['종목명'].isin(diff), '기타', '보통주')
                    )
                )
            )
            
            kor_ticker = kor_ticker.reset_index(drop=True)
            kor_ticker.columns = kor_ticker.columns.str.replace(' ', '')
            kor_ticker = kor_ticker[['종목코드', '종목명', '시장구분', '종가', '시가총액', '기준일', 'EPS', '선행EPS', 'BPS', '주당배당금', '종목구분']]
            kor_ticker['종목코드'] = kor_ticker['종목코드'].astype(str).str.zfill(6)
            # NaN, inf, -inf를 None으로 변환
            kor_ticker = kor_ticker.replace([np.inf, -np.inf], np.nan)
            kor_ticker = kor_ticker.where(pd.notnull(kor_ticker), None)
            
            # 파일 저장
            output_filename = os.path.join(self.config.raw_dir, f'kor_ticker_{biz_day}.csv')
            kor_ticker.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            # Supabase 업데이트
            # data = kor_ticker.to_dict(orient='records')
            # self.supabase.table('kor_ticker').upsert(data).execute()
            
            logger.info(f"종목 리스트 수집 완료: {len(kor_ticker)}개 종목")
            return kor_ticker
            
        except Exception as e:
            logger.error(f"종목 리스트 수집 실패: {e}")
            raise
    
    def fetch_sector_list(self, biz_day: str) -> pd.DataFrame:
        """섹터 정보 수집"""
        try:
            sector_codes = ['G25', 'G35', 'G50', 'G40', 'G10', 'G20', 'G55', 'G30', 'G15', 'G45']
            data_sector = []
            
            for sector_code in tqdm(sector_codes, desc="섹터 정보 수집"):
                url = f'http://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={biz_day}&sec_cd={sector_code}'
                
                for attempt in range(self.config.max_retries):
                    try:
                        response = self.session.get(url)
                        response.raise_for_status()
                        data = response.json()
                        data_pd = pd.json_normalize(data['list'])
                        data_sector.append(data_pd)
                        break
                    except Exception as e:
                        if attempt == self.config.max_retries - 1:
                            logger.error(f"섹터 {sector_code} 수집 실패: {e}")
                        else:
                            time.sleep(self.config.request_delay)
                            continue
                
                time.sleep(self.config.request_delay)
            
            kor_sector = pd.concat(data_sector, axis=0)
            kor_sector = kor_sector[['IDX_CD', 'CMP_CD', 'CMP_KOR', 'SEC_NM_KOR']]
            kor_sector['기준일'] = biz_day
            kor_sector['기준일'] = pd.to_datetime(kor_sector['기준일'])
            kor_sector['CMP_CD'] = kor_sector['CMP_CD'].astype(str).str.zfill(6)
            # NaN, inf, -inf를 None으로 변환
            kor_sector = kor_sector.replace([np.inf, -np.inf], np.nan)
            kor_sector = kor_sector.where(pd.notnull(kor_sector), None)
            
            # 파일 저장
            output_filename = os.path.join(self.config.raw_dir, f'kor_sector_{biz_day}.csv')
            kor_sector.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            # Supabase 업데이트
            # data = kor_sector.to_dict(orient='records')
            # self.supabase.table('kor_sector').upsert(data).execute()
            
            logger.info(f"섹터 정보 수집 완료: {len(kor_sector)}개 종목")
            return kor_sector
            
        except Exception as e:
            logger.error(f"섹터 정보 수집 실패: {e}")
            raise
    
    def fetch_price_data(self, biz_day: str, ticker_df: pd.DataFrame = None) -> pd.DataFrame:
        """주가 데이터 수집"""
        try:
            if ticker_df is None:
                ticker_df = pd.read_csv(
                    os.path.join(self.config.raw_dir, f'kor_ticker_{biz_day}.csv'),
                    dtype={'종목코드': str}
                )
            
            # 보통주만 선택
            ticker_list = ticker_df[ticker_df['종목구분'] == '보통주']['종목코드'].tolist()
            
            if self.config.test_mode:
                ticker_list = ticker_list[:self.config.test_count]
            
            all_prices = pd.DataFrame()
            error_list = []
            
            fr = (date.today() + relativedelta(years=-5)).strftime("%Y%m%d")
            to = date.today().strftime("%Y%m%d")
            
            for ticker in tqdm(ticker_list, desc="주가 데이터 수집"):
                for attempt in range(self.config.max_retries):
                    try:
                        url = f'https://fchart.stock.naver.com/siseJson.nhn?symbol={ticker}&requestType=1&startTime={fr}&endTime={to}&timeframe=day'
                        response = self.session.get(url)
                        response.raise_for_status()
                        
                        data_price = pd.read_csv(BytesIO(response.content))
                        price = data_price.iloc[:, 0:6]
                        price.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
                        price = price.dropna()
                        price['날짜'] = price['날짜'].str.extract('(\\d+)')
                        price['날짜'] = pd.to_datetime(price['날짜'])
                        price['종목코드'] = ticker
                        
                        all_prices = pd.concat([all_prices, price], ignore_index=True)
                        break
                        
                    except Exception as e:
                        if attempt == self.config.max_retries - 1:
                            logger.error(f"{ticker} 주가 데이터 수집 실패: {e}")
                            error_list.append(ticker)
                        else:
                            time.sleep(self.config.request_delay)
                            continue
                
                time.sleep(self.config.request_delay)
            
            all_prices['종목코드'] = all_prices['종목코드'].astype(str).str.zfill(6)
            # NaN, inf, -inf를 None으로 변환
            all_prices = all_prices.replace([np.inf, -np.inf], np.nan)
            all_prices = all_prices.where(pd.notnull(all_prices), None)
            
            # 파일 저장
            output_filename = os.path.join(self.config.raw_dir, f'kor_price_{biz_day}.csv')
            all_prices.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            # Supabase 업데이트
            # data = all_prices.to_dict(orient='records')
            # self.supabase.table('kor_price').upsert(data).execute()
            
            logger.info(f"주가 데이터 수집 완료: {len(all_prices)}개 레코드, 오류: {len(error_list)}개")
            if error_list:
                logger.warning(f"오류 발생 종목: {error_list}")
            
            return all_prices
            
        except Exception as e:
            logger.error(f"주가 데이터 수집 실패: {e}")
            raise
    
    def fetch_financial_statements(self, biz_day: str, ticker_df: pd.DataFrame = None) -> pd.DataFrame:
        """재무제표 데이터 수집"""
        try:
            if ticker_df is None:
                ticker_df = pd.read_csv(
                    os.path.join(self.config.raw_dir, f'kor_ticker_{biz_day}.csv'),
                    dtype={'종목코드': str}
                )
            
            ticker_list = ticker_df[ticker_df['종목구분'] == '보통주']['종목코드'].tolist()
            
            if self.config.test_mode:
                ticker_list = ticker_list[:self.config.test_count]
            
            all_fs_data = pd.DataFrame()
            error_list = []
            
            def clean_fs(df: pd.DataFrame, ticker: str, frequency: str) -> pd.DataFrame:
                """재무제표 데이터 클렌징"""
                df = df[~df.loc[:, ~df.columns.isin(['계정'])].isna().all(axis=1)]
                df = df.drop_duplicates(['계정'], keep='first')
                df = pd.melt(df, id_vars='계정', var_name='기준일', value_name='값')
                df = df[~pd.isnull(df['값'])]
                df['계정'] = df['계정'].replace({'계산에 참여한 계정 펼치기': ''}, regex=True)
                df['기준일'] = pd.to_datetime(df['기준일'], format='%Y/%m') + pd.tseries.offsets.MonthEnd()
                df['종목코드'] = ticker
                df['공시구분'] = frequency
                return df
            
            for ticker in tqdm(ticker_list, desc="재무제표 데이터 수집"):
                for attempt in range(self.config.max_retries):
                    try:
                        url = f'http://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A{ticker}'
                        data = pd.read_html(url, displayed_only=False)
                        
                        # 연간 데이터
                        data_fs_y = pd.concat([
                            data[0].iloc[:, ~data[0].columns.str.contains('전년동기')],
                            data[2], data[4]
                        ])
                        data_fs_y = data_fs_y.rename(columns={data_fs_y.columns[0]: "계정"})
                        
                        # 결산년 찾기
                        page_data = self.session.get(url)
                        page_data_html = BeautifulSoup(page_data.content, 'html.parser')
                        fiscal_data = page_data_html.select('div.corp_group1 > h2')
                        fiscal_data_text = fiscal_data[1].text
                        fiscal_data_text = re.findall('[0-9]+', fiscal_data_text)
                        
                        data_fs_y = data_fs_y.loc[:, (data_fs_y.columns == '계정') | 
                                                  (data_fs_y.columns.str[-2:].isin(fiscal_data_text))]
                        data_fs_y_clean = clean_fs(data_fs_y, ticker, 'y')
                        
                        # 분기 데이터
                        data_fs_q = pd.concat([
                            data[1].iloc[:, ~data[1].columns.str.contains('전년동기')],
                            data[3], data[5]
                        ])
                        data_fs_q = data_fs_q.rename(columns={data_fs_q.columns[0]: "계정"})
                        data_fs_q_clean = clean_fs(data_fs_q, ticker, 'q')
                        
                        # 데이터 병합
                        data_fs_bind = pd.concat([data_fs_y_clean, data_fs_q_clean])
                        all_fs_data = pd.concat([all_fs_data, data_fs_bind], ignore_index=True)
                        break
                        
                    except Exception as e:
                        if attempt == self.config.max_retries - 1:
                            logger.error(f"{ticker} 재무제표 수집 실패: {e}")
                            error_list.append(ticker)
                        else:
                            time.sleep(self.config.request_delay)
                            continue
                
                time.sleep(self.config.request_delay)
            
            all_fs_data['종목코드'] = all_fs_data['종목코드'].astype(str).str.zfill(6)
            # NaN, inf, -inf를 None으로 변환
            all_fs_data = all_fs_data.replace([np.inf, -np.inf], np.nan)
            all_fs_data = all_fs_data.where(pd.notnull(all_fs_data), None)
            
            # 파일 저장
            output_filename = os.path.join(self.config.raw_dir, f'kor_fs_{biz_day}.csv')
            all_fs_data.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            # Supabase 업데이트
            # data = all_fs_data.to_dict(orient='records')
            # self.supabase.table('kor_fs').upsert(data).execute()
            
            logger.info(f"재무제표 데이터 수집 완료: {len(all_fs_data)}개 레코드, 오류: {len(error_list)}개")
            if error_list:
                logger.warning(f"오류 발생 종목: {error_list}")
            
            return all_fs_data
            
        except Exception as e:
            logger.error(f"재무제표 데이터 수집 실패: {e}")
            raise
    
    def calculate_valuation_metrics(self, biz_day: str, fs_df: pd.DataFrame = None, ticker_df: pd.DataFrame = None) -> pd.DataFrame:
        """투자 지표 계산"""
        try:
            if fs_df is None:
                fs_df = pd.read_csv(
                    os.path.join(self.config.raw_dir, f'kor_fs_{biz_day}.csv'),
                    dtype={'종목코드': str}
                )
            
            if ticker_df is None:
                ticker_df = pd.read_csv(
                    os.path.join(self.config.raw_dir, f'kor_ticker_{biz_day}.csv'),
                    dtype={'종목코드': str}
                )
            
            # 분기 데이터에서 필요한 계정만 선택
            kor_fs = fs_df[
                (fs_df['공시구분'] == 'q') & 
                (fs_df['계정'].isin(['당기순이익', '자본', '영업활동으로인한현금흐름', '매출액']))
            ]
            
            # TTM 계산
            kor_fs = kor_fs.sort_values(['종목코드', '계정', '기준일'])
            kor_fs['ttm'] = kor_fs.groupby(['종목코드', '계정'], as_index=False)['값'].rolling(
                window=4, min_periods=4).sum()['값']
            
            # 자본은 평균값 사용
            kor_fs['ttm'] = np.where(kor_fs['계정'] == '자본', kor_fs['ttm'] / 4, kor_fs['ttm'])
            kor_fs = kor_fs.groupby(['계정', '종목코드']).tail(1)
            
            # 시가총액과 병합하여 지표 계산
            kor_fs_merge = kor_fs[['계정', '종목코드', 'ttm']].merge(
                ticker_df[['종목코드', '시가총액', '기준일']], on='종목코드')
            kor_fs_merge['시가총액'] = kor_fs_merge['시가총액'] / 100000000  # 억원 단위
            
            kor_fs_merge['value'] = kor_fs_merge['시가총액'] / kor_fs_merge['ttm']
            kor_fs_merge['value'] = kor_fs_merge['value'].round(4)
            
            # 지표 구분
            kor_fs_merge['지표'] = np.where(
                kor_fs_merge['계정'] == '매출액', 'PSR',
                np.where(
                    kor_fs_merge['계정'] == '영업활동으로인한현금흐름', 'PCR',
                    np.where(kor_fs_merge['계정'] == '자본', 'PBR',
                            np.where(kor_fs_merge['계정'] == '당기순이익', 'PER', None))
                )
            )
            
            kor_fs_merge.rename(columns={'value': '값'}, inplace=True)
            kor_fs_merge = kor_fs_merge[['종목코드', '기준일', '지표', '값']]
            
            # 배당수익률 계산
            ticker_df['값'] = ticker_df['주당배당금'] / ticker_df['종가']
            ticker_df['값'] = ticker_df['값'].round(4)
            ticker_df['지표'] = 'DY'
            dy_list = ticker_df[['종목코드', '기준일', '지표', '값']]
            dy_list = dy_list[dy_list['값'] != 0]
            
            # 모든 지표 데이터 병합
            all_value_data = pd.concat([kor_fs_merge, dy_list], ignore_index=True)
            all_value_data['종목코드'] = all_value_data['종목코드'].astype(str).str.zfill(6)
            # NaN, inf, -inf, np.nan을 None으로 변환
            all_value_data = all_value_data.replace([np.inf, -np.inf, np.nan], None)
            
            # 파일 저장
            output_filename = os.path.join(self.config.raw_dir, f'kor_value_{biz_day}.csv')
            all_value_data.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            # Supabase 업데이트
            # data = all_value_data.to_dict(orient='records')
            # self.supabase.table('kor_value').upsert(data).execute()
            
            logger.info(f"투자 지표 계산 완료: {len(all_value_data)}개 레코드")
            return all_value_data
            
        except Exception as e:
            logger.error(f"투자 지표 계산 실패: {e}")
            raise
    
    def process_all_data(self) -> Dict[str, pd.DataFrame]:
        """전체 데이터 처리"""
        try:
            logger.info("국내주식 데이터 처리 시작")
            
            # 영업일 수집
            biz_day = self.fetch_business_day()
            
            # 종목 리스트 수집
            ticker_df = self.fetch_ticker_list(biz_day)
            
            # 섹터 정보 수집
            sector_df = self.fetch_sector_list(biz_day)
            
            # 주가 데이터 수집
            price_df = self.fetch_price_data(biz_day, ticker_df)
            
            # 재무제표 데이터 수집
            fs_df = self.fetch_financial_statements(biz_day, ticker_df)
            
            # 투자 지표 계산
            value_df = self.calculate_valuation_metrics(biz_day, fs_df, ticker_df)
            
            logger.info("국내주식 데이터 처리 완료")
            
            return {
                'ticker': ticker_df,
                'sector': sector_df,
                'price': price_df,
                'financial_statements': fs_df,
                'valuation_metrics': value_df
            }
            
        except Exception as e:
            logger.error(f"데이터 처리 실패: {e}")
            raise

def main():
    """메인 실행 함수"""
    config = DataConfig(test_mode=True)  # 테스트 모드로 실행
    processor = KoreanStockDataProcessor(config)
    
    try:
        results = processor.process_all_data()
        print("데이터 처리 완료!")
        
        # 결과 요약 출력
        for key, df in results.items():
            print(f"{key}: {len(df)}개 레코드")
            
    except Exception as e:
        print(f"데이터 처리 실패: {e}")

if __name__ == "__main__":
    main() 