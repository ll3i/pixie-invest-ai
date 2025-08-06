#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
금융 데이터 수집 모듈
- 외부 금융 데이터 수집 (KRX, Naver Finance, FnGuide, WISE 등)
- 데이터 정제 및 저장
- docs/data_processing.ipynb 기반 구현
"""

import os
import requests
import pandas as pd
import numpy as np
import time
import re
import csv
from bs4 import BeautifulSoup
from io import BytesIO
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

class DataCollector:
    """금융 데이터 수집 클래스 - Jupyter notebook 로직 기반"""
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.script_dir), "data")
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")
        
        # 디렉토리 초기화
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # KRX 헤더 설정
        self.headers = {
            'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        }
        
        # URL 설정
        self.gen_otp_url = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
        self.down_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
    
    def get_business_day(self):
        """최신 영업일 가져오기"""
        try:
            url = 'https://finance.naver.com/sise/sise_deposit.nhn'
            data = requests.get(url)
            data_html = BeautifulSoup(data.content, features='html.parser')
            parse_day = data_html.select_one('div.subtop_sise_graph2 > ul.subtop_chart_note > li > span.tah').text
            
            biz_day = re.findall('[0-9]+', parse_day)
            biz_day = ''.join(biz_day)
            
            return biz_day
        except Exception as e:
            print(f"영업일 가져오기 실패: {e}")
            # 실패시 오늘 날짜 반환
            return datetime.now().strftime('%Y%m%d')
    
    def collect_stock_tickers(self):
        """주식 종목 코드 수집 - KRX 데이터 기반"""
        try:
            print("주식 종목 코드 수집 중...")
            bday = self.get_business_day()
            
            # 코스피 데이터 수집
            gen_otp_stk = {
                'mktId': 'STK',
                'trdDd': bday,
                'money': '1',
                'csvxls_isNo': 'false',
                'name': 'fileDown',
                'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
            }
            
            otp_stk = requests.post(self.gen_otp_url, gen_otp_stk, headers=self.headers).text
            time.sleep(1)
            
            down_sector_stk = requests.post(self.down_url, {'code': otp_stk}, headers=self.headers)
            time.sleep(1)
            
            sector_stk = pd.read_csv(BytesIO(down_sector_stk.content), encoding='EUC-KR')
            
            # 코스닥 데이터 수집
            gen_otp_ksq = {
                'mktId': 'KSQ',
                'trdDd': bday,
                'money': '1',
                'csvxls_isNo': 'false',
                'name': 'fileDown',
                'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
            }
            
            otp_ksq = requests.post(self.gen_otp_url, gen_otp_ksq, headers=self.headers).text
            time.sleep(1)
            
            down_sector_ksq = requests.post(self.down_url, {'code': otp_ksq}, headers=self.headers)
            time.sleep(1)
            
            sector_ksq = pd.read_csv(BytesIO(down_sector_ksq.content), encoding='EUC-KR')
            
            # 데이터 병합
            krx_sector = pd.concat([sector_stk, sector_ksq]).reset_index(drop=True)
            krx_sector['종목명'] = krx_sector['종목명'].str.strip()
            krx_sector['기준일'] = bday
            
            # 산업별 현황 데이터 수집
            gen_otp_data = {
                'searchType': '1',
                'mktId': 'ALL',
                'trdDd': bday,
                'csvxls_isNo': 'false',
                'name': 'fileDown',
                'url': 'dbms/MDC/STAT/standard/MDCSTAT03501'
            }
            
            otp = requests.post(self.gen_otp_url, gen_otp_data, headers=self.headers).text
            time.sleep(1)
            
            krx_ind = requests.post(self.down_url, {'code': otp}, headers=self.headers)
            time.sleep(1)
            krx_ind = pd.read_csv(BytesIO(krx_ind.content), encoding='EUC-KR')
            
            krx_ind['종목명'] = krx_ind['종목명'].str.strip()
            krx_ind['기준일'] = bday
            
            # 차집합 확인
            diff = list(set(krx_sector['종목명']).symmetric_difference(set(krx_ind['종목명'])))
            
            # 데이터 병합
            kor_ticker = pd.merge(krx_sector,
                                krx_ind,
                                on=krx_sector.columns.intersection(krx_ind.columns).tolist(),
                                how='outer')
            
            # 종목 구분 작업
            kor_ticker['종목구분'] = np.where(kor_ticker['종목명'].str.contains('스팩|제[0-9]+호'), '스팩',
                                        np.where(kor_ticker['종목코드'].str[-1:] != '0', '우선주',
                                                np.where(kor_ticker['종목명'].str.endswith('리츠'), '리츠',
                                                            np.where(kor_ticker['종목명'].isin(diff), '기타',
                                                            '보통주'))))
            kor_ticker = kor_ticker.reset_index(drop=True)
            
            # 공백 제거
            kor_ticker.columns = kor_ticker.columns.str.replace(' ', '')
            
            # 필요한 정보 슬라이싱
            kor_ticker = kor_ticker[['종목코드', '종목명', '시장구분', '종가',
                                    '시가총액', '기준일', 'EPS', '선행EPS', 'BPS', '주당배당금', '종목구분']]
            
            kor_ticker['종목코드'] = kor_ticker['종목코드'].astype(str).str.zfill(6)
            
            # 파일 저장
            output_filename = os.path.join(self.raw_data_dir, f'kor_ticker_{bday}.csv')
            kor_ticker.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            print(f"종목 코드 수집 완료: {len(kor_ticker)}개 종목, 저장 경로: {output_filename}")
            return kor_ticker
        except Exception as e:
            print(f"종목 코드 수집 실패: {e}")
            return None
    
    def collect_sector_data(self):
        """섹터별 종목 리스트 수집"""
        try:
            print("섹터별 종목 리스트 수집 중...")
            bday = self.get_business_day()
            
            sector_code = [
                'G25', 'G35', 'G50', 'G40', 'G10', 'G20', 'G55', 'G30', 'G15', 'G45'
            ]
            data_sector = []
            
            for i in tqdm(sector_code):
                url = f'http://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={bday}&sec_cd={i}'
                data = requests.get(url).json()
                data_pd = pd.json_normalize(data['list'])
                data_sector.append(data_pd)
                time.sleep(2)
            
            kor_sector = pd.concat(data_sector, axis=0)
            kor_sector = kor_sector[['IDX_CD', 'CMP_CD', 'CMP_KOR', 'SEC_NM_KOR']]
            kor_sector['기준일'] = bday
            kor_sector['기준일'] = pd.to_datetime(kor_sector['기준일'])
            
            # CMP_CD(종목코드)를 문자열로 변환하고 6자리로 맞춤
            kor_sector['CMP_CD'] = kor_sector['CMP_CD'].astype(str).str.zfill(6)
            
            # CSV 파일로 저장
            output_filename = os.path.join(self.raw_data_dir, f'kor_sector_{bday}.csv')
            kor_sector.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            print(f'섹터 리스트 수집 완료: {output_filename}')
            return kor_sector
        except Exception as e:
            print(f"섹터 데이터 수집 실패: {e}")
            return None
    
    def collect_stock_prices(self, ticker_limit=None):
        """주식 가격 데이터 수집 (5년간)"""
        try:
            print("주식 가격 데이터 수집 중...")
            bday = self.get_business_day()
            
            # 종목 리스트 로드
            ticker_file = os.path.join(self.raw_data_dir, f'kor_ticker_{bday}.csv')
            if not os.path.exists(ticker_file):
                print("종목 코드 파일이 없습니다. 종목 코드 수집을 먼저 실행하세요.")
                return None
            
            ticker_list = pd.read_csv(ticker_file, dtype={'종목코드': str})
            ticker_list = ticker_list[ticker_list['종목구분'] == '보통주']
            
            # 테스트를 위해 제한 설정
            if ticker_limit:
                ticker_list = ticker_list[:ticker_limit]
            
            error_list = []
            all_prices = pd.DataFrame()
            
            # 시작일과 종료일 설정
            fr = (date.today() + relativedelta(years=-5)).strftime("%Y%m%d")
            to = date.today().strftime("%Y%m%d")
            
            # 전종목 주가 다운로드
            for i in tqdm(range(0, len(ticker_list))):
                ticker = ticker_list['종목코드'].iloc[i]
                
                try:
                    # URL 생성
                    url = f'https://fchart.stock.naver.com/siseJson.nhn?symbol={ticker}&requestType=1&startTime={fr}&endTime={to}&timeframe=day'
                    
                    # 데이터 다운로드
                    data = requests.get(url).content
                    data_price = pd.read_csv(BytesIO(data))
                    
                    # 데이터 클렌징
                    price = data_price.iloc[:, 0:6]
                    price.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
                    price = price.dropna()
                    price['날짜'] = price['날짜'].str.extract('(\d+)')
                    price['날짜'] = pd.to_datetime(price['날짜'])
                    price['종목코드'] = ticker
                    
                    # 전체 가격 데이터에 추가
                    all_prices = pd.concat([all_prices, price], ignore_index=True)
                    
                    print(f'\n{ticker} 처리 완료.')
                    
                except Exception as e:
                    print(f"\n{ticker} 데이터 수집 오류: {e}")
                    error_list.append(ticker)
                
                print(f'현재까지 {len(error_list)}개 오류 발생.\n')
                # 타임 슬립
                time.sleep(2)
            
            # 종목코드를 문자열로 변환하고 6자리로 맞춤
            all_prices['종목코드'] = all_prices['종목코드'].astype(str).str.zfill(6)
            
            # CSV 파일로 저장
            output_filename = os.path.join(self.raw_data_dir, f'kor_price_{bday}.csv')
            all_prices.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            print(f'주가 데이터 수집 완료: {output_filename}')
            print(f'총 {len(error_list)}개 오류 발생.')
            if error_list:
                print("오류 종목:", error_list)
                
            return all_prices
        except Exception as e:
            print(f"주가 데이터 수집 실패: {e}")
            return None
            
    def clean_financial_statement(self, df, ticker, frequency):
        """재무제표 클렌징 함수"""
        df = df[~df.loc[:, ~df.columns.isin(['계정'])].isna().all(axis=1)]
        df = df.drop_duplicates(['계정'], keep='first')
        df = pd.melt(df, id_vars='계정', var_name='기준일', value_name='값')
        df = df[~pd.isnull(df['값'])]
        df['계정'] = df['계정'].replace({'계산에 참여한 계정 펼치기': ''}, regex=True)
        df['기준일'] = pd.to_datetime(df['기준일'], format='%Y/%m') + pd.tseries.offsets.MonthEnd()
        df['종목코드'] = ticker
        df['공시구분'] = frequency
        return df
    
    def collect_financial_statements(self, ticker_limit=None):
        """재무제표 데이터 수집 - FnGuide 기반"""
        try:
            print("재무제표 데이터 수집 중...")
            bday = self.get_business_day()
            
            # 종목 리스트 로드
            ticker_file = os.path.join(self.raw_data_dir, f'kor_ticker_{bday}.csv')
            if not os.path.exists(ticker_file):
                print("종목 코드 파일이 없습니다. 종목 코드 수집을 먼저 실행하세요.")
                return None
            
            ticker_list = pd.read_csv(ticker_file, dtype={'종목코드': str})
            ticker_list = ticker_list[ticker_list['종목구분'] == '보통주']
            
            # 테스트를 위해 제한 설정
            if ticker_limit:
                ticker_list = ticker_list[:ticker_limit]
            
            error_list = []
            all_fs_data = pd.DataFrame()
            
            # 전종목 재무제표 다운로드
            for i in tqdm(range(0, len(ticker_list))):
                ticker = ticker_list['종목코드'].iloc[i]
                
                try:
                    # URL 생성
                    url = f'http://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A{ticker}'
                    
                    # 데이터 받아오기
                    data = pd.read_html(url, displayed_only=False)
                    
                    # 연간 데이터
                    data_fs_y = pd.concat([
                        data[0].iloc[:, ~data[0].columns.str.contains('전년동기')],
                        data[2],
                        data[4]
                    ])
                    data_fs_y = data_fs_y.rename(columns={data_fs_y.columns[0]: "계정"})
                    
                    # 결산년 찾기
                    page_data = requests.get(url)
                    page_data_html = BeautifulSoup(page_data.content, 'html.parser')
                    
                    fiscal_data = page_data_html.select('div.corp_group1 > h2')
                    fiscal_data_text = fiscal_data[1].text
                    fiscal_data_text = re.findall('[0-9]+', fiscal_data_text)
                    
                    # 결산년에 해당하는 계정만 남기기
                    data_fs_y = data_fs_y.loc[:, (data_fs_y.columns == '계정') | 
                                                 (data_fs_y.columns.str[-2:].isin(fiscal_data_text))]
                    
                    # 클렌징
                    data_fs_y_clean = self.clean_financial_statement(data_fs_y, ticker, 'y')
                    
                    # 분기 데이터
                    data_fs_q = pd.concat([
                        data[1].iloc[:, ~data[1].columns.str.contains('전년동기')],
                        data[3],
                        data[5]
                    ])
                    data_fs_q = data_fs_q.rename(columns={data_fs_q.columns[0]: "계정"})
                    
                    data_fs_q_clean = self.clean_financial_statement(data_fs_q, ticker, 'q')
                    
                    # 두개 합치기
                    data_fs_bind = pd.concat([data_fs_y_clean, data_fs_q_clean])
                    
                    # 전체 재무제표 데이터에 추가
                    all_fs_data = pd.concat([all_fs_data, data_fs_bind], ignore_index=True)
                    
                    print(f'\n{ticker} 처리 완료.')
                    
                except Exception as e:
                    print(f"\n{ticker} 데이터 수집 오류: {e}")
                    error_list.append(ticker)
                
                print(f'현재까지 {len(error_list)}개 오류 발생.\n')
                # 타임슬립 적용
                time.sleep(2)
            
            # 종목코드를 문자열로 변환하고 6자리로 맞춤
            all_fs_data['종목코드'] = all_fs_data['종목코드'].astype(str).str.zfill(6)
            
            # CSV 파일로 저장
            output_filename = os.path.join(self.raw_data_dir, f'kor_fs_{bday}.csv')
            all_fs_data.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            print(f'재무제표 데이터 수집 완료: {output_filename}')
            print(f'총 {len(error_list)}개 오류 발생.')
            if error_list:
                print("오류 종목:", error_list)
                
            return all_fs_data
        except Exception as e:
            print(f"재무제표 데이터 수집 실패: {e}")
            return None
    
    def collect_valuation_metrics(self):
        """주식 가치평가 지표 수집 - TTM 기반 계산"""
        try:
            print("주식 가치평가 지표 수집 중...")
            bday = self.get_business_day()
            
            # 재무제표 데이터 로드
            fs_file = os.path.join(self.raw_data_dir, f'kor_fs_{bday}.csv')
            if not os.path.exists(fs_file):
                print("재무제표 파일이 없습니다. 재무제표 수집을 먼저 실행하세요.")
                return None
                
            kor_fs = pd.read_csv(fs_file, dtype={'종목코드': str})
            kor_fs = kor_fs[
                (kor_fs['공시구분'] == 'q') & 
                (kor_fs['계정'].isin(['당기순이익', '자본', '영업활동으로인한현금흐름', '매출액']))
            ]
            
            # 티커 리스트 로드
            ticker_file = os.path.join(self.raw_data_dir, f'kor_ticker_{bday}.csv')
            ticker_list = pd.read_csv(ticker_file, dtype={'종목코드': str})
            ticker_list = ticker_list[ticker_list['종목구분'] == '보통주']
            
            # TTM 구하기
            kor_fs = kor_fs.sort_values(['종목코드', '계정', '기준일'])
            kor_fs['ttm'] = kor_fs.groupby(['종목코드', '계정'], as_index=False)['값'].rolling(
                window=4, min_periods=4).sum()['값']
            
            # 자본은 평균 구하기
            kor_fs['ttm'] = np.where(kor_fs['계정'] == '자본', kor_fs['ttm'] / 4, kor_fs['ttm'])
            kor_fs = kor_fs.groupby(['계정', '종목코드']).tail(1)
            
            kor_fs_merge = kor_fs[['계정', '종목코드', 'ttm']].merge(
                ticker_list[['종목코드', '시가총액', '기준일']], on='종목코드')
            kor_fs_merge['시가총액'] = kor_fs_merge['시가총액'] / 100000000
            
            kor_fs_merge['value'] = kor_fs_merge['시가총액'] / kor_fs_merge['ttm']
            kor_fs_merge['value'] = kor_fs_merge['value'].round(4)
            kor_fs_merge['지표'] = np.where(
                kor_fs_merge['계정'] == '매출액', 'PSR',
                np.where(
                    kor_fs_merge['계정'] == '영업활동으로인한현금흐름', 'PCR',
                    np.where(kor_fs_merge['계정'] == '자본', 'PBR',
                            np.where(kor_fs_merge['계정'] == '당기순이익', 'PER', None))))
            
            kor_fs_merge.rename(columns={'value': '값'}, inplace=True)
            kor_fs_merge = kor_fs_merge[['종목코드', '기준일', '지표', '값']]
            kor_fs_merge = kor_fs_merge.replace([np.inf, -np.inf, np.nan], None)
            
            # 배당수익률 계산
            ticker_list['값'] = ticker_list['주당배당금'] / ticker_list['종가']
            ticker_list['값'] = ticker_list['값'].round(4)
            ticker_list['지표'] = 'DY'
            dy_list = ticker_list[['종목코드', '기준일', '지표', '값']]
            dy_list = dy_list.replace([np.inf, -np.inf, np.nan], None)
            dy_list = dy_list[dy_list['값'] != 0]
            
            # 모든 밸류 데이터 합치기
            all_value_data = pd.concat([kor_fs_merge, dy_list], ignore_index=True)
            
            # 종목코드를 문자열로 변환하고 6자리로 맞춤
            all_value_data['종목코드'] = all_value_data['종목코드'].astype(str).str.zfill(6)
            
            # CSV 파일로 저장
            output_filename = os.path.join(self.raw_data_dir, f'kor_value_{bday}.csv')
            all_value_data.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            print(f'가치평가 지표 계산 완료: {output_filename}')
            return all_value_data
        except Exception as e:
            print(f"가치평가 지표 수집 실패: {e}")
            return None
    
    
    def run_all_collectors(self):
        """모든 데이터 수집 함수 실행"""
        print("모든 데이터 수집 시작...")
        
        # 1. 종목 코드 수집
        ticker_df = self.collect_stock_tickers()
        
        if ticker_df is not None:
            # 2. 섹터 리스트 수집
            self.collect_sector_data()
            
            # 3. 주식 가격 데이터 수집 (테스트로 30개만)
            self.collect_stock_prices(ticker_limit=30)
            
            # 4. 재무제표 데이터 수집 (테스트로 30개만)
            self.collect_financial_statements(ticker_limit=30)
            
            # 5. 가치평가 지표 수집
            self.collect_valuation_metrics()
        
        print("모든 데이터 수집 완료")

    def collect_us_stock_tickers(self):
        """
        미국 우량주 20개 티커 리스트 수집 및 저장
        - 주요 기술주, 금융주, 소비재주 등 우량주 20개 선별
        - 결과: data/raw/us_ticker_YYYYMMDD.csv 저장
        - 주요 컬럼: Ticker, Name, Market, Sector
        """
        try:
            import yfinance as yf
            print("미국 우량주 20개 티커 리스트 수집 중...")
            
            # 우량주 20개 선별 (시가총액, 거래량, 안정성 고려)
            top_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'JNJ', 'JPM',
                'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'NFLX', 'ADBE', 'CRM'
            ]
            
            records = []
            for i, ticker in enumerate(top_stocks):
                try:
                    info = yf.Ticker(ticker).info
                    name = info.get('shortName', ticker)
                    market = info.get('exchange', 'NASDAQ')
                    sector = info.get('sector', 'Technology')
                    records.append({'Ticker': ticker, 'Name': name, 'Market': market, 'Sector': sector})
                    print(f"{i+1}/{len(top_stocks)}: {ticker} - {name}")
                    time.sleep(0.1)
                except Exception as e:
                    print(f"  - {ticker} 정보 조회 실패: {e}")
                    continue
            
            import pandas as pd
            df = pd.DataFrame(records)
            today = datetime.now().strftime('%Y%m%d')
            output_file = os.path.join(self.raw_data_dir, f"us_ticker_{today}.csv")
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"미국 우량주 20개 티커 리스트 수집 완료: {len(df)}개, 저장 경로: {output_file}")
            return df
        except Exception as e:
            print(f"미국 우량주 티커 리스트 수집 실패: {e}")
            return None

    def collect_us_stock_prices(self, tickers=None, days=1095):
        """
        미국 우량주 20개 일별 가격 데이터 수집 (yfinance) - 3년간
        - tickers: 우량주 20개 티커 리스트(없으면 자동으로 선별된 우량주 사용)
        - days: 최근 n일치(기본 3년)
        - 결과: data/raw/us_price_YYYYMMDD.csv 저장
        - 주요 컬럼: Date, Open, High, Low, Close, Volume, Ticker
        """
        try:
            import yfinance as yf
            print("미국 우량주 20개 가격 데이터 수집 중...")
            
            if tickers is None:
                # 우량주 20개 선별
                top_stocks = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'JNJ', 'JPM',
                    'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'NFLX', 'ADBE', 'CRM'
                ]
                tickers = top_stocks
                print(f"미국 우량주 20개 티커 로드됨")
            
            end_date = datetime.now() - timedelta(days=1)  # 어제까지
            start_date = end_date - timedelta(days=days)
            print(f"데이터 수집 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
            
            all_prices = []
            total_tickers = len(tickers)
            
            for i, ticker in enumerate(tickers):
                try:
                    print(f"[{i+1}/{total_tickers}] {ticker} 가격 데이터 수집 중...")
                    
                    # yfinance로 데이터 수집
                    stock = yf.Ticker(ticker)
                    df = stock.history(
                        start=start_date.strftime('%Y-%m-%d'), 
                        end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                        interval='1d'
                    )
                    
                    if not df.empty:
                        df = df.reset_index()
                        df['Ticker'] = ticker
                        # 컬럼 순서 정리
                        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
                        # 결측값 제거
                        df = df.dropna()
                        all_prices.append(df)
                        print(f"  - {ticker}: {len(df)}개 데이터 수집 완료")
                    else:
                        print(f"  - {ticker}: 데이터 없음")
                    
                    time.sleep(0.1)  # API 제한 고려
                except Exception as e:
                    print(f"  - {ticker} 데이터 수집 실패: {e}")
                    continue
            
            if all_prices:
                result_df = pd.concat(all_prices, ignore_index=True)
                result_df = result_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
                
                # 파일 저장
                today = datetime.now().strftime('%Y%m%d')
                output_file = os.path.join(self.raw_data_dir, f"us_price_{today}.csv")
                result_df.to_csv(output_file, index=False, encoding='utf-8')
                
                print(f"S&P 500 가격 데이터 수집 완료: {len(result_df)}개 레코드 ({len(tickers[:total_tickers])}개 종목, {days}일간), 저장 경로: {output_file}")
                return result_df
            else:
                print("수집된 S&P 500 가격 데이터가 없습니다.")
                return None
        except Exception as e:
            print(f"S&P 500 가격 데이터 수집 실패: {e}")
            return None

    def collect_news_data(self, keywords=None, days=30):
        """
        뉴스 데이터 수집 (Alpha Vantage News API 또는 무료 RSS)
        - keywords: 검색 키워드 리스트
        - days: 최근 n일치 뉴스
        """
        try:
            print("뉴스 데이터 수집 중...")
            
            if keywords is None:
                keywords = ['stock market', 'economy', 'finance', '주식', '경제', '금융']
            
            # 1. 먼저 네이버 MCP로 뉴스 수집 시도
            try:
                from .naver_news_mcp_collector import NaverNewsMCPCollector
                mcp_collector = NaverNewsMCPCollector()
                mcp_news = mcp_collector.collect_news(days=days)
                
                if mcp_news is not None and not mcp_news.empty:
                    print(f"네이버 MCP로 {len(mcp_news)}개 뉴스 수집 성공")
                    # MCP 뉴스가 있으면 이를 메인으로 사용
                    today = datetime.now().strftime('%Y%m%d')
                    output_file = os.path.join(self.raw_data_dir, f"news_{today}.csv")
                    
                    # 기존 RSS 뉴스도 추가로 수집하여 병합
                    rss_news = self._collect_rss_news(days, keywords)
                    if rss_news is not None and not rss_news.empty:
                        # MCP 뉴스와 RSS 뉴스 병합
                        all_news = pd.concat([mcp_news, rss_news], ignore_index=True)
                        all_news = all_news.drop_duplicates(subset=['title'], keep='first')
                        all_news = all_news.sort_values('published', ascending=False)
                        all_news.to_csv(output_file, index=False, encoding='utf-8')
                        print(f"MCP + RSS 통합 뉴스: {len(all_news)}개")
                        return all_news
                    else:
                        mcp_news.to_csv(output_file, index=False, encoding='utf-8')
                        return mcp_news
                        
            except Exception as e:
                print(f"MCP 뉴스 수집 실패: {e}")
                print("RSS 피드로 대체 수집합니다...")
            
            # 2. MCP 실패시 RSS 피드로 뉴스 수집
            return self._collect_rss_news(days, keywords)
            
        except Exception as e:
            print(f"뉴스 데이터 수집 실패: {e}")
            return None
    
    def _collect_rss_news(self, days=1, keywords=None):
        """RSS 피드를 사용한 뉴스 수집"""
        try:
            # RSS 피드를 사용한 뉴스 수집 (무료)
            import feedparser
            
            rss_feeds = [
                # 국내 뉴스 RSS 피드
                'https://www.sedaily.com/RSS/Stock.xml',  # 서울경제 증권
                'https://www.mk.co.kr/rss/40300001/',  # 매일경제 증권
                'https://rss.hankyung.com/feed/finance.xml',  # 한국경제 금융
                'http://rss.edaily.co.kr/stock_news.xml',  # 이데일리 증권
                'https://www.fnnews.com/rss/r20/fn_realnews_stock.xml',  # 파이낸셜뉴스 증권
                
                # 해외 뉴스 RSS 피드 (백업용)
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://feeds.bloomberg.com/markets/news.rss'
            ]
            
            all_news = []
            for feed_url in rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries:
                        pub_date = datetime.now()
                        
                        # 다양한 날짜 형식 처리
                        date_formats = [
                            '%a, %d %b %Y %H:%M:%S %z',
                            '%a, %d %b %Y %H:%M:%S %Z',
                            '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%dT%H:%M:%S',
                            '%Y-%m-%dT%H:%M:%S%z',
                            '%Y-%m-%dT%H:%M:%S.%f%z'
                        ]
                        
                        published_str = entry.get('published', entry.get('pubDate', ''))
                        if published_str:
                            for date_format in date_formats:
                                try:
                                    pub_date = datetime.strptime(published_str.replace('T', ' ').split('+')[0].split('.')[0], 
                                                               date_format.replace('T', ' ').split('%z')[0].split('%Z')[0].rstrip())
                                    break
                                except:
                                    continue
                        
                        # 최근 n일 이내의 뉴스만
                        if (datetime.now() - pub_date).days <= days:
                            # 한국 뉴스인지 구분
                            is_korean = any(korean_domain in feed_url for korean_domain in 
                                          ['sedaily.com', 'mk.co.kr', 'hankyung.com', 'edaily.co.kr', 'fnnews.com'])
                            
                            # 소스 이름 추출
                            source_name = 'Unknown'
                            if 'sedaily.com' in feed_url:
                                source_name = '서울경제'
                            elif 'mk.co.kr' in feed_url:
                                source_name = '매일경제'
                            elif 'hankyung.com' in feed_url:
                                source_name = '한국경제'
                            elif 'edaily.co.kr' in feed_url:
                                source_name = '이데일리'
                            elif 'fnnews.com' in feed_url:
                                source_name = '파이낸셜뉴스'
                            elif 'yahoo.com' in feed_url:
                                source_name = 'Yahoo Finance'
                            elif 'bloomberg.com' in feed_url:
                                source_name = 'Bloomberg'
                            
                            news_item = {
                                'title': entry.title,
                                'summary': entry.get('summary', entry.get('description', ''))[:500],
                                'link': entry.link,
                                'published': pub_date.isoformat(),
                                'source': source_name,
                                'source_url': feed_url,
                                'is_korean': is_korean
                            }
                            all_news.append(news_item)
                except Exception as e:
                    print(f"RSS 피드 오류 ({feed_url}): {e}")
                    continue
            
            if all_news:
                # 중복 제거 (제목 기준)
                seen_titles = set()
                unique_news = []
                for news in all_news:
                    if news['title'] not in seen_titles:
                        seen_titles.add(news['title'])
                        unique_news.append(news)
                
                # 데이터프레임으로 변환
                news_df = pd.DataFrame(unique_news)
                
                # RSS 뉴스는 MCP와 병합을 위해 반환만 하고 저장하지 않음
                print(f"RSS 뉴스 수집 완료: {len(news_df)}개 기사")
                return news_df
            else:
                print("수집된 뉴스 데이터가 없습니다.")
                return None
        except Exception as e:
            print(f"뉴스 데이터 수집 실패: {e}")
            return None

    def run_daily_update(self):
        """매일 자동 실행되는 데이터 업데이트"""
        print("=== 일일 데이터 업데이트 시작 ===")
        
        try:
            # 1. 최신 주식 가격 (1일치)
            print("1. 국내 주식 가격 업데이트...")
            self.collect_stock_prices(days=1)
            
            # 2. 미국 주식 가격 (1일치)
            print("2. 미국 주식 가격 업데이트...")
            self.collect_us_stock_prices(days=1)
            
            # 3. 뉴스 데이터 (최근 3일)
            print("3. 뉴스 데이터 업데이트...")
            self.collect_news_data(days=3)
            
            # 4. 종목 코드 업데이트 (주 1회, 월요일에만)
            if datetime.now().weekday() == 0:  # 월요일
                print("4. 종목 코드 업데이트...")
                self.collect_stock_tickers()
                self.collect_us_stock_tickers()
            
            print("=== 일일 데이터 업데이트 완료 ===")
            return True
        except Exception as e:
            print(f"일일 데이터 업데이트 실패: {e}")
            return False
    
    def run_initial_data_setup(self):
        """최초 설정 시 3년간 데이터 수집"""
        print("=== 초기 데이터 수집 시작 (3년간) ===")
        
        try:
            # 1. 종목 코드 수집
            print("1. 국내 종목 코드 수집...")
            self.collect_stock_tickers()
            
            print("2. 미국 종목 코드 수집...")
            self.collect_us_stock_tickers()
            
            # 2. 3년간 주식 가격 데이터
            print("3. 국내 주식 3년간 가격 데이터 수집...")
            self.collect_stock_prices(days=1095)  # 3년
            
            print("4. 미국 주식 3년간 가격 데이터 수집...")
            self.collect_us_stock_prices(days=1095)  # 3년
            
            # 3. 재무제표 및 기타 데이터
            print("5. 재무제표 데이터 수집...")
            self.collect_financial_statements()
            
            print("6. 가치평가 지표 수집...")
            self.collect_valuation_metrics()
            
            print("7. 섹터 데이터 수집...")
            self.collect_sector_data()
            
            # 4. 최근 뉴스
            print("8. 뉴스 데이터 수집...")
            self.collect_news_data(days=30)
            
            print("=== 초기 데이터 수집 완료 ===")
            return True
        except Exception as e:
            print(f"초기 데이터 수집 실패: {e}")
            return False

# 모듈 테스트용 코드
if __name__ == "__main__":
    collector = DataCollector()
    collector.run_all_collectors() 