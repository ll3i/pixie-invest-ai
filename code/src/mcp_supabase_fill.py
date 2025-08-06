import pandas as pd
from src.korean_stock_data_processor import KoreanStockDataProcessor, DataConfig
from src.stock_evaluator import StockEvaluator
from src.db_client import get_supabase_client
import numpy as np

def clean_value(v):
    if isinstance(v, float):
        if v is not None and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    elif isinstance(v, str):
        if v.lower() in ['nan', 'inf', '-inf', 'none']:
            return None
        return v
    elif isinstance(v, dict):
        return {k: clean_value(val) for k, val in v.items()}
    elif isinstance(v, list):
        return [clean_value(item) for item in v]
    return v

def upsert_to_supabase(table_name, df):
    supabase = get_supabase_client()
    # 모든 컬럼을 str로 변환 후 nan/inf/None 문자열도 None으로 치환
    df = df.astype(str)
    df = df.replace(['nan', 'inf', '-inf', 'None'], None)
    data = df.to_dict(orient='records')
    data = [clean_value(row) for row in data]
    # 디버깅: 문제값이 남아있는지 확인
    for i, row in enumerate(data):
        for k, v in row.items():
            if isinstance(v, float) and (v is not None) and (np.isnan(v) or np.isinf(v)):
                print(f"[DEBUG] {table_name} row {i} key {k} has bad value: {v}")
    if not data:
        print(f"{table_name}: 업서트할 데이터가 없습니다.")
        return
    supabase.table(table_name).upsert(data).execute()
    print(f"Supabase에 {table_name} 업서트 완료 ({len(data)} rows)")

def main():
    # 1. 데이터 수집/가공
    config = DataConfig(test_mode=False)
    processor = KoreanStockDataProcessor(config)
    results = processor.process_all_data()  # dict: ticker, sector, price, financial_statements, valuation_metrics

    # 2. 각 테이블별로 업서트
    upsert_to_supabase('kor_ticker', results['ticker'])
    upsert_to_supabase('kor_sector', results['sector'])
    upsert_to_supabase('kor_price', results['price'])
    upsert_to_supabase('kor_fs', results['financial_statements'])
    upsert_to_supabase('kor_value', results['valuation_metrics'])

    # 3. 평가 결과도 업서트
    evaluator = StockEvaluator()
    eval_df = evaluator.evaluate_all_stocks(
        results['financial_statements'],
        results['price'],
        results['ticker'],
        results['valuation_metrics']
    )
    upsert_to_supabase('stock_evaluation_results', eval_df)

if __name__ == "__main__":
    main() 