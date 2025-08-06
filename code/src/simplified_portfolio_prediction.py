# simplified_portfolio_prediction.py
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
import os
import glob
from user_profile_analyzer import UserProfileAnalyzer
from db_client import get_supabase_client
from datetime import datetime

def load_user_profile():
    """사용자 투자 성향 프로필을 로드합니다."""
    try:
        with open('analysis_results.json', 'r', encoding='utf-8') as f:
            profile = json.load(f)
        return profile
    except Exception as e:
        print(f"Error loading user profile: {str(e)}")
        # 기본 프로필 반환
        return {
            "risk_tolerance_analysis": "위험 감수성 점수가 -2.0으로 중간 수준입니다.",
            "investment_time_horizon_analysis": "투자 시간 범위 점수가 0.0으로 중간 수준입니다."
        }

def get_risk_level(profile):
    """사용자 위험 감수성을 기반으로 위험 수준을 결정합니다."""
    risk_tolerance = profile.get('risk_tolerance_analysis', '')
    if '매우 낮' in risk_tolerance or '낮' in risk_tolerance:
        return 'low'
    elif '중간' in risk_tolerance:
        return 'medium'
    else:
        return 'high'

def get_investment_horizon(profile):
    """사용자 투자 시간 범위를 결정합니다."""
    time_horizon = profile.get('investment_time_horizon_analysis', '')
    if '단기' in time_horizon:
        return 'short'
    elif '중기' in time_horizon:
        return 'medium'
    else:
        return 'long'

def select_prediction_model(risk_level, investment_horizon):
    """사용자 성향에 맞는 예측 모델 파라미터를 선택합니다."""
    if risk_level == 'low':
        # 보수적 투자자: 안정적인 예측 모델, 더 긴 학습 기간
        return {
            'model_type': 'ensemble',  # 앙상블 모델 사용
            'training_period': 500,    # 더 긴 학습 기간
            'prediction_horizon': 30 if investment_horizon == 'short' else 90,
            'confidence_interval': 0.95,  # 높은 신뢰구간
            'features': ['MA', 'EMA', 'MACD', 'RSI', 'BB']  # 안정적인 지표 위주
        }
    elif risk_level == 'medium':
        # 중립적 투자자: 균형 잡힌 모델
        return {
            'model_type': 'random_forest',
            'training_period': 365,
            'prediction_horizon': 60 if investment_horizon == 'medium' else 30,
            'confidence_interval': 0.9,
            'features': ['MA', 'EMA', 'MACD', 'RSI', 'BB', 'OBV', 'ATR']
        }
    else:
        # 공격적 투자자: 단기 변동성 포착에 강한 모델
        return {
            'model_type': 'gradient_boosting',  # 그래디언트 부스팅 사용
            'training_period': 252,  # 약 1년 거래일
            'prediction_horizon': 14 if investment_horizon == 'short' else 45,
            'confidence_interval': 0.8,  # 낮은 신뢰구간 (더 공격적)
            'features': ['MA', 'EMA', 'MACD', 'RSI', 'BB', 'OBV', 'ATR', 'MFI', 'CCI', 'ROC']  # 더 많은 기술적 지표
        }

def load_price_data(ticker):
    """특정 종목의 가격 데이터를 로드합니다."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        
        # 가능한 모든 경로에서 주가 데이터 파일 검색
        possible_files = []
        
        # 1. 스크립트 디렉토리에서 검색
        for file in os.listdir(script_dir):
            if file.startswith('kor_price_') and file.endswith('.csv'):
                possible_files.append(os.path.join(script_dir, file))
        
        # 2. 루트 디렉토리에서 검색
        for file in os.listdir(root_dir):
            if file.startswith('kor_price_') and file.endswith('.csv'):
                possible_files.append(os.path.join(root_dir, file))
        
        # 3. 'src' 디렉토리에서 검색
        src_dir = os.path.join(root_dir, 'src')
        if os.path.exists(src_dir):
            for file in os.listdir(src_dir):
                if file.startswith('kor_price_') and file.endswith('.csv'):
                    possible_files.append(os.path.join(src_dir, file))
        
        print(f"가능한 주가 데이터 파일들: {possible_files}")
        
        # 파일이 있으면 가장 최신 파일 사용
        if possible_files:
            latest_file = max(possible_files, key=os.path.getctime)
            print(f"Using price data from: {latest_file}")
            df = pd.read_csv(latest_file)
            ticker_data = df[df['종목코드'] == ticker].sort_values('날짜')
            ticker_data['날짜'] = pd.to_datetime(ticker_data['날짜'])
            return ticker_data.set_index('날짜')
        
        # 코스닥/코스피 종목 리스트에서 검색
        for file in os.listdir(script_dir):
            if file.startswith('kor_ticker_') and file.endswith('.csv'):
                ticker_file = os.path.join(script_dir, file)
                ticker_df = pd.read_csv(ticker_file)
                if ticker in ticker_df['종목코드'].values:
                    ticker_info = ticker_df[ticker_df['종목코드'] == ticker].iloc[0]
                    print(f"티커 정보: {ticker_info['종목명']} ({ticker})")
        
        # 특정 파일 이름 시도
        file_paths = [
            os.path.join(script_dir, 'kor_price_20250326.csv'),
            os.path.join(root_dir, 'kor_price_20250326.csv'),
            os.path.join(script_dir, 'kor_price_20250320.csv'),
            os.path.join(root_dir, 'kor_price_20250320.csv')
        ]
        
        for path in file_paths:
            if os.path.exists(path):
                print(f"Found specific file: {path}")
                df = pd.read_csv(path)
                ticker_data = df[df['종목코드'] == ticker].sort_values('날짜')
                ticker_data['날짜'] = pd.to_datetime(ticker_data['날짜'])
                return ticker_data.set_index('날짜')
        
        print(f"No price data file found for {ticker}")
        raise FileNotFoundError(f"주가 데이터 파일을 찾을 수 없습니다: 종목코드 {ticker}")
        
    except Exception as e:
        print(f"Error loading price data for {ticker}: {str(e)}")
        # 샘플 데이터 생성
        print("임의 샘플 데이터 생성")
        dates = pd.date_range(end=datetime.now(), periods=252)
        # 종목에 따라 다른 기본값 설정
        if ticker == '091990':  # 셀트리온헬스케어
            base_price = 58000
        elif ticker == '247540':  # 에코프로비엠
            base_price = 320000
        elif ticker == '293490':  # 카카오게임즈
            base_price = 45000
        else:
            base_price = 50000
            
        sample_data = pd.DataFrame({
            '종가': np.random.normal(base_price, base_price*0.1, len(dates)),
            '시가': np.random.normal(base_price, base_price*0.1, len(dates)),
            '고가': np.random.normal(base_price*1.05, base_price*0.1, len(dates)),
            '저가': np.random.normal(base_price*0.95, base_price*0.1, len(dates)),
            '거래량': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        return sample_data

def add_technical_indicators(df, features):
    """주가 데이터에 기술적 지표를 추가합니다."""
    # 기본 데이터 확인 및 준비
    if '종가' not in df.columns:
        raise ValueError("데이터에 '종가' 컬럼이 없습니다.")
    
    # 필요한 컬럼이 없는 경우 기본값 설정
    if '거래량' not in df.columns:
        df['거래량'] = 0
    if '고가' not in df.columns:
        df['고가'] = df['종가']
    if '저가' not in df.columns:
        df['저가'] = df['종가']
    if '시가' not in df.columns:
        df['시가'] = df['종가']
    
    # 이동평균선
    if 'MA' in features:
        df['MA_20'] = df['종가'].rolling(window=20).mean()
        df['MA_50'] = df['종가'].rolling(window=50).mean()
        df['MA_200'] = df['종가'].rolling(window=200).mean()
    
    # 지수이동평균선
    if 'EMA' in features:
        df['EMA_20'] = df['종가'].ewm(span=20, adjust=False).mean()
    
    # MACD
    if 'MACD' in features:
        ema_12 = df['종가'].ewm(span=12, adjust=False).mean()
        ema_26 = df['종가'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    
    # RSI
    if 'RSI' in features:
        delta = df['종가'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    if 'BB' in features:
        df['BB_mid'] = df['종가'].rolling(window=20).mean()
        std = df['종가'].rolling(window=20).std()
        df['BB_high'] = df['BB_mid'] + 2 * std
        df['BB_low'] = df['BB_mid'] - 2 * std
        df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']
    
    # OBV (On-Balance Volume)
    if 'OBV' in features and '거래량' in df.columns:
        df['OBV'] = (np.sign(df['종가'].diff()) * df['거래량']).fillna(0).cumsum()
    
    # ATR (Average True Range)
    if 'ATR' in features:
        tr1 = df['고가'] - df['저가']
        tr2 = abs(df['고가'] - df['종가'].shift())
        tr3 = abs(df['저가'] - df['종가'].shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
    
    # MFI (Money Flow Index)
    if 'MFI' in features and '거래량' in df.columns:
        typical_price = (df['고가'] + df['저가'] + df['종가']) / 3
        money_flow = typical_price * df['거래량']
        
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0).rolling(window=14).sum()
        negative_flow = money_flow.where(delta < 0, 0).rolling(window=14).sum()
        
        money_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + money_ratio))
    
    # CCI (Commodity Channel Index)
    if 'CCI' in features:
        typical_price = (df['고가'] + df['저가'] + df['종가']) / 3
        mean_dev = abs(typical_price - typical_price.rolling(window=20).mean()).rolling(window=20).mean()
        df['CCI'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_dev)
    
    # ROC (Rate of Change)
    if 'ROC' in features:
        df['ROC'] = df['종가'].pct_change(periods=10) * 100
    
    return df

def prepare_data_for_ml(df, target_column='종가', test_split=0.2, sequence_length=30):
    """머신러닝 모델 학습을 위한 데이터를 준비합니다."""
    # 타겟 컬럼 선택
    data = df.copy()
    
    # 시계열 특성 추가
    for i in range(1, sequence_length + 1):
        data[f'종가_lag_{i}'] = data['종가'].shift(i)
    
    # 결측치 제거
    data = data.dropna()
    
    # 타겟 변수와 특성 분리
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # 학습/테스트 데이터 분할
    train_size = int(len(X) * (1 - test_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test

def create_random_forest_model():
    """Random Forest 모델을 생성합니다."""
    return RandomForestRegressor(n_estimators=100, random_state=42)

def create_gradient_boosting_model():
    """Gradient Boosting 모델을 생성합니다."""
    return GradientBoostingRegressor(n_estimators=100, random_state=42)

def create_ensemble_model():
    """앙상블 모델(Random Forest + Gradient Boosting + Linear Regression)을 생성합니다."""
    rf_model = create_random_forest_model()
    gb_model = create_gradient_boosting_model()
    lr_model = LinearRegression()
    
    return {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'lr_model': lr_model
    }

def train_prediction_model(df, model_params, sequence_length=30):
    """선택된 모델 파라미터에 따라 예측 모델을 학습합니다."""
    # 데이터 준비
    X_train, y_train, X_test, y_test = prepare_data_for_ml(
        df, target_column='종가', test_split=0.2, sequence_length=sequence_length
    )
    
    if model_params['model_type'] == 'random_forest':
        model = create_random_forest_model()
        model.fit(X_train, y_train)
        return {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'type': 'random_forest',
            'feature_names': X_train.columns.tolist()
        }
    
    elif model_params['model_type'] == 'gradient_boosting':
        model = create_gradient_boosting_model()
        model.fit(X_train, y_train)
        return {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'type': 'gradient_boosting',
            'feature_names': X_train.columns.tolist()
        }
    
    elif model_params['model_type'] == 'ensemble':
        ensemble = create_ensemble_model()
        ensemble['rf_model'].fit(X_train, y_train)
        ensemble['gb_model'].fit(X_train, y_train)
        ensemble['lr_model'].fit(X_train, y_train)
        return {
            'model': ensemble,
            'X_test': X_test,
            'y_test': y_test,
            'type': 'ensemble',
            'feature_names': X_train.columns.tolist()
        }
    
    else:
        raise ValueError(f"지원하지 않는 모델 유형: {model_params['model_type']}")

def predict_with_model(model_result, df, prediction_horizon, sequence_length=30):
    """학습된 모델을 사용하여 미래 가격을 예측합니다."""
    # 마지막 데이터 포인트 가져오기
    last_data = df.copy()
    
    # 시계열 특성 추가
    for i in range(1, sequence_length + 1):
        last_data[f'종가_lag_{i}'] = last_data['종가'].shift(i)
    
    last_data = last_data.dropna().iloc[-1:]
    last_features = last_data.drop(columns=['종가'])
    
    # 예측 수행
    predictions = []
    current_features = last_features.copy()
    
    for _ in range(prediction_horizon):
        if model_result['type'] == 'ensemble':
            # 앙상블 모델 예측 (평균)
            rf_pred = model_result['model']['rf_model'].predict(current_features)[0]
            gb_pred = model_result['model']['gb_model'].predict(current_features)[0]
            lr_pred = model_result['model']['lr_model'].predict(current_features)[0]
            next_pred = (rf_pred + gb_pred + lr_pred) / 3
        else:
            # 단일 모델 예측
            next_pred = model_result['model'].predict(current_features)[0]
        
        predictions.append(next_pred)
        
        # 다음 예측을 위한 특성 업데이트
        new_features = current_features.copy()
        
        # 시계열 특성 업데이트
        for i in range(sequence_length, 1, -1):
            new_features[f'종가_lag_{i}'] = new_features[f'종가_lag_{i-1}']
        new_features['종가_lag_1'] = next_pred
        
        current_features = new_features
    
    return predictions

def visualize_prediction_with_confidence(df, predictions, ticker, confidence_interval=0.9):
    """예측 결과를 신뢰구간과 함께 시각화합니다."""
    plt.figure(figsize=(12, 6))
    
    # 과거 데이터 플롯
    historical_dates = df.index[-60:]  # 최근 60일 데이터만 표시
    historical_prices = df['종가'].iloc[-60:]
    plt.plot(historical_dates, historical_prices, label='Historical Data', color='blue')
    
    # 예측 기간 생성
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, len(predictions)+1)]
    
    # 예측값 플롯
    plt.plot(forecast_dates, predictions, 'r--', label='Predicted Price')
    
    # 신뢰구간 계산 및 플롯
    std_dev = np.std(df['종가'].pct_change().dropna()) * np.sqrt(np.arange(1, len(predictions)+1))
    z_score = {0.8: 1.28, 0.9: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_score.get(confidence_interval, 1.645)
    
    upper_bound = [predictions[i] * (1 + z * std_dev[i]) for i in range(len(predictions))]
    lower_bound = [predictions[i] * (1 - z * std_dev[i]) for i in range(len(predictions))]
    
    plt.fill_between(forecast_dates, lower_bound, upper_bound, color='red', alpha=0.2, label=f'{confidence_interval*100}% Confidence Interval')
    
    # 그래프 설정
    plt.title(f'Stock Price Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 파일로 저장
    chart_path = f'{ticker}_prediction.png'
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

def summarize_prediction_results(df, predictions, chart_path):
    """예측 결과를 요약합니다."""
    current_price = df['종가'].iloc[-1]
    predicted_price = predictions[-1]
    change_pct = ((predicted_price - current_price) / current_price) * 100
    
    # 추세 분석
    trend = "상승" if predicted_price > current_price else "하락"
    
    # 변동성 분석
    volatility = np.std(df['종가'].pct_change().dropna()) * 100
    
    return {
        'current_price': float(current_price),
        'predicted_price': float(predicted_price),
        'change_pct': float(change_pct),
        'trend': trend,
        'volatility': float(volatility),
        'chart_path': chart_path
    }

def analyze_portfolio_performance(ticker_results, risk_level):
    """포트폴리오 전체 성과를 분석합니다."""
    # 각 종목의 예측 변화율 수집
    change_pcts = [result['change_pct'] for result in ticker_results.values()]
    
    # 포트폴리오 전체 예상 수익률 (단순 평균)
    avg_return = np.mean(change_pcts)
    
    # 위험 수준에 따른 평가
    if risk_level == 'low':
        risk_assessment = "안정적" if avg_return > 0 and avg_return < 5 else "불안정"
    elif risk_level == 'medium':
        risk_assessment = "안정적" if avg_return > 0 and avg_return < 10 else "불안정"
    else:  # high
        risk_assessment = "안정적" if avg_return > 0 and avg_return < 15 else "불안정"
    
    return {
        'expected_return': float(avg_return),
        'risk_assessment': risk_assessment,
        'recommendation': "유지" if risk_assessment == "안정적" else "재조정 필요"
    }

def extract_portfolio_tickers(ai_a2_ai_b_history):
    """AI-A2와 AI-B의 대화에서 추천된 포트폴리오 종목 코드를 추출합니다."""
    # 대화 내용에서 종목 코드 패턴 찾기
    portfolio_tickers = []
    
    # 종목명 매핑 (한글 종목명 -> 종목 코드)
    stock_name_to_code = {
        "셀트리온헬스케어": "091990",  # 코스닥
        "에코프로비엠": "247540",      # 코스닥
        "카카오게임즈": "293490",      # 코스닥
        "셀트리온제약": "068760",      # 코스닥
        "알테오젠": "196170",         # 코스닥
        "CJ ENM": "035760",          # 코스닥
        "SK아이이테크놀로지": "361610", # 코스닥
        "펄어비스": "263750",          # 코스닥
        "엘앤에프": "066970",          # 코스닥
        "위메이드": "112040",          # 코스닥
        "삼성전자": "005930",          # 코스피
        "SK하이닉스": "000660",        # 코스피
        "NAVER": "035420",           # 코스피
        "카카오": "035720",           # 코스피
    }
    
    # 종목명 추출을 위한 추가 패턴
    for message in ai_a2_ai_b_history:
        content = message.get('content', '')
        
        # 다양한 종목 코드 패턴 검색
        # 1. 기본 종목 코드 패턴 (예: 005930, 000660 등)
        basic_ticker_pattern = r'\b\d{6}\b'
        # 2. 괄호 안에 있는 종목 코드 패턴 (예: (005930), [000660] 등)
        parenthesis_ticker_pattern = r'[\(\[]([\d]{6})[\)\]]'
        # 3. 종목명과 함께 있는 패턴 (예: 삼성전자(005930), SK하이닉스(000660) 등)
        name_ticker_pattern = r'([가-힣A-Za-z\s]+)\(?(\d{6})\)?'
        
        # 기본 패턴 검색
        found_tickers = re.findall(basic_ticker_pattern, content)
        
        # 괄호 안 패턴 검색
        parenthesis_matches = re.findall(parenthesis_ticker_pattern, content)
        found_tickers.extend(parenthesis_matches)
        
        # 종목명과 함께 있는 패턴 검색
        name_matches = re.findall(name_ticker_pattern, content)
        for match in name_matches:
            if len(match) > 1:
                found_tickers.append(match[1])  # 종목 코드 부분만 추출
        
        # 종목명 검색 (종목 코드 없이 이름만 언급된 경우)
        for stock_name, stock_code in stock_name_to_code.items():
            if stock_name in content and stock_code not in found_tickers:
                found_tickers.append(stock_code)
        
        # 중복 제거하면서 추가
        for ticker in found_tickers:
            if ticker not in portfolio_tickers and len(ticker) == 6 and ticker.isdigit():
                portfolio_tickers.append(ticker)
    
    print(f"추출된 종목 코드: {portfolio_tickers}")
    
    # 종목 코드가 없을 경우 기본 코드 반환
    if not portfolio_tickers:
        print("추출된 종목 코드가 없습니다. 코스닥 대표 종목을 사용합니다.")
        # 코스닥 대표 종목 (셀트리온헬스케어, 에코프로비엠, 카카오게임즈, 알테오젠, 엘앤에프)
        portfolio_tickers = ['091990', '247540', '293490', '196170', '066970']
    
    return portfolio_tickers

def analyze_portfolio_with_user_profile(portfolio_tickers, user_id='default'):
    """사용자 투자 성향에 맞는 포트폴리오 예측 분석을 수행합니다."""
    # 사용자 프로필 로드
    user_profile = load_user_profile()
    risk_level = get_risk_level(user_profile)
    investment_horizon = get_investment_horizon(user_profile)
    
    # 사용자 성향에 맞는 모델 파라미터 선택
    model_params = select_prediction_model(risk_level, investment_horizon)
    
    # Try to use advanced ARIMA-X predictor first
    try:
        from advanced_stock_predictor import AdvancedStockPredictor
        
        predictor = AdvancedStockPredictor()
        advanced_results = predictor.analyze_multiple_stocks(portfolio_tickers)
        
        # Convert advanced results to compatible format
        results = {}
        for ticker in portfolio_tickers:
            if ticker in advanced_results['individual_results']:
                ticker_result = advanced_results['individual_results'][ticker]
                
                if 'error' not in ticker_result:
                    # Convert to compatible format
                    current_price = ticker_result['current_price']
                    predicted_prices = ticker_result['predicted_prices']
                    final_predicted = predicted_prices[-1] if predicted_prices else current_price
                    
                    change_pct = ticker_result['insights']['expected_change_pct']
                    trend = ticker_result['insights']['trend']
                    
                    results[ticker] = {
                        'current_price': current_price,
                        'predicted_price': final_predicted,
                        'change_pct': change_pct,
                        'trend': 'bullish' if trend == 'bullish' else 'bearish',
                        'volatility': ticker_result['insights']['volatility'],
                        'chart_path': ticker_result.get('chart_path', ''),
                        'prediction_horizon': '7_days_arimax',
                        'model_type': 'ARIMA-X with News Sentiment'
                    }
        
        print(f"Advanced ARIMA-X 예측 완료: {len(results)}개 종목")
        
    except Exception as e:
        print(f"Advanced ARIMA-X 예측 실패, 기존 모델 사용: {str(e)}")
        
        # Fallback to existing prediction method
        results = {}
        for ticker in portfolio_tickers:
            try:
                # 주가 데이터 로드
                price_data = load_price_data(ticker)
                
                # 기술적 지표 추가
                price_data = add_technical_indicators(price_data, model_params['features'])
                
                # 결측치 처리
                price_data = price_data.dropna()
                
                if len(price_data) > model_params['training_period']:
                    # 모델 학습
                    model_result = train_prediction_model(price_data, model_params)
                    
                    # 예측 수행 (1주일로 변경)
                    forecast = predict_with_model(model_result, price_data, 7)  # 7 days prediction
                    
                    # 결과 시각화
                    chart_path = visualize_prediction_with_confidence(
                        price_data, forecast, ticker, 
                        confidence_interval=model_params['confidence_interval']
                    )
                    
                    # 예측 결과 요약
                    results[ticker] = summarize_prediction_results(price_data, forecast, chart_path)
                    results[ticker]['model_type'] = 'Traditional ML'
                else:
                    print(f"Warning: {ticker}의 데이터가 충분하지 않습니다.")
            except Exception as e:
                print(f"Error processing ticker {ticker}: {str(e)}")
    
    # 포트폴리오 전체 분석 결과
    portfolio_analysis = analyze_portfolio_performance(results, risk_level) if results else {}
    
    result = {
        'ticker_predictions': results,
        'portfolio_analysis': portfolio_analysis,
        'user_profile': {
            'risk_level': risk_level,
            'investment_horizon': investment_horizon
        },
        'prediction_model': 'ARIMA-X Enhanced' if len([r for r in results.values() if 'ARIMA-X' in r.get('model_type', '')]) > 0 else 'Traditional ML'
    }
    
    # supabase 저장
    try:
        supabase = get_supabase_client()
        data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "portfolio_json": json.dumps(result, ensure_ascii=False)
        }
        supabase.table("portfolio_recommendations").upsert(data).execute()
        print(f"포트폴리오 추천 supabase upsert 완료: {user_id}")
    except Exception as e:
        print(f"Supabase 저장 실패: {e}")
    
    return result

def get_portfolio_recommendations(user_id, limit=5):
    """
    supabase에서 사용자별 포트폴리오 추천/분석 이력 조회
    최신순 limit개 반환 (portfolio_json 파싱)
    """
    supabase = get_supabase_client()
    res = supabase.table("portfolio_recommendations").select("portfolio_json,created_at").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
    results = []
    if res.data:
        for row in res.data:
            try:
                results.append(json.loads(row["portfolio_json"]))
            except Exception as e:
                print(f"portfolio_json 파싱 오류: {e}")
        print(f"supabase에서 포트폴리오 추천 이력 {len(results)}건 조회: {user_id}")
    else:
        print(f"supabase에 포트폴리오 추천 이력 없음: {user_id}")
    return results

# 테스트 코드
if __name__ == "__main__":
    # 테스트 포트폴리오
    test_tickers = ['005930', '000660', '035420']
    results = analyze_portfolio_with_user_profile(test_tickers)
    print(json.dumps(results, indent=2, ensure_ascii=False))
