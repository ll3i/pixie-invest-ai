# Pixie 차트 분석 기능 문서

## 개요
Pixie 투자 서비스의 차트 분석 페이지는 사용자의 투자 성향에 맞는 포트폴리오 예측 결과를 시각적으로 제공하는 기능입니다.

## 주요 기능

### 1. 포트폴리오 예측 시각화
- 사용자 맞춤형 5개 종목 추천
- 각 종목별 30일 가격 예측 차트
- Chart.js를 활용한 인터랙티브 차트

### 2. 투자 지표 제공
- **현재가**: 실시간 주식 가격
- **예상 수익률**: 30일 기준 수익률 예측
- **변동성**: 가격 변동 위험도
- **신뢰도**: AI 예측 신뢰도

### 3. AI 분석 의견
- 각 종목별 맞춤형 분석 코멘트
- 사용자 투자 성향 기반 추천 이유

## 기술 구현

### 백엔드 (Flask)

#### 라우트 정의
```python
@app.route('/chart-analysis')
def chart_analysis():
    """차트 분석 페이지를 렌더링합니다."""
    # 사용자 인증 확인
    # 프로필 데이터 조회
    # 포트폴리오 예측 데이터 생성
    # 템플릿 렌더링
```

#### 데이터 구조
```python
{
    "stock_code": "005930",
    "stock_name": "삼성전자",
    "current_price": 70000,
    "expected_return": 5.2,
    "volatility": 15.3,
    "confidence": 0.85,
    "trend": "bullish",
    "predictions": [
        {"day": 1, "predicted_price": 70500},
        {"day": 2, "predicted_price": 71000},
        # ... 30일 예측 데이터
    ],
    "analysis_summary": "AI 분석 의견"
}
```

### 프론트엔드 (HTML/JavaScript)

#### Chart.js 통합
```javascript
new Chart(ctx, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [{
            label: '예상 가격',
            data: prices,
            borderColor: '#1454FE',
            backgroundColor: 'rgba(20, 84, 254, 0.1)'
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false
    }
});
```

## 데이터 수집 시스템

### 한국 주식 데이터
- **대상**: KOSPI/KOSDAQ 상위 200개 종목
- **수집 주기**: 매일 오전 6시
- **데이터 소스**: KRX, FinanceDataReader
- **저장 위치**: `data/raw/kor_price_YYYYMMDD.csv`

### 미국 주식 데이터
- **대상**: 20개 우량주 (AAPL, MSFT, GOOGL 등)
- **수집 주기**: 매일 오전 6시
- **데이터 소스**: yfinance
- **저장 위치**: `data/raw/us_price_YYYYMMDD.csv`

### 뉴스 데이터
- **수집 주기**: 매일 오전 7시
- **데이터 소스**: RSS 피드 (Yahoo Finance, CNN Money, Bloomberg)
- **저장 위치**: `data/raw/news_YYYYMMDD.csv`

## 사용자 경험 (UX)

### 접근 경로
1. 메인 네비게이션 → "차트 분석" 클릭
2. 투자 성향 분석 완료 후 자동 연결
3. AI 챗봇에서 추천

### 필수 조건
- 투자 성향 분석 완료
- 유효한 세션 ID

### 예외 처리
- 미로그인 시 → 설문 페이지로 리다이렉트
- 데이터 없음 → 안내 메시지 표시

## 디자인 시스템

### Pixie 컬러 팔레트
```css
--pixie-primary: #1454FE;
--pixie-primary-hover: #0B3DCF;
--pixie-secondary: #CADBFF;
--pixie-black: #000000;
--pixie-dark-gray: #474747;
--pixie-light-gray: #F5F5F5;
--pixie-white: #FFFFFF;
```

### 반응형 브레이크포인트
- 데스크톱: 1200px+
- 태블릿: 768px - 1199px
- 모바일: 767px 이하

## API 엔드포인트

### GET /chart-analysis
- **설명**: 차트 분석 페이지 렌더링
- **인증**: 세션 기반 (user_id 필수)
- **응답**: HTML 페이지

### GET /api/predictions
- **설명**: 포트폴리오 예측 데이터 조회
- **인증**: 세션 기반
- **응답**: JSON 형식의 예측 데이터

## 성능 최적화

### 캐싱 전략
- 예측 데이터 15분 캐싱
- 정적 리소스 브라우저 캐싱
- Chart.js CDN 활용

### 로딩 최적화
- 차트 순차 렌더링
- 지연 로딩 구현
- 압축된 데이터 전송

## 향후 개선 계획

1. **실시간 업데이트**
   - WebSocket 통한 실시간 가격 반영
   - 자동 새로고침 옵션

2. **고급 차트 기능**
   - 기술적 지표 추가 (이동평균선, RSI 등)
   - 비교 차트 기능
   - 줌/팬 기능

3. **개인화 강화**
   - 관심 종목 설정
   - 알림 기능 연동
   - 투자 기록 추적

## 문제 해결

### 일반적인 이슈
1. **차트가 표시되지 않음**
   - Chart.js 라이브러리 로드 확인
   - 콘솔 에러 확인
   - 데이터 형식 검증

2. **데이터 불일치**
   - 캐시 삭제
   - 데이터 수집 로그 확인
   - DB 동기화 상태 확인

### 디버깅
```python
# 로그 위치
/logs/minerva_YYYYMMDD.log
/logs/data_update.log

# 디버그 모드 활성화
FLASK_ENV=development
```

## 관련 문서
- [투자 성향 분석 가이드](./SURVEY_GUIDE.md)
- [AI 챗봇 통합 문서](./CHATBOT_INTEGRATION.md)
- [데이터 수집 시스템](./DATA_COLLECTION.md)