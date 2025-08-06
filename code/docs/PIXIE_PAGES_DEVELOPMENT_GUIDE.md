# 📚 Pixie 페이지 개발 가이드

## 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [기술 스택 및 아키텍처](#기술-스택-및-아키텍처)
3. [각 페이지별 개발 가이드](#각-페이지별-개발-가이드)
   - [메인 대시보드](#메인-대시보드)
   - [투자 설문](#투자-설문)
   - [투자 학습](#투자-학습)
   - [AI 챗봇](#ai-챗봇)
   - [주가 예측](#주가-예측)
   - [뉴스/이슈](#뉴스이슈)
   - [MY 투자](#my-투자)
4. [공통 컴포넌트 및 서비스](#공통-컴포넌트-및-서비스)
5. [데이터베이스 설계](#데이터베이스-설계)
6. [API 엔드포인트 정의](#api-엔드포인트-정의)
7. [보안 및 인증](#보안-및-인증)
8. [테스트 전략](#테스트-전략)
9. [배포 가이드](#배포-가이드)

---

## 🎯 프로젝트 개요

Pixie는 AI 기반 개인화 투자 자문 시스템으로, 사용자에게 맞춤형 투자 조언과 교육을 제공합니다.

### 핵심 기능
- **개인화 투자 분석**: 사용자 프로필 기반 맞춤형 투자 추천
- **AI 챗봇 상담**: 다중 AI 에이전트 시스템을 통한 실시간 투자 상담
- **투자 교육**: 단계별 학습 시스템과 인터랙티브 퀴즈
- **시장 분석**: 실시간 주가 예측 및 뉴스 분석
- **포트폴리오 관리**: 개인 투자 현황 추적 및 성과 분석

---

## 🛠️ 기술 스택 및 아키텍처

### 프론트엔드
- **언어**: HTML5, CSS3, JavaScript (ES6+)
- **프레임워크**: Bootstrap 5
- **차트 라이브러리**: Chart.js, D3.js
- **실시간 통신**: WebSocket, AJAX

### 백엔드
- **언어**: Python 3.9+
- **프레임워크**: Flask
- **AI/ML**: OpenAI API, TensorFlow, scikit-learn
- **데이터베이스**: SQLite (개발), PostgreSQL (프로덕션)
- **캐싱**: Redis

### 프로젝트 구조
```
투자챗봇/
├── web/                        # 웹 애플리케이션
│   ├── app.py                 # 메인 Flask 애플리케이션
│   ├── blueprints/            # 모듈화된 라우트
│   ├── services/              # 비즈니스 로직
│   ├── static/                # 정적 파일 (CSS, JS, 이미지)
│   └── templates/             # HTML 템플릿
├── src/                       # 핵심 비즈니스 로직
│   ├── investment_advisor.py  # AI 투자 상담 시스템
│   ├── data_collector.py      # 데이터 수집
│   └── financial_data_processor.py  # 데이터 처리
└── docs/                      # 문서
```

---

## 📱 각 페이지별 개발 가이드

### 메인 대시보드

#### 개요
사용자에게 종합적인 투자 현황과 시장 정보를 제공하는 중앙 허브

#### 핵심 기능
1. **시장 현황 표시**
   - KOSPI/KOSDAQ 지수
   - 환율 정보
   - 주요 뉴스

2. **포트폴리오 요약**
   - 총 자산 가치
   - 수익률
   - 보유 종목 현황

3. **AI 추천**
   - 일일 투자 추천
   - 위험 알림

#### 구현 예시
```python
# web/app.py
@app.route('/dashboard')
@login_required
def dashboard():
    """메인 대시보드"""
    user_id = session.get('user_id')
    
    # 시장 데이터 조회
    market_data = get_market_summary()
    
    # 포트폴리오 조회
    portfolio = get_user_portfolio(user_id)
    
    # AI 추천 생성
    recommendations = get_ai_recommendations(user_id)
    
    return render_template('dashboard.html',
                         market_data=market_data,
                         portfolio=portfolio,
                         recommendations=recommendations)

@app.route('/api/market-summary')
def get_market_summary():
    """실시간 시장 데이터 API"""
    try:
        data = {
            'kospi': get_kospi_index(),
            'kosdaq': get_kosdaq_index(),
            'exchange_rate': get_exchange_rate(),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(data)
    except Exception as e:
        logger.error(f"시장 데이터 조회 실패: {e}")
        return jsonify({'error': '데이터 조회 실패'}), 500
```

#### 프론트엔드 구현
```javascript
// static/js/dashboard.js
class DashboardManager {
    constructor() {
        this.init();
    }
    
    init() {
        this.loadMarketData();
        this.loadPortfolio();
        this.setupRefreshInterval();
    }
    
    async loadMarketData() {
        try {
            const response = await fetch('/api/market-summary');
            const data = await response.json();
            this.updateMarketUI(data);
        } catch (error) {
            console.error('시장 데이터 로드 실패:', error);
        }
    }
    
    updateMarketUI(data) {
        // KOSPI 업데이트
        document.getElementById('kospi-index').textContent = data.kospi.value;
        document.getElementById('kospi-change').textContent = data.kospi.change + '%';
        
        // KOSDAQ 업데이트
        document.getElementById('kosdaq-index').textContent = data.kosdaq.value;
        document.getElementById('kosdaq-change').textContent = data.kosdaq.change + '%';
    }
}
```

---

### 투자 설문

#### 개요
사용자의 투자 성향과 위험 선호도를 분석하여 개인화된 투자 프로필 생성

#### 핵심 기능
1. **단계별 설문**
   - 투자 경험
   - 위험 선호도
   - 투자 목표
   - 재무 상황

2. **실시간 분석**
   - 답변 기반 점수 계산
   - 투자 성향 분류
   - 위험 등급 결정

#### 구현 예시
```python
# web/services/survey_service.py
class SurveyService:
    def __init__(self):
        self.questions = self.load_questions()
    
    def analyze_responses(self, responses):
        """설문 응답 분석"""
        scores = {
            'risk_tolerance': 0,
            'investment_horizon': 0,
            'financial_knowledge': 0,
            'investment_goals': 0
        }
        
        for response in responses:
            category = response['category']
            scores[category] += response['value']
        
        # 투자 성향 결정
        profile = self.determine_profile(scores)
        
        return {
            'scores': scores,
            'profile': profile,
            'risk_level': self.calculate_risk_level(scores),
            'recommendations': self.generate_recommendations(profile)
        }
    
    def determine_profile(self, scores):
        """투자 성향 결정"""
        if scores['risk_tolerance'] < 30:
            return 'conservative'
        elif scores['risk_tolerance'] < 70:
            return 'balanced'
        else:
            return 'aggressive'
```

#### 설문 UI 구현
```html
<!-- templates/survey.html -->
<div class="survey-container">
    <div class="progress-bar">
        <div class="progress-fill" style="width: 0%"></div>
    </div>
    
    <form id="survey-form">
        <div class="question-container" data-question="1">
            <h3>투자 손실에 대한 귀하의 반응은?</h3>
            <div class="options">
                <label>
                    <input type="radio" name="q1" value="1">
                    매우 민감함 - 잠을 못 이룸
                </label>
                <label>
                    <input type="radio" name="q1" value="3">
                    보통 - 적당히 신경 씀
                </label>
                <label>
                    <input type="radio" name="q1" value="5">
                    전혀 민감하지 않음
                </label>
            </div>
        </div>
    </form>
    
    <div class="navigation-buttons">
        <button id="prev-btn" onclick="surveyManager.previousQuestion()">이전</button>
        <button id="next-btn" onclick="surveyManager.nextQuestion()">다음</button>
    </div>
</div>
```

---

### 투자 학습

#### 개요
투자 초보자부터 중급자까지 단계별 투자 교육 제공

#### 핵심 기능
1. **학습 모듈**
   - 기초: 투자 개념, 주식 시장 이해
   - 중급: 기술적/기본적 분석
   - 고급: 파생상품, 글로벌 투자

2. **인터랙티브 콘텐츠**
   - 카드뉴스
   - 퀴즈 시스템
   - 진도 추적

#### 구현 예시
```python
# web/services/learning_service.py
class LearningService:
    def __init__(self):
        self.modules = self.load_modules()
    
    def get_user_progress(self, user_id):
        """사용자 학습 진도 조회"""
        progress = db.query(
            "SELECT * FROM user_learning_progress WHERE user_id = ?",
            (user_id,)
        )
        
        completed_lessons = len([p for p in progress if p['completed']])
        total_lessons = self.get_total_lessons()
        
        return {
            'completed': completed_lessons,
            'total': total_lessons,
            'percentage': (completed_lessons / total_lessons) * 100,
            'current_module': self.get_current_module(user_id),
            'achievements': self.get_achievements(user_id)
        }
    
    def complete_lesson(self, user_id, lesson_id):
        """강의 완료 처리"""
        # 완료 기록
        db.execute(
            "INSERT INTO user_learning_progress (user_id, lesson_id, completed_at) VALUES (?, ?, ?)",
            (user_id, lesson_id, datetime.now())
        )
        
        # 퀴즈 확인
        quiz = self.get_lesson_quiz(lesson_id)
        if quiz:
            return {'next': 'quiz', 'quiz_id': quiz['id']}
        else:
            return {'next': 'lesson', 'lesson_id': self.get_next_lesson(lesson_id)}
```

#### 카드뉴스 구현
```javascript
// static/js/learning-cardnews.js
class CardNewsViewer {
    constructor(container) {
        this.container = container;
        this.currentSlide = 0;
        this.slides = [];
        this.init();
    }
    
    init() {
        this.loadSlides();
        this.renderSlide(0);
        this.setupSwipeGestures();
    }
    
    loadSlides() {
        // 서버에서 슬라이드 데이터 로드
        fetch('/api/learning/cardnews/' + this.container.dataset.moduleId)
            .then(res => res.json())
            .then(data => {
                this.slides = data.slides;
                this.renderSlide(0);
            });
    }
    
    renderSlide(index) {
        const slide = this.slides[index];
        this.container.innerHTML = `
            <div class="cardnews-slide">
                <img src="${slide.image}" alt="${slide.title}">
                <div class="slide-content">
                    <h3>${slide.title}</h3>
                    <p>${slide.content}</p>
                </div>
            </div>
        `;
    }
}
```

---

### AI 챗봇

#### 개요
다중 AI 에이전트 시스템을 통한 개인화된 투자 상담

#### 핵심 기능
1. **다중 AI 에이전트**
   - AI-A: 초기 의도 분석
   - AI-A2: 질문 정제
   - AI-B: 데이터 분석
   - Final: 종합 응답

2. **대화 관리**
   - 컨텍스트 유지
   - 메모리 시스템
   - 세션 관리

#### 구현 예시
```python
# src/investment_advisor.py
class InvestmentAdvisor:
    def chat(self, user_id, message):
        """AI 챗봇 메인 로직"""
        # 사용자 프로필 로드
        user_profile = self.get_user_profile(user_id)
        
        # 시장 컨텍스트 수집
        market_context = self.get_market_context()
        
        # AI-A: 초기 분석
        ai_a_response = self.generate_ai_a_response(
            message, user_profile, market_context
        )
        
        # AI-A2: 질문 정제
        ai_a2_response = self.generate_ai_a2_response(
            message, ai_a_response, user_profile
        )
        
        # AI-B: 데이터 분석
        ai_b_response = self.generate_ai_b_response(
            ai_a2_response, market_context
        )
        
        # 최종 응답 생성
        final_response = self.generate_final_response(
            message, ai_a_response, ai_a2_response, ai_b_response
        )
        
        # 대화 저장
        self.save_conversation(user_id, message, final_response)
        
        return final_response
```

#### 챗봇 UI 구현
```javascript
// static/js/chatbot.js
class ChatbotUI {
    constructor() {
        this.socket = null;
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.setupEventListeners();
    }
    
    connectWebSocket() {
        this.socket = new WebSocket('ws://localhost:5000/ws/chat');
        
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
    }
    
    sendMessage(message) {
        // 사용자 메시지 표시
        this.addMessage(message, 'user');
        
        // 서버로 전송
        this.socket.send(JSON.stringify({
            type: 'message',
            content: message,
            timestamp: new Date().toISOString()
        }));
        
        // 타이핑 표시
        this.showTypingIndicator();
    }
    
    handleMessage(data) {
        this.hideTypingIndicator();
        
        if (data.type === 'response') {
            this.addMessage(data.content, 'ai');
        } else if (data.type === 'suggestions') {
            this.showSuggestions(data.suggestions);
        }
    }
}
```

---

### 주가 예측

#### 개요
AI 기반 주가 예측 및 기술적 분석 제공

#### 핵심 기능
1. **예측 모델**
   - LSTM 기반 시계열 예측
   - 기술적 지표 분석
   - 신뢰구간 제시

2. **시각화**
   - 캔들스틱 차트
   - 거래량 분석
   - 예측 결과 그래프

#### 구현 예시
```python
# src/advanced_stock_predictor.py
class StockPredictor:
    def __init__(self):
        self.model = self.load_model()
    
    def predict_price(self, ticker, days=30):
        """주가 예측"""
        # 과거 데이터 수집
        historical_data = self.get_historical_data(ticker)
        
        # 특징 추출
        features = self.extract_features(historical_data)
        
        # 예측 수행
        predictions = self.model.predict(features)
        
        # 신뢰구간 계산
        confidence_interval = self.calculate_confidence_interval(predictions)
        
        return {
            'ticker': ticker,
            'predictions': predictions.tolist(),
            'confidence_interval': confidence_interval,
            'technical_indicators': self.calculate_indicators(historical_data),
            'recommendation': self.generate_recommendation(predictions)
        }
    
    def calculate_indicators(self, data):
        """기술적 지표 계산"""
        return {
            'rsi': self.calculate_rsi(data),
            'macd': self.calculate_macd(data),
            'bollinger_bands': self.calculate_bollinger_bands(data),
            'moving_averages': {
                'ma5': data['close'].rolling(5).mean(),
                'ma20': data['close'].rolling(20).mean(),
                'ma60': data['close'].rolling(60).mean()
            }
        }
```

#### 차트 구현
```javascript
// static/js/stock-chart.js
class StockChart {
    constructor(container) {
        this.container = container;
        this.chart = null;
        this.init();
    }
    
    init() {
        this.createChart();
        this.loadData();
    }
    
    createChart() {
        const ctx = this.container.getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'candlestick',
            data: {
                datasets: [{
                    label: '주가',
                    data: []
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day'
                        }
                    }
                }
            }
        });
    }
    
    async loadData() {
        const ticker = this.container.dataset.ticker;
        const response = await fetch(`/api/stock/prediction/${ticker}`);
        const data = await response.json();
        
        this.updateChart(data);
    }
    
    updateChart(data) {
        // 과거 데이터 표시
        this.chart.data.datasets[0].data = data.historical;
        
        // 예측 데이터 추가
        this.chart.data.datasets.push({
            label: '예측',
            data: data.predictions,
            borderColor: 'rgba(255, 99, 132, 0.8)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderDash: [5, 5]
        });
        
        this.chart.update();
    }
}
```

---

### 뉴스/이슈

#### 개요
실시간 금융 뉴스 수집 및 AI 기반 감성 분석

#### 핵심 기능
1. **뉴스 수집**
   - RSS 피드 크롤링
   - 키워드 필터링
   - 중복 제거

2. **감성 분석**
   - AI 기반 긍정/부정 분석
   - 시장 영향도 평가
   - 관련 종목 매핑

#### 구현 예시
```python
# src/news_sentiment_analyzer.py
class NewsSentimentAnalyzer:
    def analyze_news(self, news_items):
        """뉴스 감성 분석"""
        results = []
        
        for news in news_items:
            # 감성 분석
            sentiment = self.analyze_sentiment(news['content'])
            
            # 관련 종목 추출
            related_stocks = self.extract_stock_mentions(news['content'])
            
            # 시장 영향도 평가
            market_impact = self.evaluate_market_impact(
                sentiment, 
                related_stocks,
                news['source_credibility']
            )
            
            results.append({
                'news_id': news['id'],
                'title': news['title'],
                'sentiment': sentiment,
                'related_stocks': related_stocks,
                'market_impact': market_impact,
                'published_at': news['published_at']
            })
        
        return results
    
    def analyze_sentiment(self, text):
        """OpenAI를 사용한 감성 분석"""
        prompt = f"""
        다음 뉴스의 감성을 분석하세요:
        {text}
        
        응답 형식:
        - sentiment: positive/negative/neutral
        - score: -1.0 ~ 1.0
        - reasoning: 판단 근거
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return json.loads(response.choices[0].message.content)
```

#### 뉴스 UI 구현
```html
<!-- templates/news.html -->
<div class="news-container">
    <div class="news-filters">
        <button class="filter-btn active" data-filter="all">전체</button>
        <button class="filter-btn" data-filter="positive">긍정적</button>
        <button class="filter-btn" data-filter="negative">부정적</button>
    </div>
    
    <div class="news-list" id="news-list">
        <!-- 뉴스 아이템 동적 로드 -->
    </div>
</div>

<script>
class NewsManager {
    constructor() {
        this.currentFilter = 'all';
        this.init();
    }
    
    init() {
        this.loadNews();
        this.setupFilters();
        this.setupAutoRefresh();
    }
    
    async loadNews() {
        const response = await fetch('/api/news/latest');
        const news = await response.json();
        
        this.displayNews(news);
    }
    
    displayNews(news) {
        const container = document.getElementById('news-list');
        container.innerHTML = news
            .filter(item => this.filterNews(item))
            .map(item => this.createNewsCard(item))
            .join('');
    }
    
    createNewsCard(news) {
        const sentimentClass = news.sentiment === 'positive' ? 'positive' : 
                             news.sentiment === 'negative' ? 'negative' : 'neutral';
        
        return `
            <div class="news-card ${sentimentClass}">
                <div class="news-header">
                    <h3>${news.title}</h3>
                    <span class="sentiment-badge">${news.sentiment}</span>
                </div>
                <div class="news-content">
                    <p>${news.summary}</p>
                    <div class="related-stocks">
                        ${news.related_stocks.map(stock => 
                            `<span class="stock-tag">${stock}</span>`
                        ).join('')}
                    </div>
                </div>
                <div class="news-footer">
                    <span class="news-time">${this.formatTime(news.published_at)}</span>
                    <a href="${news.link}" target="_blank">자세히 보기</a>
                </div>
            </div>
        `;
    }
}
</script>
```

---

### MY 투자

#### 개요
개인 투자 포트폴리오 관리 및 성과 분석

#### 핵심 기능
1. **포트폴리오 관리**
   - 보유 종목 현황
   - 매수/매도 기록
   - 수익률 계산

2. **성과 분석**
   - 일/월/년 수익률
   - 벤치마크 비교
   - 리스크 분석

#### 구현 예시
```python
# web/services/portfolio_service.py
class PortfolioService:
    def get_portfolio_summary(self, user_id):
        """포트폴리오 요약"""
        holdings = self.get_user_holdings(user_id)
        
        # 현재 가치 계산
        total_value = 0
        total_profit = 0
        
        for holding in holdings:
            current_price = self.get_current_price(holding['ticker'])
            holding['current_value'] = current_price * holding['quantity']
            holding['profit_loss'] = (current_price - holding['avg_price']) * holding['quantity']
            holding['profit_rate'] = ((current_price - holding['avg_price']) / holding['avg_price']) * 100
            
            total_value += holding['current_value']
            total_profit += holding['profit_loss']
        
        return {
            'holdings': holdings,
            'total_value': total_value,
            'total_invested': total_value - total_profit,
            'total_profit': total_profit,
            'profit_rate': (total_profit / (total_value - total_profit)) * 100 if total_value > total_profit else 0,
            'asset_allocation': self.calculate_asset_allocation(holdings),
            'risk_metrics': self.calculate_risk_metrics(holdings)
        }
    
    def add_transaction(self, user_id, transaction):
        """거래 추가"""
        # 거래 검증
        if not self.validate_transaction(transaction):
            raise ValueError("유효하지 않은 거래입니다")
        
        # 거래 저장
        db.execute("""
            INSERT INTO transactions 
            (user_id, ticker, type, quantity, price, date) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            transaction['ticker'],
            transaction['type'],
            transaction['quantity'],
            transaction['price'],
            transaction['date']
        ))
        
        # 포트폴리오 업데이트
        self.update_holdings(user_id, transaction)
```

#### 포트폴리오 차트 구현
```javascript
// static/js/portfolio-chart.js
class PortfolioChart {
    constructor() {
        this.charts = {};
        this.init();
    }
    
    init() {
        this.createAssetAllocationChart();
        this.createPerformanceChart();
        this.loadPortfolioData();
    }
    
    createAssetAllocationChart() {
        const ctx = document.getElementById('asset-allocation-chart').getContext('2d');
        this.charts.allocation = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#FF6384',
                        '#36A2EB',
                        '#FFCE56',
                        '#4BC0C0',
                        '#9966FF'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${percentage}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    async loadPortfolioData() {
        const response = await fetch('/api/portfolio/summary');
        const data = await response.json();
        
        this.updateCharts(data);
        this.updateSummaryCards(data);
    }
    
    updateCharts(data) {
        // 자산 배분 차트 업데이트
        const allocation = data.asset_allocation;
        this.charts.allocation.data.labels = Object.keys(allocation);
        this.charts.allocation.data.datasets[0].data = Object.values(allocation);
        this.charts.allocation.update();
        
        // 성과 차트 업데이트
        this.updatePerformanceChart(data.performance_history);
    }
}
```

---

## 🔧 공통 컴포넌트 및 서비스

### 인증 서비스
```python
# web/services/auth_service.py
class AuthService:
    def create_session(self, user_id):
        """세션 생성"""
        session_id = str(uuid.uuid4())
        session['user_id'] = user_id
        session['session_id'] = session_id
        session['created_at'] = datetime.now().isoformat()
        
        # Redis에 세션 저장
        self.redis_client.setex(
            f"session:{session_id}",
            86400,  # 24시간
            json.dumps({
                'user_id': user_id,
                'created_at': session['created_at']
            })
        )
        
        return session_id
    
    def validate_session(self, session_id):
        """세션 검증"""
        session_data = self.redis_client.get(f"session:{session_id}")
        return json.loads(session_data) if session_data else None
```

### 데이터 캐싱 서비스
```python
# web/services/cache_service.py
class CacheService:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def cache_data(self, key, data, ttl=3600):
        """데이터 캐싱"""
        self.redis_client.setex(key, ttl, json.dumps(data))
    
    def get_cached_data(self, key):
        """캐시 데이터 조회"""
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    @functools.lru_cache(maxsize=128)
    def get_or_compute(self, key, compute_func, ttl=3600):
        """캐시 또는 계산"""
        cached = self.get_cached_data(key)
        if cached:
            return cached
        
        result = compute_func()
        self.cache_data(key, result, ttl)
        return result
```

---

## 📊 데이터베이스 설계

### 주요 테이블 구조

```sql
-- 사용자 프로필
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) UNIQUE NOT NULL,
    risk_tolerance INTEGER,
    investment_horizon INTEGER,
    financial_goals TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 포트폴리오
CREATE TABLE portfolio (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price DECIMAL(10,2) NOT NULL,
    purchase_date DATE NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

-- 거래 기록
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    type VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    date TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

-- 채팅 기록
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

-- 학습 진도
CREATE TABLE learning_progress (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    module_id INTEGER NOT NULL,
    lesson_id INTEGER NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMP,
    quiz_score DECIMAL(5,2),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);
```

---

## 🔌 API 엔드포인트 정의

### 인증 관련
- `POST /api/auth/login` - 로그인
- `POST /api/auth/logout` - 로그아웃
- `GET /api/auth/session` - 세션 확인

### 대시보드
- `GET /api/dashboard/summary` - 대시보드 요약
- `GET /api/market/summary` - 시장 현황
- `GET /api/portfolio/summary` - 포트폴리오 요약

### 설문
- `GET /api/survey/questions` - 설문 질문 조회
- `POST /api/survey/submit` - 설문 제출
- `GET /api/survey/result/{user_id}` - 설문 결과 조회

### 학습
- `GET /api/learning/modules` - 학습 모듈 목록
- `GET /api/learning/lesson/{id}` - 강의 콘텐츠
- `POST /api/learning/complete` - 강의 완료
- `GET /api/learning/quiz/{id}` - 퀴즈 조회
- `POST /api/learning/quiz/submit` - 퀴즈 제출

### AI 챗봇
- `POST /api/chat/send` - 메시지 전송
- `GET /api/chat/history` - 대화 기록
- `WS /ws/chat` - WebSocket 연결

### 주가 예측
- `GET /api/stock/prediction/{ticker}` - 주가 예측
- `GET /api/stock/technical/{ticker}` - 기술적 분석
- `GET /api/stock/search` - 종목 검색

### 뉴스
- `GET /api/news/latest` - 최신 뉴스
- `GET /api/news/analysis` - 뉴스 분석
- `POST /api/news/keywords` - 키워드 설정

### 포트폴리오
- `GET /api/portfolio/holdings` - 보유 종목
- `POST /api/portfolio/transaction` - 거래 추가
- `GET /api/portfolio/performance` - 성과 분석

---

## 🔐 보안 및 인증

### 보안 설정
```python
# web/config.py
class SecurityConfig:
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')
    SESSION_COOKIE_SECURE = True  # HTTPS only
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    
    # CORS 설정
    CORS_ORIGINS = ['http://localhost:3000']
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = "redis://localhost:6379"
    RATELIMIT_DEFAULT = "100 per hour"
```

### 인증 데코레이터
```python
# web/blueprints/utils/decorators.py
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': '로그인이 필요합니다'}), 401
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(max_calls=100, window=3600):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = session.get('user_id', request.remote_addr)
            key = f"rate_limit:{f.__name__}:{user_id}"
            
            try:
                current = redis_client.incr(key)
                if current == 1:
                    redis_client.expire(key, window)
                
                if current > max_calls:
                    return jsonify({'error': '요청 한도 초과'}), 429
                    
            except Exception as e:
                logger.error(f"Rate limit error: {e}")
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator
```

---

## 🧪 테스트 전략

### 단위 테스트
```python
# tests/test_services.py
import pytest
from web.services.portfolio_service import PortfolioService

class TestPortfolioService:
    def setup_method(self):
        self.service = PortfolioService()
        self.test_user_id = "test_user_123"
    
    def test_calculate_profit_loss(self):
        """수익률 계산 테스트"""
        holdings = [{
            'ticker': 'AAPL',
            'quantity': 10,
            'avg_price': 150.0,
            'current_price': 170.0
        }]
        
        result = self.service.calculate_profit_loss(holdings)
        
        assert result['total_profit'] == 200.0
        assert result['profit_rate'] == pytest.approx(13.33, 0.01)
    
    def test_empty_portfolio(self):
        """빈 포트폴리오 테스트"""
        result = self.service.get_portfolio_summary("empty_user")
        
        assert result['total_value'] == 0
        assert result['holdings'] == []
```

### 통합 테스트
```python
# tests/test_integration.py
class TestChatbotIntegration:
    def test_full_chat_flow(self, client):
        """전체 챗봇 플로우 테스트"""
        # 1. 세션 생성
        response = client.post('/api/auth/login', json={
            'user_id': 'test_user'
        })
        assert response.status_code == 200
        
        # 2. 메시지 전송
        response = client.post('/api/chat/send', json={
            'message': '삼성전자에 투자해도 될까요?'
        })
        assert response.status_code == 200
        assert 'response' in response.json
        
        # 3. 대화 기록 확인
        response = client.get('/api/chat/history')
        assert response.status_code == 200
        assert len(response.json['messages']) > 0
```

---

## 🚀 배포 가이드

### 환경 설정
```bash
# .env 파일
FLASK_ENV=production
FLASK_SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:pass@localhost/pixie
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=your-openai-key
```

### Docker 설정
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "web.app:app", "-b", "0.0.0.0:5000"]
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://pixie:password@db/pixie
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=pixie
      - POSTGRES_USER=pixie
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:6
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### 배포 명령어
```bash
# 빌드 및 실행
docker-compose up -d

# 데이터베이스 마이그레이션
docker-compose exec web python -m flask db upgrade

# 로그 확인
docker-compose logs -f web
```

---

## 📋 체크리스트

### 개발 전 준비사항
- [ ] Python 3.9+ 설치
- [ ] Node.js 14+ 설치 (프론트엔드 도구)
- [ ] PostgreSQL 설치
- [ ] Redis 설치
- [ ] OpenAI API 키 발급

### 개발 진행 사항
- [ ] 프로젝트 구조 설정
- [ ] 데이터베이스 스키마 생성
- [ ] 기본 라우트 구현
- [ ] 각 페이지 UI 구현
- [ ] 백엔드 서비스 구현
- [ ] AI 기능 통합
- [ ] 테스트 작성
- [ ] 보안 설정
- [ ] 배포 준비

### 테스트 체크리스트
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] UI/UX 테스트
- [ ] 성능 테스트
- [ ] 보안 테스트

---

이 문서는 Pixie 투자 챗봇의 전체 개발 가이드입니다. 각 섹션을 참고하여 체계적으로 개발을 진행하시기 바랍니다.