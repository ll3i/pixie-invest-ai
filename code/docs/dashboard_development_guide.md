# 📊 Pixie 대시보드 개발 가이드

## 🎯 개요

Pixie 투자챗봇의 메인 대시보드(dashboard.html)는 사용자가 처음 접하는 랜딩 페이지로, 서비스의 핵심 가치를 전달하고 주요 기능으로의 진입점을 제공합니다.

## 🏗️ 파일 구조

### 핵심 파일
- **템플릿**: `web/templates/dashboard.html` (1322줄)
- **백엔드**: `web/app.py` - 라우트 설정
- **스타일**: 인라인 CSS + Bootstrap 5
- **스크립트**: jQuery 기반 동적 기능

## 🎨 주요 구성 요소

### 1. 커스텀 스타일 설정 (3-274줄)

```css
/* Pixie 디자인 시스템 변수 */
:root {
    --primary-color: #7c3aed;      /* 보라색 메인 */
    --secondary-color: #a78bfa;    /* 연보라 */
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

/* 애니메이션 효과 */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
```

**주요 특징**:
- CSS 변수를 통한 일관된 컬러 시스템
- 부드러운 애니메이션 효과 정의
- 반응형 디자인을 위한 미디어 쿼리

### 2. 히어로 섹션 (278-330줄)

```html
<section class="hero-section" id="home">
    <div class="container">
        <div class="row align-items-center min-vh-100">
            <div class="col-lg-6" data-aos="fade-right">
                <h1 class="display-4 fw-bold mb-4 gradient-text">
                    당신의 투자,<br>
                    <span class="highlight">Pixie</span>가 함께합니다
                </h1>
                <p class="lead mb-4">
                    AI 기반 개인화 투자 상담으로<br>
                    더 스마트한 투자 결정을 내리세요
                </p>
                <div class="d-flex gap-3">
                    <a href="/survey" class="btn btn-primary btn-lg">
                        투자 성향 분석하기
                    </a>
                    <a href="/chatbot" class="btn btn-outline-primary btn-lg">
                        AI 상담 시작하기
                    </a>
                </div>
            </div>
            <div class="col-lg-6" data-aos="fade-left">
                <div class="chart-animation">
                    <!-- 차트 애니메이션 SVG -->
                </div>
            </div>
        </div>
    </div>
</section>
```

**기능 설명**:
- AOS(Animate On Scroll) 라이브러리를 사용한 스크롤 애니메이션
- 그라디언트 텍스트 효과로 시각적 강조
- CTA(Call-to-Action) 버튼으로 핵심 기능 유도

### 3. 이슈종목 추천 섹션 (398-498줄)

```html
<section class="trending-section py-5" id="trending">
    <div class="container">
        <h2 class="section-title">오늘의 이슈 종목</h2>
        <div class="row g-4" id="trending-stocks">
            <!-- 동적으로 생성되는 종목 카드 -->
        </div>
    </div>
</section>
```

**동적 종목 카드 생성 JavaScript**:
```javascript
// 이슈 종목 데이터 (실제로는 API에서 가져옴)
const trendingStocks = [
    {
        name: '삼성전자',
        code: '005930',
        price: '70,500',
        change: '+2.5%',
        volume: '15,234,567',
        reason: 'AI 반도체 수요 증가'
    },
    // ... 더 많은 종목
];

// 종목 카드 렌더링
function renderTrendingStocks() {
    const container = $('#trending-stocks');
    container.empty();
    
    trendingStocks.forEach(stock => {
        const card = `
            <div class="col-md-6 col-lg-4">
                <div class="stock-card">
                    <div class="stock-header">
                        <h4>${stock.name}</h4>
                        <span class="stock-code">${stock.code}</span>
                    </div>
                    <div class="stock-body">
                        <div class="price-info">
                            <span class="price">${stock.price}원</span>
                            <span class="change ${stock.change.startsWith('+') ? 'up' : 'down'}">
                                ${stock.change}
                            </span>
                        </div>
                        <div class="stock-meta">
                            <span>거래량: ${stock.volume}</span>
                        </div>
                        <div class="stock-reason">
                            <i class="fas fa-lightbulb"></i> ${stock.reason}
                        </div>
                    </div>
                </div>
            </div>
        `;
        container.append(card);
    });
}
```

### 4. 트렌드 맵 섹션 (820-919줄)

```html
<section class="trend-map-section py-5">
    <div class="container">
        <h2 class="section-title">실시간 투자 트렌드</h2>
        <div class="trend-map-container">
            <div class="trend-tags-wrapper">
                <div class="trend-tags" id="trend-tags">
                    <!-- 동적 트렌드 태그 -->
                </div>
            </div>
        </div>
    </div>
</section>
```

**트렌드 태그 애니메이션 구현**:
```javascript
$(document).ready(function() {
    // 트렌드 키워드 데이터
    const trendKeywords = [
        { text: 'AI 반도체', size: 'large', heat: 95 },
        { text: '2차전지', size: 'medium', heat: 85 },
        { text: '바이오', size: 'small', heat: 70 },
        { text: 'K-방산', size: 'large', heat: 90 },
        { text: '수소경제', size: 'medium', heat: 80 },
        // ... 더 많은 키워드
    ];
    
    // 태그 생성 및 애니메이션
    function createTrendTags() {
        const container = $('#trend-tags');
        
        trendKeywords.forEach((keyword, index) => {
            const delay = index * 0.1;
            const tag = $(`
                <div class="trend-tag ${keyword.size}" 
                     style="animation-delay: ${delay}s"
                     data-heat="${keyword.heat}">
                    <span>${keyword.text}</span>
                    <div class="heat-indicator" style="width: ${keyword.heat}%"></div>
                </div>
            `);
            
            container.append(tag);
        });
        
        // 무한 스크롤 애니메이션
        animateScroll();
    }
    
    function animateScroll() {
        const wrapper = $('.trend-tags');
        const scrollWidth = wrapper[0].scrollWidth;
        const clientWidth = wrapper[0].clientWidth;
        
        // CSS 애니메이션으로 부드러운 스크롤
        wrapper.css({
            'animation': `scroll ${scrollWidth / 50}s linear infinite`
        });
    }
    
    // 키워드 클릭 이벤트
    $(document).on('click', '.trend-tag', function() {
        const keyword = $(this).find('span').text();
        window.location.href = `/news?keyword=${encodeURIComponent(keyword)}`;
    });
});
```

### 5. 서비스 소개 섹션 (549-763줄)

```html
<section class="services-section py-5" id="services">
    <div class="container">
        <h2 class="section-title">Pixie와 함께하는 스마트한 투자</h2>
        <div class="row g-4">
            <!-- 서비스 카드들 -->
            <div class="col-md-6 col-lg-4" data-aos="fade-up">
                <div class="service-card h-100">
                    <div class="service-icon">
                        <i class="fas fa-chart-line"></i>
                    </div>
                    <h3>AI 투자 분석</h3>
                    <p>딥러닝 기반 시장 분석으로 최적의 투자 시점을 포착합니다</p>
                    <a href="/stock" class="btn btn-sm btn-outline-primary">
                        자세히 보기
                    </a>
                </div>
            </div>
            <!-- 더 많은 서비스 카드... -->
        </div>
    </div>
</section>
```

### 6. 네비게이션 바 (1041-1123줄)

```html
<nav class="navbar navbar-expand-lg fixed-top" id="mainNav">
    <div class="container">
        <a class="navbar-brand" href="#home">
            <img src="/static/images/logo.png" alt="Pixie" height="40">
        </a>
        <button class="navbar-toggler" type="button" 
                data-bs-toggle="collapse" 
                data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
                <li class="nav-item">
                    <a class="nav-link" href="#home">홈</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#services">서비스</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#trending">트렌드</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="/chatbot">AI 상담</a>
                </li>
            </ul>
        </div>
    </div>
</nav>
```

**스크롤 기반 네비게이션 스타일 변경**:
```javascript
// 스크롤 시 네비게이션 바 스타일 변경
$(window).scroll(function() {
    const scrollTop = $(this).scrollTop();
    const navbar = $('#mainNav');
    
    if (scrollTop > 100) {
        navbar.addClass('navbar-scrolled');
    } else {
        navbar.removeClass('navbar-scrolled');
    }
});

// 부드러운 스크롤 구현
$('a[href^="#"]').on('click', function(e) {
    e.preventDefault();
    const target = $($(this).attr('href'));
    
    if (target.length) {
        $('html, body').animate({
            scrollTop: target.offset().top - 80
        }, 800, 'easeInOutQuad');
    }
});
```

## 🔧 백엔드 연동 (app.py)

### 대시보드 라우트 설정

```python
@app.route('/')
def dashboard():
    """메인 대시보드 페이지"""
    try:
        # 실시간 데이터 수집
        trending_stocks = get_trending_stocks()
        market_summary = get_market_summary()
        trend_keywords = get_trend_keywords()
        
        return render_template('dashboard.html',
                             trending_stocks=trending_stocks,
                             market_summary=market_summary,
                             trend_keywords=trend_keywords)
    except Exception as e:
        app.logger.error(f"대시보드 로드 오류: {e}")
        return render_template('dashboard.html',
                             trending_stocks=[],
                             market_summary={},
                             trend_keywords=[])

def get_trending_stocks():
    """이슈 종목 데이터 조회"""
    # 실제로는 데이터베이스나 API에서 가져옴
    return [
        {
            'name': '삼성전자',
            'code': '005930',
            'price': 70500,
            'change_rate': 2.5,
            'volume': 15234567,
            'reason': 'AI 반도체 수요 증가'
        },
        # ... 더 많은 종목
    ]

def get_trend_keywords():
    """실시간 트렌드 키워드 조회"""
    # 뉴스 분석 결과나 검색 트렌드에서 추출
    return [
        {'keyword': 'AI 반도체', 'heat': 95, 'count': 1234},
        {'keyword': '2차전지', 'heat': 85, 'count': 987},
        # ... 더 많은 키워드
    ]
```

## 🎨 디자인 시스템

### 색상 팔레트
- **Primary**: #7c3aed (보라색)
- **Secondary**: #a78bfa (연보라)
- **Success**: #10b981 (녹색)
- **Danger**: #ef4444 (빨간색)
- **Background**: #f8f9fa (연회색)

### 타이포그래피
- **제목**: Pretendard, sans-serif
- **본문**: -apple-system, BlinkMacSystemFont, 시스템 폰트
- **크기**: 반응형 rem 단위 사용

### 애니메이션
- **Fade In**: 요소 등장 효과
- **Float**: 부유하는 효과
- **Scroll**: 무한 스크롤 효과

## 📱 반응형 디자인

### 브레이크포인트
- **Mobile**: < 576px
- **Tablet**: 576px - 992px
- **Desktop**: > 992px

### 모바일 최적화
```css
@media (max-width: 768px) {
    .hero-section h1 {
        font-size: 2rem;
    }
    
    .stock-card {
        margin-bottom: 1rem;
    }
    
    .trend-tag {
        font-size: 0.875rem;
    }
}
```

## 🚀 성능 최적화

### 1. 이미지 최적화
- WebP 포맷 사용
- Lazy loading 구현
- 적절한 이미지 크기 제공

### 2. JavaScript 최적화
- 디바운싱/쓰로틀링 적용
- 이벤트 위임 사용
- 불필요한 리플로우 최소화

### 3. CSS 최적화
- Critical CSS 인라인 처리
- 사용하지 않는 스타일 제거
- CSS 압축

## 🔒 보안 고려사항

### XSS 방지
- 사용자 입력 데이터 이스케이프
- Content Security Policy 설정

### CSRF 방지
- Flask-WTF CSRF 토큰 사용
- 안전한 쿠키 설정

## 📈 추후 개선 계획

### 단기 (1-2개월)
1. 실시간 주가 업데이트 (WebSocket)
2. 다크 모드 지원
3. PWA 지원

### 중기 (3-6개월)
1. 차트 라이브러리 통합 (Chart.js)
2. 개인화된 대시보드
3. 위젯 시스템

### 장기 (6개월+)
1. 대시보드 커스터마이징
2. AI 기반 인사이트 제공
3. 소셜 기능 통합

## 🧪 테스트

### 단위 테스트
```python
def test_dashboard_route():
    """대시보드 라우트 테스트"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Pixie' in response.data
```

### E2E 테스트
```javascript
describe('Dashboard Page', () => {
    it('should load trending stocks', () => {
        cy.visit('/');
        cy.get('.stock-card').should('have.length.greaterThan', 0);
    });
    
    it('should navigate to survey on CTA click', () => {
        cy.visit('/');
        cy.contains('투자 성향 분석하기').click();
        cy.url().should('include', '/survey');
    });
});
```

## 📚 참고 자료

### 사용된 라이브러리
- **Bootstrap 5.3**: UI 프레임워크
- **jQuery 3.6**: DOM 조작 및 이벤트 처리
- **AOS 2.3**: 스크롤 애니메이션
- **Font Awesome 6**: 아이콘

### 디자인 참고
- Material Design 3 가이드라인
- Apple Human Interface Guidelines
- 국내 금융 앱 UI/UX 트렌드

이 문서는 Pixie 대시보드의 개발 과정과 주요 기능 구현을 상세히 설명합니다. 지속적인 개선을 통해 사용자에게 최고의 경험을 제공하는 것이 목표입니다.