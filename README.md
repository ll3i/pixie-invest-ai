# Pixie Investment Advisor 🤖💼

AI 기반 투자 자문 시스템 - 한국 주식 시장 분석, 뉴스 감정 분석, 포트폴리오 추천

## 📊 프로젝트 개요

Pixie Investment Advisor는 인공지능을 활용한 종합적인 투자 자문 시스템입니다. 한국 주식 시장의 실시간 데이터 분석, 뉴스 감정 분석, 그리고 개인화된 포트폴리오 추천을 제공합니다.

### 🎯 주요 기능

- **📈 실시간 주식 데이터 분석**
- **📰 뉴스 감정 분석 및 영향도 평가**
- **🎯 개인화된 포트폴리오 추천**
- **📊 투자 성과 시각화**
- **🤖 AI 챗봇 투자 상담**
- **📚 투자 학습 콘텐츠**

## 🚀 기술 스택

### Backend
- **Python 3.8+**
- **Flask** - 웹 프레임워크
- **SQLAlchemy** - 데이터베이스 ORM
- **Pandas** - 데이터 처리
- **NumPy** - 수치 계산

### AI/ML
- **Transformers** - 자연어 처리
- **Scikit-learn** - 머신러닝
- **TensorFlow/PyTorch** - 딥러닝
- **KoBERT** - 한국어 언어 모델

### Frontend
- **HTML5/CSS3/JavaScript**
- **Chart.js** - 데이터 시각화
- **Bootstrap** - UI 프레임워크

### Database
- **SQLite** - 로컬 데이터베이스
- **PostgreSQL** - 프로덕션 데이터베이스

## 📁 프로젝트 구조

```
pixie-investment-advisor/
├── src/                    # 메인 소스 코드
│   ├── main.py            # 애플리케이션 진입점
│   ├── api_service.py     # API 서비스
│   ├── data_collector.py  # 데이터 수집기
│   ├── news_analyzer.py   # 뉴스 분석기
│   └── stock_predictor.py # 주식 예측 모델
├── web/                   # 웹 애플리케이션
│   ├── app.py            # Flask 앱
│   ├── templates/        # HTML 템플릿
│   ├── static/          # 정적 파일
│   └── blueprints/      # Flask 블루프린트
├── data/                 # 데이터 파일
│   ├── raw/             # 원시 데이터
│   └── processed/       # 처리된 데이터
├── models/              # AI 모델 파일
├── tests/               # 테스트 코드
└── docs/               # 문서
```

## 🛠️ 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/[username]/pixie-investment-advisor.git
cd pixie-investment-advisor
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정
```bash
cp env_example.txt .env
# .env 파일을 편집하여 API 키 등을 설정
```

### 5. 데이터베이스 초기화
```bash
python src/database_migrator.py
```

### 6. 애플리케이션 실행
```bash
# 개발 서버
python src/main.py

# 웹 애플리케이션
cd web
python app.py
```

## 📊 주요 기능 설명

### 1. 주식 데이터 분석
- 실시간 주가 데이터 수집
- 기술적 지표 계산
- 시장 동향 분석

### 2. 뉴스 감정 분석
- 한국어 뉴스 텍스트 분석
- 감정 점수 계산
- 주가 영향도 예측

### 3. 포트폴리오 추천
- 개인 투자 성향 분석
- 리스크 관리
- 최적 자산 배분

### 4. AI 챗봇
- 자연어 투자 상담
- 실시간 질의응답
- 투자 가이드 제공

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

- **프로젝트 링크**: [https://github.com/[username]/pixie-investment-advisor](https://github.com/[username]/pixie-investment-advisor)
- **이슈 리포트**: [https://github.com/[username]/pixie-investment-advisor/issues](https://github.com/[username]/pixie-investment-advisor/issues)

## 🙏 감사의 말

- 한국투자증권 API
- 네이버 뉴스 API
- OpenAI GPT 모델
- Hugging Face Transformers

---

⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요! 