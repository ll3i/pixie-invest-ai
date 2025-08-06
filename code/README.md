# 🤖 Pixie - AI 기반 맞춤형 투자 자문 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 프로젝트 소개

Pixie는 AI 기술을 활용한 개인 맞춤형 투자 자문 시스템입니다. 사용자의 투자 성향을 분석하고, 실시간 시장 데이터를 기반으로 최적의 투자 전략을 제공합니다.

### 🌟 핵심 가치
- **개인화**: 사용자별 맞춤형 투자 전략 제공
- **실시간**: 최신 시장 데이터 기반 분석
- **교육적**: 투자 학습 콘텐츠를 통한 금융 지식 향상
- **접근성**: 초보자도 쉽게 사용할 수 있는 직관적 인터페이스

## 주요 기능

### 1. 투자 설문 (Investment Survey)
- 10가지 질문을 통한 투자 성향 분석
- AI 기반 프로필 생성 및 투자자 유형 분류
- 맞춤형 포트폴리오 추천

### 2. 투자 학습 (Investment Learning)
- 투자 용어 학습
- 퀴즈를 통한 지식 테스트
- 카드뉴스 형식의 교육 콘텐츠

### 3. AI 챗봇 (AI Chatbot)
- 다중 AI 에이전트 체인 (AI-A → AI-A2 → AI-B)
- 실시간 금융 데이터 기반 조언
- 개인화된 투자 상담

### 4. 주가 예측 (Stock Prediction)
- ARIMA-X 모델 기반 주가 예측
- 국내 주식 분석
- 시각화된 예측 차트

### 5. 뉴스/이슈 (News/Issues)
- 실시간 금융 뉴스 수집
- 감정 분석 기반 시장 심리 파악
- 개인화된 뉴스 필터링

### 6. MY 투자 (My Investment)
- 포트폴리오 관리
- 투자 성과 추적
- 리스크 알림 서비스

## 기술 스택

- **Backend**: Python 3.11+, Flask
- **AI/ML**: OpenAI API, ARIMA 모델
- **Database**: Supabase (PostgreSQL) + SQLite
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Deployment**: AWS Elastic Beanstalk

## 🚀 빠른 시작

### 필수 요구사항
- Python 3.8 이상
- OpenAI API Key ([발급 방법](https://platform.openai.com/api-keys))
- 4GB 이상의 RAM

### 설치 방법

1. **저장소 클론**
   ```bash
   git clone https://github.com/[username]/pixie-investment-advisor.git
   cd pixie-investment-advisor
   ```

2. **환경 설정**
   ```bash
   # Windows
   copy .env.example .env
   
   # macOS/Linux
   cp .env.example .env
   ```
   
   `.env` 파일을 열어 필수 API 키 입력:
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key
   FLASK_SECRET_KEY=your-32-character-secret-key-here!!!
   ```

3. **자동 설치 및 실행**
   ```bash
   # Windows
   setup_and_run.bat
   
   # macOS/Linux
   pip install -r requirements.txt
   cd web && python app.py
   ```

4. **웹 브라우저에서 접속**
   ```
   http://localhost:5000
   ```

## 📚 상세 문서

- [🇰🇷 한국어 설치 가이드](설치_가이드.md)
- [🇺🇸 English Installation Guide](INSTALLATION_GUIDE_EN.md)
- [📖 기술 문서](docs/CLAUDE.md)
- [🚀 배포 가이드](DEPLOYMENT_GUIDE.md)

## 프로젝트 구조

```
code/
├── src/                    # 핵심 비즈니스 로직
│   ├── investment_advisor.py    # AI 에이전트 체인
│   ├── llm_service.py          # LLM API 추상화
│   ├── data_collector.py       # 데이터 수집
│   └── ...
├── web/                    # 웹 애플리케이션
│   ├── app.py             # Flask 메인
│   ├── templates/         # HTML 템플릿
│   ├── static/           # CSS, JS, 이미지
│   └── services/         # 서비스 레이어
├── tests/                 # 테스트 코드
└── docs/                  # 문서
```

## 🤝 기여하기

프로젝트에 기여하고 싶으시다면:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👥 팀 정보

- **개발팀**: KB AI Challenge 참가팀
- **프로젝트명**: Pixie
- **개발기간**: 2025.07~08
- **문의**: pixie-support@example.com

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 도움을 받았습니다:
- Flask - 웹 프레임워크
- OpenAI - AI 모델
- pandas - 데이터 처리
- scikit-learn - 머신러닝
- Bootstrap - UI 프레임워크

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**