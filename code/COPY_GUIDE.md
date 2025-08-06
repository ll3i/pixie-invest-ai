# Pixie 프로젝트 파일 복사 가이드

이 문서는 Pixie 프로젝트의 필수 파일들을 수동으로 복사하는 방법을 안내합니다.

## 복사해야 할 파일 목록

### 1. src 디렉토리 (C:\Users\work4\OneDrive\바탕 화면\투자챗봇\src\)
다음 파일들을 code\src\ 디렉토리로 복사하세요:

**핵심 파일들:**
- main.py
- investment_advisor.py (핵심 AI 체인 - 절대 수정 금지)
- config.py
- llm_service.py
- db_client.py
- financial_data_processor.py
- data_collector.py
- user_profile_analyzer.py
- memory_manager.py
- simplified_portfolio_prediction.py
- advanced_stock_predictor.py
- stock_search_engine.py
- financial_report_analyzer.py
- news_collector_service.py
- user_survey.py
- survey_to_profile.py
- recommend_portfolio.py
- prompt_AI-A.txt
- prompt_AI-A2.txt
- prompt_AI-B.txt

**추가 파일들 (있는 경우):**
- prompt_manager.py
- data_processor.py
- stock_evaluator.py
- korean_stock_data_processor.py
- api_service.py
- data_update_scheduler.py
- response_filter.py

### 2. web 디렉토리 (C:\Users\work4\OneDrive\바탕 화면\투자챗봇\web\)

**메인 파일들을 code\web\로 복사:**
- app.py (Flask 메인 애플리케이션)
- news_api_helper.py
- config.py (web용 설정)
- requirements.txt

**templates 디렉토리 (web\templates\를 code\web\templates\로):**
- layout.html
- index.html
- dashboard.html
- survey.html
- survey_result.html
- result.html
- learning.html
- learning_terms.html
- learning_term.html
- learning_quiz.html
- learning_cardnews.html
- chatbot.html
- stock.html
- time_series_prediction.html
- news.html
- news_enhanced.html
- my-investment.html
- my_invest.html
- alerts.html
- alert_history.html
- market_sentiment.html
- minerva.html

**static 디렉토리 전체 복사:**
- web\static\ → code\web\static\ (하위 폴더 포함)
  - css\style.css
  - css\chatbot-widget.css
  - js\main.js
  - js\chatbot-widget.js
  - js\portfolio-chart.js
  - images\ (모든 이미지 파일)

**services 디렉토리 전체 복사:**
- web\services\ → code\web\services\
  - __init__.py
  - alert_service.py
  - database_service.py
  - learning_service.py
  - survey_service.py
  - user_service.py

### 3. 루트 파일들 (C:\Users\work4\OneDrive\바탕 화면\투자챗봇\)
code\ 디렉토리로 복사:
- start_web_server.bat
- requirements.txt
- CLAUDE.md
- README.md (있는 경우)

### 4. 테스트 파일 (선택사항)
- tests\test_investment_advisor.py → code\tests\

## 복사 후 확인사항

1. **디렉토리 구조 확인:**
   ```
   code/
   ├── src/          (20개 이상의 파일)
   ├── web/
   │   ├── app.py
   │   ├── templates/  (22개 HTML 파일)
   │   ├── static/
   │   └── services/
   ├── tests/
   ├── data/
   │   ├── raw/
   │   └── processed/
   └── docs/
   ```

2. **중요 파일 존재 확인:**
   - src/investment_advisor.py (핵심)
   - src/prompt_AI-*.txt (3개 파일)
   - web/app.py
   - web/templates/chatbot.html

3. **예상 파일 수:**
   - 총 70-80개 파일
   - src: 20-25개
   - web/templates: 22개
   - web/static: 20-30개

## 주의사항

1. **절대 수정하지 말아야 할 파일들:**
   - src/investment_advisor.py의 AI 체인 로직
   - src/prompt_AI-*.txt 파일들

2. **중복 파일 제거:**
   - web/ 루트에 있는 HTML 파일들은 복사하지 마세요
   - web/src/, web/eb_deploy/, web/eb_full_deploy/ 디렉토리는 복사하지 마세요

3. **가상환경 제외:**
   - venv/, web/venv/ 디렉토리는 복사하지 마세요

## 수동 복사 명령어

Windows 탐색기를 사용하거나 다음 명령어를 사용할 수 있습니다:

```batch
xcopy "C:\Users\work4\OneDrive\바탕 화면\투자챗봇\src\*" "C:\Users\work4\OneDrive\바탕 화면\kb ai challenge\code\src\" /E /Y
xcopy "C:\Users\work4\OneDrive\바탕 화면\투자챗봇\web\templates\*" "C:\Users\work4\OneDrive\바탕 화면\kb ai challenge\code\web\templates\" /E /Y
xcopy "C:\Users\work4\OneDrive\바탕 화면\투자챗봇\web\static\*" "C:\Users\work4\OneDrive\바탕 화면\kb ai challenge\code\web\static\" /E /Y
xcopy "C:\Users\work4\OneDrive\바탕 화면\투자챗봇\web\services\*" "C:\Users\work4\OneDrive\바탕 화면\kb ai challenge\code\web\services\" /E /Y
```

이 가이드를 따라 필요한 파일들을 복사하면 Pixie 프로젝트의 핵심 코드가 준비됩니다.