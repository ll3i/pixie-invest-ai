# Pixie 설치 및 실행 가이드

## 1. 환경 설정

### Python 가상환경 생성 (권장)
```bash
# 프로젝트 루트 디렉토리에서 실행
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate
```

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 환경변수 설정
1. `.env.example` 파일을 `.env`로 복사
2. `.env` 파일을 열어 필요한 API 키 입력:
   - `OPENAI_API_KEY`: OpenAI API 키 (필수)
   - `FLASK_SECRET_KEY`: 최소 32자 이상의 비밀 키
   - `SUPABASE_URL`, `SUPABASE_KEY`: Supabase 설정 (선택사항)

## 2. 실행 방법

### 방법 1: 배치 파일 사용 (Windows)
```bash
run_pixie_web.bat
```

### 방법 2: 직접 실행
```bash
cd web
python app.py
```

### 방법 3: 루트에서 실행
```bash
python web/app.py
```

## 3. 서비스 접속

웹 브라우저에서 다음 주소로 접속:
```
http://localhost:5000
```

## 4. 주요 기능 확인

1. **홈페이지**: http://localhost:5000/
2. **투자 설문**: http://localhost:5000/survey
3. **AI 챗봇**: http://localhost:5000/chatbot
4. **뉴스**: http://localhost:5000/news
5. **주가 조회**: http://localhost:5000/stock
6. **투자 학습**: http://localhost:5000/learning
7. **MY 투자**: http://localhost:5000/my-investment

## 5. 문제 해결

### 포트 충돌 시
app.py 파일에서 포트 번호 변경:
```python
app.run(debug=True, port=5001)  # 5000 -> 5001로 변경
```

### 모듈 임포트 오류 시
Python 경로 확인:
```python
import sys
print(sys.path)
```

### 데이터베이스 오류 시
- `investment_data.db` 파일이 web 디렉토리에 있는지 확인
- 없으면 자동으로 생성됨

### API 키 오류 시
- `.env` 파일에 올바른 API 키가 입력되었는지 확인
- 환경변수가 제대로 로드되는지 확인

## 6. 개발 모드 vs 프로덕션 모드

### 개발 모드 (기본)
```bash
FLASK_ENV=development
```
- 디버그 모드 활성화
- 자동 리로드 활성화

### 프로덕션 모드
```bash
FLASK_ENV=production
```
- 디버그 모드 비활성화
- 성능 최적화

## 7. 데이터 업데이트

### 수동 데이터 수집
```bash
python src/main.py --update-data all
```

### 뉴스 데이터 업데이트
```bash
python src/main.py --update-data news
```

## 8. 로그 확인

로그 파일 위치:
- `minerva_YYYYMMDD.log`
- `web/api_service.log`
- `data_update.log`

## 9. 최소 시스템 요구사항

- Python 3.8 이상
- RAM: 4GB 이상
- 디스크: 1GB 이상의 여유 공간
- 인터넷 연결 (API 호출용)

## 10. 지원

문제가 발생하면 다음을 확인하세요:
1. README.md 파일
2. CLAUDE.md 파일 (상세 기술 문서)
3. 로그 파일
4. .env 설정