# GitHub Push 가이드 🚀

## 📋 사전 준비사항

### 1. GitHub 계정 확인
- GitHub.com에 로그인되어 있는지 확인
- Personal Access Token이 설정되어 있는지 확인

### 2. Personal Access Token 생성 (필요한 경우)
1. GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token" 클릭
3. 권한 설정:
   - `repo` (전체 저장소 접근)
   - `workflow` (GitHub Actions)
4. 토큰 생성 후 안전한 곳에 저장

## 🔧 GitHub 저장소 생성

### 1. GitHub에서 새 저장소 생성
1. GitHub.com 접속
2. 우측 상단 "+" 버튼 → "New repository" 클릭
3. 저장소 설정:
   - **Repository name**: `pixie-investment-advisor`
   - **Description**: `AI 기반 투자 자문 시스템 - 한국 주식 시장 분석, 뉴스 감정 분석, 포트폴리오 추천`
   - **Visibility**: Public
   - **README 초기화**: ❌ 체크 해제
   - **.gitignore**: ❌ 체크 해제
   - **License**: ❌ 체크 해제

### 2. 저장소 생성 완료
- 저장소 URL 확인: `https://github.com/[your-username]/pixie-investment-advisor`

## 🚀 로컬 저장소 Push

### 1. 원격 저장소 연결
```bash
git remote add origin https://github.com/[your-username]/pixie-investment-advisor.git
```

### 2. 브랜치 이름 확인 및 변경 (필요한 경우)
```bash
# 현재 브랜치 확인
git branch

# 브랜치 이름이 'master'가 아니라면 변경
git branch -M master
```

### 3. Push 실행
```bash
git push -u origin master
```

### 4. 인증 요청 시
- **Username**: GitHub 사용자명
- **Password**: Personal Access Token (비밀번호 아님)

## 🔍 Push 확인

### 1. GitHub 저장소 확인
- 브라우저에서 저장소 페이지 접속
- 파일들이 정상적으로 업로드되었는지 확인

### 2. 커밋 히스토리 확인
```bash
git log --oneline
```

## 📊 업로드된 파일 통계

- **총 파일 수**: 234개
- **주요 구성요소**:
  - 📁 `code/` - 메인 소스 코드
  - 📁 `web/` - Flask 웹 애플리케이션
  - 📁 `src/` - Python 모듈들
  - 📁 `data/` - 데이터 파일들
  - 📁 `models/` - AI 모델 파일들
  - 📄 `README.md` - 프로젝트 문서
  - 📄 `LICENSE` - MIT 라이선스

## 🛠️ 문제 해결

### 1. 인증 오류
```bash
# Personal Access Token 재설정
git config --global credential.helper store
git push -u origin master
# 토큰 입력
```

### 2. 브랜치 충돌
```bash
# 강제 Push (주의: 기존 내용 덮어씀)
git push -u origin master --force
```

### 3. 파일 크기 제한
```bash
# Git LFS 설정 (대용량 파일용)
git lfs install
git lfs track "*.pkl"
git lfs track "*.h5"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## 🎯 다음 단계

### 1. GitHub Pages 설정 (선택사항)
1. 저장소 Settings → Pages
2. Source: Deploy from a branch
3. Branch: master, folder: / (root)
4. Save

### 2. GitHub Actions 설정 (선택사항)
- CI/CD 파이프라인 구성
- 자동 테스트 및 배포

### 3. 프로젝트 위키 설정 (선택사항)
- 상세한 사용법 문서 작성
- API 문서화

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. Personal Access Token이 올바른지
2. 저장소 이름이 정확한지
3. 네트워크 연결 상태
4. Git 설정이 올바른지

---

**성공적으로 Push되면 GitHub에서 프로젝트를 확인할 수 있습니다!** 🎉 