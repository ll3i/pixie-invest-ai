# GitHub Push 가이드

## GitHub 저장소 생성 및 Push 방법

### 1. GitHub에서 새 저장소 생성

1. https://github.com 로그인
2. 우측 상단 '+' 버튼 → 'New repository' 클릭
3. 저장소 정보 입력:
   - Repository name: `pixie-investment-advisor`
   - Description: `Pixie - AI 기반 맞춤형 투자 자문 시스템`
   - Public 선택
   - **중요**: "Initialize this repository with:" 옵션들은 모두 체크 해제
4. 'Create repository' 클릭

### 2. 로컬 저장소와 GitHub 연결

GitHub에서 제공하는 명령어를 복사하거나 아래 명령어 실행:

```bash
cd "C:\Users\work4\OneDrive\바탕 화면\kb ai challenge\code"

# GitHub 원격 저장소 추가 (your-username을 실제 GitHub 사용자명으로 변경)
git remote add origin https://github.com/your-username/pixie-investment-advisor.git

# 기본 브랜치 이름을 main으로 변경 (선택사항)
git branch -M main

# GitHub에 Push
git push -u origin main
```

### 3. GitHub 인증

Push 시 인증이 필요합니다:

#### 방법 1: Personal Access Token (권장)
1. GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 'Generate new token' 클릭
3. 권한 선택: `repo` 전체 체크
4. 생성된 토큰 복사
5. Push 시 비밀번호 대신 이 토큰 사용

#### 방법 2: GitHub Desktop 사용
1. [GitHub Desktop](https://desktop.github.com/) 다운로드
2. 로컬 저장소 추가
3. GUI를 통해 Push

### 4. Push 확인

1. GitHub 저장소 페이지 새로고침
2. 모든 파일이 업로드되었는지 확인
3. README.md가 메인 페이지에 표시되는지 확인

### 5. 추가 설정 (선택사항)

#### About 섹션 설정
1. 저장소 페이지 우측 톱니바퀴 아이콘 클릭
2. Topics 추가: `ai`, `investment`, `flask`, `python`, `korean-stock`
3. Website: 배포된 URL 입력 (있을 경우)

#### 기본 브랜치 보호
1. Settings → Branches
2. Add rule → Branch name pattern: `main`
3. 필요한 보호 규칙 설정

## 문제 해결

### Push 실패 시
```bash
# 원격 저장소 확인
git remote -v

# 원격 저장소 URL 변경 (필요시)
git remote set-url origin https://github.com/your-username/pixie-investment-advisor.git

# 강제 Push (주의: 기존 내용 덮어쓰기)
git push -f origin main
```

### 대용량 파일 문제
```bash
# 100MB 이상 파일 확인
find . -type f -size +100M

# Git LFS 사용 (필요시)
git lfs track "*.pkl"
git lfs track "*.db"
```

## 완료 체크리스트

- [ ] GitHub 저장소 생성
- [ ] 로컬 저장소와 연결
- [ ] 첫 Push 성공
- [ ] README.md 표시 확인
- [ ] .gitignore 작동 확인 (.env 파일 제외)
- [ ] About 섹션 설정
- [ ] 저장소 공개 설정 확인