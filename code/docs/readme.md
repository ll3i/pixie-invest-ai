# MInerva AI Private Banker

이 프로젝트는 대규모 언어 모델(LLM)을 활용하여 사용자 개인 성향 맞춤형 AI Private Banker 서비스를 제공합니다.

MINERVA은 개인화된 투자 조언을 제공하여 사용자의 금융 투자를 돕습니다.

## 파일 설명

### 1. `analysis_results.json`

- `user_survey.py`에서 실행된 사용자의 투자 성향을 분석한 결과를 저장합니다.
- 사용자의 투자 성향을 기반으로 `AI-A`가 `AI-A2`에게 전달하여 사용자에게 맞춤형 서비스를 제공합니다.

### 2. `long_term_memory.json`

- 사용자와의 장기적인 상호작용 기록을 저장하는 파일입니다.
- 사용자의 입력과 그에 대한 AI의 응답을 지속적으로 기록하여 `AI-A`에게 제공합니다.

### 3. `user_survey.py`

- 초기 설정을 로드하고, 사용자 입력을 처리하며, 각 모듈을 호출하여 전체 서비스를 실행합니다.
- `analysis_results.json`, `long_term_memory.json` 파일과 상호작용하며, `main.py`와 함께 작동합니다.

### 4. `minerva.py`

- Multi-Agent System을 구현하기 위한 파일입니다.
- `user_survey.py`와 함께 사용되며, `AI-A`와 `AI-A2`, `AI-B`를 호출하여 사용자에게 맞춤형 서비스를 제공합니다.
- 특정 기능을 독립적으로 실행하여 테스트하거나 보조적인 작업을 수행합니다.

### 5. `main.py`

- `user_survey.py`와 `minerva.py`를 실행할 때 사용되는 모듈입니다.

### 6-1. `prompt_survery-score.txt`

- 사용자의 투자 성향을 분석하기 위한 프롬프트 텍스트 파일입니다.

### 6-2. `prompt_survery-analysis.txt`

- 사용자의 투자 성향을 분석한 결과를 기반으로 AI-A2에게 전달하기 위한 프롬프트 텍스트 파일입니다.

### 6-3. `prompt_AI-A.txt`

- `AI-A`를 위한 프롬프트 텍스트 파일입니다.

### 6-4. `prompt_AI-A2.txt`

- `AI-B`와의 상요작용을 위한 `AI-A2`의 프롬프트 텍스트 파일입니다.

### 6-5. `prompt_AI-B.txt`

- `AI-A2`와의 상호작용을 위한 `AI-B`를 위한 프롬프트 텍스트 파일입니다.

### 7. `data_processing.ipynb`

- `Minerva` 서비스에 사용되는 데이터를 수집하는 과정이 기록된 파일입니다.
- 데이터 크롤링 및 전처리, 파생변수 생성, 데이터 임베딩 과정을 구분지어 제공합니다.
