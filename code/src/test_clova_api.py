import requests
import json
import os
import time

class CompletionExecutor:
    def __init__(self, host, api_key, request_id, max_retries=3, retry_delay=2):
        self._host = host
        self._api_key = api_key  # API 키는 "Bearer ..." 형식이어야 합니다.
        self._request_id = request_id
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def execute(self, messages):
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": self._api_key,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
        }
        data = {
            "messages": messages,
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 2048,
            "temperature": 0.1,
            "repeatPenalty": 1.1,
            "stopBefore": [],
            "includeAiFilters": False,
        }
        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    f"{self._host}/testapp/v1/chat-completions/HCX-003",
                    headers=headers,
                    json=data,
                )
                result = response.json()
                print(f"API 응답: {result}")
                
                if ("result" in result and result["result"] is not None and "message" in result["result"]):
                    return result["result"]["message"]["content"]
                elif "status" in result and result["status"]["code"] == "42901":
                    print(f"API 속도 제한 감지: {attempt+1}/{self._max_retries} 번째 시도, {self._retry_delay}초 대기 후 재시도")
                    time.sleep(self._retry_delay)
                else:
                    print(f"예상치 못한 API 응답: {result}")
                    if attempt < self._max_retries - 1:
                        print(f"{self._retry_delay}초 대기 후 재시도")
                        time.sleep(self._retry_delay)
                    else:
                        raise ValueError(f"API 응답 오류: {result}")
            except requests.exceptions.RequestException as e:
                print(f"API 요청 오류: {e}")
                if attempt < self._max_retries - 1:
                    print(f"{self._retry_delay}초 대기 후 재시도")
                    time.sleep(self._retry_delay)
                else:
                    raise ValueError(f"API 요청 실패: {e}")
                
        raise ValueError(f"{self._max_retries}번 시도 후 유효한 응답을 얻지 못했습니다.")

def load_prompt_template(prompt_file="prompt_survey-score.txt", src_dir=None):
    """프롬프트 템플릿을 로드합니다."""
    if src_dir is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        prompt_path = os.path.join(src_dir, prompt_file)
        if not os.path.exists(prompt_path):
            print(f"경고: 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return None
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"프롬프트 파일 로드 중 오류 발생: {e}")
        return None

def test_survey_score_api():
    """CLOVA Studio API를 테스트하여 설문 분석 결과를 확인합니다."""
    # API 설정
    host = "https://clovastudio.stream.ntruss.com"
    api_key = "Bearer nv-e302186b2e7640d38c732700bd828020zBct"
    request_id = "3d487f4d478e4a2a8a86d7d4fe509b76"
    
    # API 실행기 생성
    executor = CompletionExecutor(
        host=host,
        api_key=api_key,
        request_id=request_id
    )
    
    # 프롬프트 템플릿 로드
    prompt_template = load_prompt_template("prompt_survey-score.txt")
    if not prompt_template:
        print("프롬프트 템플릿을 로드할 수 없습니다.")
        return
    
    # 테스트용 질문과 답변
    test_question = "큰 꿈을 안고 시작한 투자, 여러분은 투자를 할 때 가장 중요한 요소가 무엇이라고 생각하나요?"
    test_answer = "저는 장기적인 성장 가능성을 가장 중요하게 생각합니다. 단기적인 변동성보다는 기업의 기본 가치와 미래 성장 잠재력을 보고 투자하는 것이 중요하다고 생각해요. 물론 리스크 관리도 중요하지만, 장기적인 관점에서 보면 좋은 기업에 투자하고 기다리는 것이 가장 좋은 전략이라고 봅니다."
    
    # 프롬프트 포맷팅
    formatted_prompt = prompt_template.replace("[question]", test_question).replace("[answer]", test_answer)
    
    # API 호출
    messages = [
        {"role": "system", "content": "너는 사용자의 요청을 엄격하게 json형태에 맞게 응답하는 AI야."},
        {"role": "user", "content": formatted_prompt},
    ]
    
    print("API 호출 중...")
    try:
        response = executor.execute(messages)
        print("\n응답 결과:")
        print(response)
        
        # JSON 형식으로 파싱 시도
        try:
            scores = json.loads(response)
            print("\n분석 결과 요약:")
            print(f"위험 감수성 (Risk Tolerance): {scores.get('risk_tolerance', 'N/A')}")
            print(f"투자 시간 범위 (Investment Time Horizon): {scores.get('investment_time_horizon', 'N/A')}")
            print(f"재무 목표 지향성 (Financial Goal Orientation): {scores.get('financial_goal_orientation', 'N/A')}")
            print(f"정보 처리 스타일 (Information Processing Style): {scores.get('information_processing_style', 'N/A')}")
            print(f"투자 두려움 (Investment Fear): {scores.get('investment_fear', 'N/A')}")
            print(f"투자 자신감 (Investment Confidence): {scores.get('investment_confidence', 'N/A')}")
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print("원본 응답:", response)
    except Exception as e:
        print(f"API 호출 오류: {e}")

if __name__ == "__main__":
    test_survey_score_api() 