import pandas as pd
import math
import re
from collections import Counter
import pickle
import requests, json, time

def load_prompt_template():
    try:
        with open('prompt.txt', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print("Error: prompt.txt file not found.")
        return None
    except IOError:
        print("Error: Could not read prompt.txt file.")
        return None

def load_prompt2_template():
    try:
        with open('prompt2.txt', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print("Error: prompt2.txt file not found.")
        return None
    except IOError:
        print("Error: Could not read prompt2.txt file.")
        return None

# 6. Chat Completion 함수 (CLOVA Studio API 사용)
class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id, max_retries=3, retry_delay=10):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def execute(self, messages):
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id
        }

        data = {
            "messages": messages,
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 2048,
            "temperature": 0.1,
            "repeatPenalty": 1.1,
            "stopBefore": [],
            "includeAiFilters": False
        }

        for attempt in range(self._max_retries):
            response = requests.post(
                f"{self._host}/testapp/v1/chat-completions/HCX-003",
                headers=headers,
                json=data
            )
            result = response.json()

            if "result" in result and result["result"] is not None and "message" in result["result"]:
                return result["result"]["message"]["content"]
            elif "status" in result and result["status"]["code"] == "42901":
                time.sleep(self._retry_delay)
            else:
                raise ValueError(f"Unexpected API response: {result}")

        raise ValueError(f"Failed to get a valid response after {self._max_retries} attempts.")

# 질문 리스트
questions = [
    "큰 꿈을 안고 시작한 투자, 여러분은 투자를 할 때 가장 중요한 요소가 무엇이라고 생각하나요? 안전하게 꾸준한 수익을 내는 투자가 옳은 투자일까요? 혹은 위험하더라도 큰 수익을 내야 진정한 투자일까요? 생각을 자유롭게 작성해 주세요.",
    "우-러 전쟁이 발발했을 때 증권 시장이 크게 요동쳤어요. 하지만 시간이 지나고 전쟁으로 요동쳤던 증권 시장이 안정화가 되었어요. 증권 시장은 알 수 없는 이유로도 단기간에 변동되기는 현상을 보여주기도 합니다. 여려분이라면 단기적으로 크게 변동되는 상황을 어떻게 대응하나요?",
    "투자는 제각각 다른 목표를 갖고 시작하곤 합니다. 누군가는 소소한 용돈을 벌기 위해, 누군가는 자가 구입을 위해 투자를 하고 있어요. 여러분의 투자 목표는 무엇인가요? 얼마나 수익을 내고, 언제까지 투자를 하고 싶나요?",
    "투자할 종목을 선택하려고 합니다. 어떤 정보를 참고하는 것이 좋을까요? 다양한 투자자들의 의견을 들어보기 위해 네이버 증권의 커뮤니티를 확인해 볼 수도 있고, 신뢰성있는 정보를 위해 뉴스나 전문가의 칼럼을 참고할 수도 있어요. 그래도 부족하다면 재무제표까지도 열어볼 수 있습니다. 여러분들은 어떤 정보를 어떻게 활용할 예정인가요?",
    "high risk-high return 이라는 말을 들어보셨나요? 어떻게 생각하시나요? 위험이 커도 높은 수익률을 추구하시나요? 아니면 수익이 낮더라도 안정적인 수익률이 나을까요?",
    "출근길 뉴스를 보니 내가 보유하고 있는 종목과 관련된 안 좋은 소식이 보도되고 있어요. 그런데 간접적이기도 하고, 반응도 꼭 나쁘지만은 않은 것 같아 보인다면, 어떻게 하실건가요? 주식을 매도하는게 나을까요? 혹은 기다리시나요? 선택과 이유를 함께 적어주세요.",
    "긴 고민 끝에 잘 만들어둔 나의 포트폴리오. 매일매일 변하는 평가손익이 자꾸 눈에 거슬리기도 합니다. 조언을 구해보면 일희일비 하지 말고, 앱을 지우는 것도 좋은 방법이라고 소개해 주었어요. 하지만 앱을 지운다면 포트폴리오를 자주, 그리고 즉각적으로 수정하기는 어려울 것 같아 고민입니다. 여러분은 앱을 지우고 목표 기간 뒤에 열어보는게 좋다고 생각하시나요? 혹은 매일매일 확인하고 포트폴리오를 즉각적으로 대응하여 수정하는 것이 좋다고 생각하시나요?",
    "우리는 지금까지 다양한 상황에서 판단을 해왔어요. 지금, 답변을 적는 이 순간 여러분의 투자 지식은 어느정도 된다고 생각하시나요? 자유롭게 기술해 주세요.",
    "나의 투자 실력을 곰곰이 생각하다보니 그 찰나에 나의 주식이 크게 떨어졌어요. 지금 이 순간을 어떻게 대처하실건가요?",
    "재빠른 대처 능력으로 위기를 잘 극복해 내었습니다. 이젠 스스로도 투자를 어느정도 잘 하고 다음 과정으로 나아가도 되겠다는 생각이 들기도 합니다. 지금 이 생각을 한 순간 눈 앞에 새로운 투자 기회나 금융 상품이 있다면 하면 어떤 기분과 생각이 드시나요?"
]
# Chat Completion 실행기 초기화
completion_executor = CompletionExecutor(
    host="https://clovastudio.apigw.ntruss.com",
    api_key='NTA0MjU2MWZlZTcxNDJiY6yXz/WB7l+4+xfKx19irvCn79XT+YIlOBzoWEOuWvys', #
    api_key_primary_val='ldCsKr7kX06FeipjEc8gg84t4A8Ad8ILUIVKN0BR', # api 연결해야함 
    request_id='d764eebd-e3a7-41d6-9db2-1b90c12e8754'
)
total_scores = {
        "risk_tolerance": 0, "investment_time_horizon": 0, "financial_goal_orientation": 0, "information_processing_style": 0, "investment_fear": 0,
        "investment_confidence": 0
    }
for question in questions:
    prompt_template = load_prompt_template()
    print(question)
    answer = input()
    formatted_prompt = prompt_template.replace("[question]", question).replace("[answer]", answer)

    messages = [
            {"role": "system", "content": "너는 사용자의 요청을 엄격하게 json형태에 맞게 응답하는 AI야."},
            {"role": "user", "content": formatted_prompt}
        ]
    scores = json.loads(completion_executor.execute(messages))
    if scores:
        for key in total_scores:
            total_scores[key] += scores[key]
prompt_template = load_prompt2_template()
formatted_prompt = prompt_template.replace("[score1]", str(total_scores["risk_tolerance"])).replace("[score2]", str(total_scores["investment_time_horizon"])).replace("[score3]", str(total_scores["financial_goal_orientation"])).replace("[score4]", str(total_scores["information_processing_style"])).replace("[score5]", str(total_scores["investment_fear"])).replace("[score6]", str(total_scores["investment_confidence"]))
messages = [
            {"role": "system", "content": "너는 사용자의 요청을 엄격하게 json형태에 맞게 응답하는 AI야."},
            {"role": "user", "content": formatted_prompt}
        ]
analysis = json.loads(completion_executor.execute(messages))
print(analysis)