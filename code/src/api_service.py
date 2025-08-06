#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API 서비스 모듈
- 외부 API 연동 (OpenAI, CLOVA 등)
- API 요청 및 응답 처리
- 에러 핸들링 및 재시도 로직
"""

import os
import json
import time
import requests
import logging
from typing import List, Dict, Any, Tuple, Optional
import openai
from flask import Flask, jsonify, request
from db_client import get_supabase_client
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("APIService")

class APIService:
    """API 서비스 클래스"""
    def __init__(self, api_type="openai"):
        self.api_type = api_type
        
        # API 키 설정
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.clova_api_key = os.environ.get("CLOVA_API_KEY", "")
        self.clova_api_host = os.environ.get("CLOVA_API_HOST", "https://api.clovastudio.com")
        
        # OpenAI 클라이언트 설정
        if self.api_type == "openai" and self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.client = openai.OpenAI(api_key=self.openai_api_key)
        
        # 재시도 설정
        self.max_retries = 3
        self.retry_delay = 1
        
        logger.info(f"API 서비스 초기화 완료 (유형: {api_type})")
    
    def call_openai_api(self, system_prompt, user_message, model="gpt-4o-mini"):
        """OpenAI API 호출"""
        if not self.openai_api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            return {"error": "OpenAI API 키가 설정되지 않았습니다."}
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"OpenAI API 호출 시도 ({attempt+1}/{self.max_retries})")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.2,
                )
                logger.debug("OpenAI API 호출 성공")
                return {"response": response.choices[0].message.content.strip()}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"OpenAI API 호출 실패 ({attempt+1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"OpenAI API 호출 실패: {e}")
                    return {"error": f"API 호출 실패: {str(e)}"}
    
    def call_clova_api(self, system_prompt, user_message):
        """CLOVA Studio API 호출"""
        if not self.clova_api_key:
            logger.error("CLOVA API 키가 설정되지 않았습니다.")
            return {"error": "CLOVA API 키가 설정되지 않았습니다."}
            
        api_url = f"{self.clova_api_host}/api/v1/completions/LK-D2"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.clova_api_key
        }
        
        payload = {
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 1024,
            "temperature": 0.3,
            "repeatPenalty": 5.0,
            "stopBefore": [],
            "includeAiFilters": True,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"CLOVA API 호출 시도 ({attempt+1}/{self.max_retries})")
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                logger.debug("CLOVA API 호출 성공")
                return {"response": result["result"]["message"]["content"]}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"CLOVA API 호출 실패 ({attempt+1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"CLOVA API 호출 실패: {e}")
                    return {"error": f"API 호출 실패: {str(e)}"}
    
    def call_ai_api(self, system_prompt, user_message):
        """API 유형에 따라 적절한 API 호출"""
        logger.info(f"AI API 호출 (유형: {self.api_type})")
        
        if self.api_type == "openai":
            return self.call_openai_api(system_prompt, user_message)
        elif self.api_type == "clova":
            return self.call_clova_api(system_prompt, user_message)
        elif self.api_type == "simulation":
            # 시뮬레이션 모드 (API 키가 없을 때)
            logger.warning("시뮬레이션 모드로 실행 중입니다. 실제 API 호출이 이루어지지 않습니다.")
            return {"response": self._generate_simulation_response(user_message)}
        else:
            logger.error(f"지원하지 않는 API 유형: {self.api_type}")
            return {"error": f"지원하지 않는 API 유형: {self.api_type}"}
    
    def _generate_simulation_response(self, user_message):
        """시뮬레이션 모드에서 응답 생성"""
        if "안녕" in user_message or "반가" in user_message:
            return "안녕하세요! MINERVA 투자 상담 시스템입니다. 투자에 관한 질문이 있으신가요?"
        
        elif "코스피" in user_message or "주가" in user_message:
            return "현재 코스피 지수는 변동성이 있지만 전반적으로 안정세를 유지하고 있습니다. 특정 종목에 관심이 있으신가요?"
        
        elif "추천" in user_message or "포트폴리오" in user_message:
            return "귀하의 투자 성향을 고려할 때, 안정성과 성장성을 모두 갖춘 포트폴리오가 적합합니다. 인덱스 ETF 40%, 우량 배당주 30%, 성장주 20%, 현금성 자산 10% 정도의 배분을 추천드립니다."
        
        elif "etf" in user_message.lower() or "인덱스" in user_message:
            return "ETF는 분산 투자에 좋은 수단입니다. 국내 ETF 중에서는 KODEX 200, TIGER 200 등의 대형주 ETF와 ARIRANG 중형주, KBSTAR 배당 등 다양한 옵션이 있습니다. 특정 섹터에 관심이 있으신가요?"
        
        elif "배당" in user_message or "dividend" in user_message.lower():
            return "배당주 투자는 안정적인 현금흐름을 원하는 투자자에게 적합합니다. 국내 주요 배당주로는 삼성전자, KT&G, 한국전력, 현대차 등이 있습니다. 배당수익률과 배당성장률을 함께 고려하는 것이 중요합니다."
        
        else:
            return "투자에 관한 질문이 있으시면 언제든지 물어보세요. 주식, ETF, 펀드, 포트폴리오 구성 등 다양한 주제에 대해 답변해 드릴 수 있습니다."
    
    def change_api_type(self, api_type):
        """API 유형 변경"""
        if api_type in ["openai", "clova", "simulation"]:
            self.api_type = api_type
            
            # OpenAI 클라이언트 재설정
            if self.api_type == "openai" and self.openai_api_key:
                openai.api_key = self.openai_api_key
                self.client = openai.OpenAI(api_key=self.openai_api_key)
            
            logger.info(f"API 유형이 변경되었습니다: {api_type}")
            return True
        else:
            logger.error(f"지원하지 않는 API 유형: {api_type}")
            return False
    
    def set_api_key(self, api_type, api_key):
        """API 키 설정"""
        if api_type == "openai":
            self.openai_api_key = api_key
            if api_key:
                openai.api_key = api_key
                self.client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI API 키가 설정되었습니다.")
            return True
        elif api_type == "clova":
            self.clova_api_key = api_key
            logger.info("CLOVA API 키가 설정되었습니다.")
            return True
        else:
            logger.error(f"지원하지 않는 API 유형: {api_type}")
            return False
    
    def test_api_connection(self):
        """API 연결 테스트"""
        logger.info(f"API 연결 테스트 시작 (유형: {self.api_type})")
        
        test_prompt = "간단한 테스트 메시지입니다."
        test_message = "안녕하세요!"
        
        result = self.call_ai_api(test_prompt, test_message)
        
        if "error" in result:
            logger.error(f"API 연결 테스트 실패: {result['error']}")
            return False, result['error']
        else:
            logger.info("API 연결 테스트 성공")
            return True, "API 연결 테스트 성공"

# 모듈 테스트용 코드
if __name__ == "__main__":
    # 환경 변수에서 API 키 로드
    from dotenv import load_dotenv
    load_dotenv()
    
    # API 서비스 초기화
    api_service = APIService()
    
    # API 연결 테스트
    success, message = api_service.test_api_connection()
    print(f"API 연결 테스트 결과: {'성공' if success else '실패'} - {message}")
    
    # 테스트 호출
    if success:
        result = api_service.call_ai_api(
            "당신은 투자 조언을 제공하는 AI 금융 상담사입니다.",
            "최근 코스피 지수 동향에 대해 알려주세요."
        )
        
        if "error" in result:
            print(f"API 호출 실패: {result['error']}")
        else:
            print(f"API 응답: {result['response']}") 

app = Flask(__name__)
supabase = get_supabase_client()

def extract_ticker_from_question(question):
    # 1. 종목코드(숫자 6자리) 추출
    m = re.search(r'\b\d{6}\b', question)
    if m:
        return m.group()
    # 2. 종목명(한글) 추출 및 kor_ticker에서 코드 매핑
    tickers = supabase.table("kor_ticker").select("종목코드,종목명").execute().data
    for t in tickers:
        if t["종목명"] in question:
            return t["종목코드"]
    return None

@app.route('/api/user_profile/<user_id>')
def get_user_profile(user_id):
    res = supabase.table("user_profiles").select("profile_json").eq("user_id", user_id).execute()
    return jsonify(res.data[0]["profile_json"] if res.data else {})

@app.route('/api/portfolio/<user_id>')
def get_portfolio(user_id):
    res = supabase.table("portfolio_recommendations").select("portfolio_json").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
    return jsonify(res.data[0]["portfolio_json"] if res.data else {})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id = data.get("user_id", "default")
    question = data.get("question", "")
    # 사용자 성향/포트폴리오 조회
    profile = supabase.table("user_profiles").select("profile_json").eq("user_id", user_id).execute().data
    portfolio = supabase.table("portfolio_recommendations").select("portfolio_json").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute().data
    # 질문에서 종목코드/이름 추출
    ticker = extract_ticker_from_question(question) or "005930"  # 기본값 삼성전자
    # 해당 종목 실시간 데이터 조회
    price = supabase.table("kor_price").select("*").eq("종목코드", ticker).order("날짜", desc=True).limit(1).execute().data
    ticker_info = supabase.table("kor_ticker").select("*").eq("종목코드", ticker).limit(1).execute().data
    news = supabase.table("news").select("*").eq("ticker", ticker).order("created_at", desc=True).limit(3).execute().data
    # 프롬프트에 실시간 데이터 포함
    price_str = f"{price[0]['종가']}원 (날짜: {price[0]['날짜']})" if price else "데이터 없음"
    ticker_name = ticker_info[0]['종목명'] if ticker_info else ticker
    news_str = '\n'.join([f"- {n.get('date', n.get('created_at', ''))}: {n.get('content', '')}" for n in news]) if news else "뉴스 없음"
    prompt = f"""
    사용자 투자 성향: {profile[0]['profile_json'] if profile else '없음'}
    추천 포트폴리오: {portfolio[0]['portfolio_json'] if portfolio else '없음'}
    {ticker_name}({ticker}) 최신 가격: {price_str}
    최근 뉴스:
    {news_str}
    사용자 질문: {question}
    위의 실시간 데이터를 바탕으로 맞춤형 답변을 해줘.
    """
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"너는 투자 분석 챗봇이야."},
            {"role":"user","content":prompt}
        ]
    )
    answer = response.choices[0].message.content
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True) 