#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Service implementation with multi-provider support.
- OpenAI GPT API integration
- CLOVA API integration  
- Comprehensive error handling and fallback mechanisms
- Response validation and post-processing
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import openai
except ImportError:
    openai = None

try:
    import requests
except ImportError:
    requests = None

try:
    from config import config
except ImportError:
    from src.config import config
    
try:
    from prompt_manager import PromptManager
except ImportError:
    from src.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service with multi-provider support and error handling."""
    
    def __init__(self, api_type: str = "openai"):
        self.api_type = api_type.lower()
        self.max_retries = config.get('MAX_RETRIES', 3)
        self.retry_delay = config.get('RETRY_DELAY', 1)
        
        # REQUIRED: Initialize API clients
        self.openai_api_key = config.get('OPENAI_API_KEY', '')
        self.clova_api_key = config.get('CLOVA_API_KEY', '')
        
        # REQUIRED: Initialize OpenAI client
        if self.openai_api_key and openai:
            openai.api_key = self.openai_api_key
            self.openai_client = openai
        else:
            self.openai_client = None
            
        # REQUIRED: Initialize prompt manager
        self.prompt_manager = PromptManager()
        
        logger.info(f"LLM Service initialized with API type: {self.api_type}")
    
    def generate_ai_response(
        self, 
        prompt_name: str, 
        user_message: str, 
        context: str = "", 
        analysis_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate AI response with comprehensive error handling.
        
        Args:
            prompt_name: Name of the prompt template (AI-A, AI-A2, AI-B, etc.)
            user_message: User's input message
            context: Additional context (news, alerts, portfolio)
            analysis_result: User profile analysis result
            
        Returns:
            Dictionary with success status and response
        """
        try:
            # REQUIRED: Get and personalize prompt
            prompt = self._get_personalized_prompt(
                prompt_name, user_message, context, analysis_result
            )
            
            # REQUIRED: Generate response with fallback
            return self._generate_ai_response_with_fallback(prompt_name, prompt)
            
        except Exception as e:
            logger.error(f"Failed to generate AI response: {e}")
            return self._generate_template_response(prompt_name, user_message)
    
    def _get_personalized_prompt(
        self, 
        prompt_name: str, 
        user_message: str, 
        context: str, 
        analysis_result: Optional[Dict[str, Any]]
    ) -> str:
        """
        Get personalized prompt with context injection.
        
        Args:
            prompt_name: Prompt template name
            user_message: User's message
            context: Market context
            analysis_result: User profile data
            
        Returns:
            Personalized prompt string
        """
        try:
            # REQUIRED: Load base prompt
            base_prompt = self.prompt_manager.get_prompt(prompt_name)
            
            # REQUIRED: Personalize based on user profile
            if analysis_result:
                logger.info(f"Personalizing prompt with analysis_result keys: {list(analysis_result.keys())}")
                # Log the type of analysis_result
                logger.info(f"analysis_result type: {type(analysis_result)}")
                
                # If analysis_result has profile_json, log its contents
                if 'profile_json' in analysis_result:
                    logger.info(f"profile_json exists, type: {type(analysis_result['profile_json'])}")
                    if isinstance(analysis_result['profile_json'], dict):
                        logger.info(f"profile_json keys: {list(analysis_result['profile_json'].keys())}")
                
                base_prompt = self.prompt_manager.personalize_prompt(
                    base_prompt, analysis_result
                )
            
            # REQUIRED: Inject context and user message
            personalized_prompt = base_prompt.replace("[USER_MESSAGE]", user_message)
            if context:
                personalized_prompt = personalized_prompt.replace("[CONTEXT]", context)
            
            # Debug: Log first 500 chars of personalized prompt
            logger.info(f"Final prompt for {prompt_name} (first 500 chars): {personalized_prompt[:500]}...")
            
            # Debug: Check if profile was actually replaced
            if "[risk_tolerance_analysis]" in personalized_prompt:
                logger.warning(f"Profile placeholders not replaced in {prompt_name} prompt!")
            else:
                logger.info(f"Profile successfully personalized in {prompt_name} prompt")
            
            return personalized_prompt
            
        except Exception as e:
            logger.error(f"Failed to personalize prompt: {e}")
            return f"사용자 질문: {user_message}\n\n한국어로 투자 관련 조언을 제공해주세요."
    
    def _generate_ai_response_with_fallback(self, agent: str, prompt: str) -> Dict[str, Any]:
        """
        AI response with full error recovery chain.
        
        Args:
            agent: AI agent name
            prompt: Prompt to send to AI
            
        Returns:
            Response dictionary with success status
        """
        # REQUIRED: Primary API attempt
        try:
            if self.api_type == "openai" and self.openai_client:
                response = self._call_openai_api(prompt)
                if response.get("success"):
                    return response
            elif self.api_type == "clova" and self.clova_api_key:
                response = self._call_clova_api(prompt)
                if response.get("success"):
                    return response
        except Exception as e:
            logger.error(f"Primary API failed for {agent}: {e}")
        
        # REQUIRED: Secondary API fallback
        try:
            fallback_api = "clova" if self.api_type == "openai" else "openai"
            logger.info(f"Trying fallback API: {fallback_api}")
            
            if fallback_api == "openai" and self.openai_client:
                response = self._call_openai_api(prompt)
                if response.get("success"):
                    return response
            elif fallback_api == "clova" and self.clova_api_key:
                response = self._call_clova_api(prompt)
                if response.get("success"):
                    return response
        except Exception as e:
            logger.error(f"Fallback API failed for {agent}: {e}")
        
        # REQUIRED: Final fallback - template response
        return self._generate_template_response(agent, prompt)
    
    def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call OpenAI API with retry logic.
        
        Args:
            prompt: Prompt to send to OpenAI
            
        Returns:
            Response dictionary
        """
        if not self.openai_client:
            return {"success": False, "error": "OpenAI client not available"}
        
        for attempt in range(self.max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.7,
                    timeout=30
                )
                
                response_text = response.choices[0].message.content
                
                # REQUIRED: Validate response
                if self._validate_ai_response(response_text, "openai"):
                    response_text = self._post_process_response(response_text, "openai")
                    return {"success": True, "response": response_text}
                else:
                    logger.warning("OpenAI response failed validation")
                    return {"success": False, "error": "Response validation failed"}
                    
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                logger.error(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def _call_clova_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call CLOVA API with retry logic.
        
        Args:
            prompt: Prompt to send to CLOVA
            
        Returns:
            Response dictionary
        """
        if not self.clova_api_key or not requests:
            return {"success": False, "error": "CLOVA API not available"}
        
        url = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-DASH-001"
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self.clova_api_key,
            "X-NCP-APIGW-API-KEY": self.clova_api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 2000,
            "temperature": 0.7,
            "repeatPenalty": 1.2,
            "stopBefore": [],
            "includeAiFilters": True
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=data, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result["result"]["message"]["content"]
                    
                    # REQUIRED: Validate response
                    if self._validate_ai_response(response_text, "clova"):
                        response_text = self._post_process_response(response_text, "clova")
                        return {"success": True, "response": response_text}
                    else:
                        logger.warning("CLOVA response failed validation")
                        return {"success": False, "error": "Response validation failed"}
                else:
                    logger.error(f"CLOVA API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"CLOVA API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def _validate_ai_response(self, response: str, provider: str) -> bool:
        """
        Validate AI response meets quality standards.
        
        Args:
            response: AI response text
            provider: API provider name
            
        Returns:
            True if response is valid
        """
        try:
            # REQUIRED: Basic validation
            if not response or len(response.strip()) < 10:
                logger.warning(f"{provider} response too short: {len(response)} chars")
                return False
            
            # REQUIRED: Korean language validation
            korean_chars = sum(1 for char in response if ord(char) >= 0xAC00 and ord(char) <= 0xD7AF)
            if len(response) > 0 and korean_chars / len(response) < 0.1:
                logger.warning(f"{provider} response not primarily Korean")
                return False
            
            # OPTIONAL: Content validation - log warning but don't fail
            investment_terms = ["투자", "포트폴리오", "주식", "펀드", "위험", "수익", "자산", "분산", "성향", "추천", "분석"]
            if not any(term in response for term in investment_terms):
                logger.info(f"{provider} response may not contain investment terms, but continuing")
                # Don't fail validation for missing investment terms
                # This allows AI to ask questions or provide other guidance
            
            return True
            
        except Exception as e:
            logger.error(f"Response validation error: {e}")
            return False
    
    def _post_process_response(self, response: str, provider: str) -> str:
        """
        Clean and format AI response.
        
        Args:
            response: Raw AI response
            provider: API provider name
            
        Returns:
            Cleaned response text
        """
        try:
            # REQUIRED: Remove common AI disclaimers
            disclaimers_to_remove = [
                "I am an AI assistant",
                "This is not financial advice",
                "Please consult a financial advisor",
                "저는 AI 어시스턴트입니다",
                "이는 투자 조언이 아닙니다"
            ]
            
            for disclaimer in disclaimers_to_remove:
                response = response.replace(disclaimer, "")
            
            # REQUIRED: Clean formatting
            response = response.strip()
            
            # REQUIRED: Ensure proper sentence ending
            if response and not response.endswith(('.', '!', '?')):
                response += '.'
            
            return response
            
        except Exception as e:
            logger.error(f"Response post-processing error: {e}")
            return response
    
    def _generate_template_response(self, agent: str, prompt: str) -> Dict[str, Any]:
        """
        Generate template response when all APIs fail.
        
        Args:
            agent: AI agent name
            prompt: Original prompt
            
        Returns:
            Template response dictionary
        """
        templates = {
            "AI-A": "죄송합니다. 현재 AI 서비스에 일시적인 문제가 있어 맞춤형 투자 조언을 제공할 수 없습니다. 잠시 후 다시 시도해주세요.",
            "AI-A2": "AI-A의 응답을 바탕으로 추가적인 금융 데이터 분석이 필요합니다.",
            "AI-B": "현재 금융 데이터 분석 서비스에 접근할 수 없습니다. 기본적인 투자 원칙을 참고해주세요.",
            "Final": "투자 상담 중 기술적 문제가 발생했습니다. 기본적인 분산투자 원칙을 권장드립니다."
        }
        
        return {
            "success": True,
            "response": templates.get(agent, "일시적인 서비스 오류가 발생했습니다."),
            "fallback": True
        }
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available AI models.
        
        Returns:
            List of available model names
        """
        available_models = []
        
        if self.openai_client and self.openai_api_key:
            available_models.append("openai")
        
        if self.clova_api_key:
            available_models.append("clova")
        
        if not available_models:
            available_models.append("simulation")
        
        return available_models
    
    def test_api_connection(self, api_type: Optional[str] = None) -> Dict[str, bool]:
        """
        Test API connections.
        
        Args:
            api_type: Specific API to test, or None for all
            
        Returns:
            Dictionary of API test results
        """
        results = {}
        
        if api_type is None or api_type == "openai":
            if self.openai_client and self.openai_api_key:
                try:
                    test_response = self._call_openai_api("안녕하세요. 간단한 테스트 메시지입니다.")
                    results["openai"] = test_response.get("success", False)
                except Exception as e:
                    logger.error(f"OpenAI connection test failed: {e}")
                    results["openai"] = False
            else:
                results["openai"] = False
        
        if api_type is None or api_type == "clova":
            if self.clova_api_key:
                try:
                    test_response = self._call_clova_api("안녕하세요. 간단한 테스트 메시지입니다.")
                    results["clova"] = test_response.get("success", False)
                except Exception as e:
                    logger.error(f"CLOVA connection test failed: {e}")
                    results["clova"] = False
            else:
                results["clova"] = False
        
        return results 