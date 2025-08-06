#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prompt management system for AI agents.
- Template loading and caching
- Dynamic prompt personalization based on user profiles
- Context-aware prompt adaptation
- Validation and formatting
"""

import os
import logging
from typing import Dict, Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompt templates and personalization."""
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompt_cache: Dict[str, str] = {}
        
        # REQUIRED: Prompt file paths
        self.prompt_files = {
            "AI-A": os.path.join(self.script_dir, "prompt_AI-A.txt"),
            "AI-A2": os.path.join(self.script_dir, "prompt_AI-A2.txt"),
            "AI-B": os.path.join(self.script_dir, "prompt_AI-B.txt"),
            "survey-analysis": os.path.join(self.script_dir, "prompt_survey-analysis.txt"),
            "survey-score": os.path.join(self.script_dir, "prompt_survey-score.txt"),
        }
        
        logger.info("PromptManager initialized successfully")
    
    def get_prompt(self, prompt_name: str) -> str:
        """
        Get prompt template with caching.
        
        Args:
            prompt_name: Name of the prompt template
            
        Returns:
            Prompt template string
        """
        # REQUIRED: Check cache first
        if prompt_name in self.prompt_cache:
            return self.prompt_cache[prompt_name]
        
        # REQUIRED: Load from file
        if prompt_name in self.prompt_files:
            try:
                prompt_path = self.prompt_files[prompt_name]
                if os.path.exists(prompt_path):
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        prompt = f.read()
                    
                    # REQUIRED: Cache the prompt
                    self.prompt_cache[prompt_name] = prompt
                    return prompt
                else:
                    logger.warning(f"Prompt file not found: {prompt_path}")
            except Exception as e:
                logger.error(f"Failed to load prompt {prompt_name}: {e}")
        
        # REQUIRED: Return default prompt if file not found
        return self._get_default_prompt(prompt_name)
    
    def personalize_prompt(self, base_prompt: str, user_profile: Dict[str, Any]) -> str:
        """
        Personalize prompt based on user investment profile.
        
        Args:
            base_prompt: Base prompt template
            user_profile: User profile data with scores and analysis
            
        Returns:
            Personalized prompt string
        """
        try:
            personalized_prompt = base_prompt
            
            # REQUIRED: Extract profile data - handle both direct and nested structures
            if 'profile_json' in user_profile:
                # If profile_json exists, use it
                actual_profile = user_profile['profile_json']
                logger.info("Using nested profile_json structure")
            else:
                actual_profile = user_profile
            
            scores = actual_profile.get('scores', {})
            detailed_analysis = actual_profile.get('detailed_analysis', {})
            
            # Log the actual data we're working with
            logger.info(f"Profile structure: {list(actual_profile.keys())}")
            logger.info(f"Scores found: {list(scores.keys())}")
            logger.info(f"Detailed analysis found: {list(detailed_analysis.keys())}")
            
            # REQUIRED: Replace analysis placeholders
            logger.info(f"Personalizing prompt with profile keys: {list(detailed_analysis.keys())}")
            replacements_made = 0
            
            # Check which placeholders exist in the prompt
            placeholders_in_prompt = []
            for key in detailed_analysis.keys():
                placeholder = f"[{key}]"
                if placeholder in personalized_prompt:
                    placeholders_in_prompt.append(placeholder)
            logger.info(f"Placeholders found in prompt: {placeholders_in_prompt}")
            
            for key, value in detailed_analysis.items():
                placeholder = f"[{key}]"
                if placeholder in personalized_prompt:
                    # Ensure value is not empty
                    if value:
                        personalized_prompt = personalized_prompt.replace(placeholder, str(value))
                        replacements_made += 1
                        logger.info(f"Replaced {placeholder} with: {str(value)[:100]}...")
                    else:
                        logger.warning(f"Empty value for {placeholder}")
            
            logger.info(f"Made {replacements_made} replacements in prompt")
            
            # Check if any placeholders remain
            remaining_placeholders = []
            for key in ['risk_tolerance_analysis', 'investment_time_horizon_analysis', 
                       'financial_goal_orientation_analysis', 'information_processing_style_analysis',
                       'investment_fear_analysis', 'investment_confidence_analysis']:
                if f"[{key}]" in personalized_prompt:
                    remaining_placeholders.append(f"[{key}]")
            
            if remaining_placeholders:
                logger.error(f"Placeholders still remaining after personalization: {remaining_placeholders}")
                # 플레이스홀더가 남아있으면 기본값으로 대체
                default_values = {
                    'risk_tolerance_analysis': '중간 수준의 위험 감수 성향을 보입니다.',
                    'investment_time_horizon_analysis': '중장기 투자를 선호합니다.',
                    'financial_goal_orientation_analysis': '균형 잡힌 수익 지향성을 보입니다.',
                    'information_processing_style_analysis': '균형 잡힌 정보 처리 스타일을 보입니다.',
                    'investment_fear_analysis': '투자에 대한 두려움이 중간 수준입니다.',
                    'investment_confidence_analysis': '중간 정도의 투자 자신감을 보입니다.'
                }
                
                for key, default_value in default_values.items():
                    placeholder = f"[{key}]"
                    if placeholder in personalized_prompt:
                        personalized_prompt = personalized_prompt.replace(placeholder, default_value)
                        logger.warning(f"Replaced {placeholder} with default value")
            
            # REQUIRED: Add risk tolerance guidance
            risk_level = scores.get('risk_tolerance', 50)
            risk_guidance = self._get_risk_guidance(risk_level)
            personalized_prompt = personalized_prompt.replace("[RISK_GUIDANCE]", risk_guidance)
            
            # REQUIRED: Add detail level based on information processing style
            info_style = scores.get('information_processing_style', 50)
            detail_level = self._get_detail_level(info_style)
            personalized_prompt = personalized_prompt.replace("[DETAIL_LEVEL]", detail_level)
            
            # REQUIRED: Add investment timeframe guidance
            timeframe = scores.get('investment_time_horizon', 50)
            timeframe_guidance = self._get_timeframe_guidance(timeframe)
            personalized_prompt = personalized_prompt.replace("[TIMEFRAME_GUIDANCE]", timeframe_guidance)
            
            return personalized_prompt
            
        except Exception as e:
            logger.error(f"Failed to personalize prompt: {e}")
            return base_prompt
    
    def _get_risk_guidance(self, risk_level: float) -> str:
        """
        Get risk guidance based on user's risk tolerance score.
        
        Args:
            risk_level: Risk tolerance score (0-100)
            
        Returns:
            Risk guidance text
        """
        if risk_level < 30:
            return "안전한 투자를 선호하며 원금 보존에 중점을 둡니다. 저위험 상품을 우선 고려하세요."
        elif risk_level > 70:
            return "높은 수익을 위해 상당한 위험을 감수할 의향이 있습니다. 고위험-고수익 투자 기회를 검토하세요."
        else:
            return "적절한 위험-수익 균형을 추구합니다. 중위험 투자 전략을 고려하세요."
    
    def _get_detail_level(self, info_style: float) -> str:
        """
        Get detail level based on user's information processing style.
        
        Args:
            info_style: Information processing style score (0-100)
            
        Returns:
            Detail level guidance text
        """
        if info_style > 60:
            return "구체적인 데이터와 분석 지표를 포함하여 상세한 설명을 제공하세요."
        else:
            return "복잡한 전문 용어보다는 이해하기 쉬운 설명과 실용적인 예시를 사용하세요."
    
    def _get_timeframe_guidance(self, timeframe: float) -> str:
        """
        Get investment timeframe guidance.
        
        Args:
            timeframe: Investment time horizon score (0-100)
            
        Returns:
            Timeframe guidance text
        """
        if timeframe < 30:
            return "단기적 성과에 집중하며 유동성이 높은 투자를 선호합니다."
        elif timeframe > 70:
            return "장기적 관점에서 투자하며 복리 효과를 중시합니다."
        else:
            return "중기적 투자 목표를 가지고 있으며 적절한 성장과 안정성을 추구합니다."
    
    def _get_default_prompt(self, prompt_name: str) -> str:
        """
        Get default prompt when file loading fails.
        
        Args:
            prompt_name: Name of the prompt
            
        Returns:
            Default prompt template
        """
        default_prompts = {
            "AI-A": """당신은 개인화된 투자 조언을 제공하는 AI 금융 상담사입니다.

사용자 투자 성향:
- 위험 감수성: [risk_tolerance_analysis]
- 투자 시간 범위: [investment_time_horizon_analysis]
- 재무 목표 지향성: [financial_goal_orientation_analysis]
- 정보 처리 스타일: [information_processing_style_analysis]
- 투자 자신감: [investment_confidence_analysis]

[RISK_GUIDANCE]
[DETAIL_LEVEL]
[TIMEFRAME_GUIDANCE]

사용자의 성향을 고려하여 맞춤형 투자 조언을 제공하세요.
한국어로 응답하며, 구체적이고 실용적인 조언을 제공하세요.

사용자 질문: [USER_MESSAGE]

추가 컨텍스트: [CONTEXT]""",

            "AI-A2": """당신은 AI-A2로, 사용자의 투자 성향을 이해한 후 금융 데이터 AI(AI-B)에게 필요한 금융 정보를 요청하는 역할을 합니다.

사용자 투자 성향:
- 위험 감수성: [risk_tolerance_analysis]
- 투자 시간 범위: [investment_time_horizon_analysis]
- 재무 목표 지향성: [financial_goal_orientation_analysis]
- 정보 처리 스타일: [information_processing_style_analysis]
- 투자 자신감: [investment_confidence_analysis]

AI-A의 초기 조언을 바탕으로 AI-B에게 구체적인 데이터 분석을 요청하세요.
요청 시 다음 사항을 포함하세요:
1. 필요한 데이터 유형
2. 분석 범위
3. 사용자 컨텍스트

한국어로 응답하세요.

사용자 질문: [USER_MESSAGE]
AI-A 응답: [AI_A_RESPONSE]
추가 컨텍스트: [CONTEXT]""",

            "AI-B": """당신은 금융 데이터를 분석하고 근거 있는 정보를 제공하는 AI-B입니다.

AI-A2의 요청에 따라 다음 사항을 제공하세요:
1. 정확하고 최신의 금융 데이터
2. 통계적 분석과 근거
3. 위험-수익 분석
4. 객관적인 시장 평가

데이터 기반의 분석을 제공하되, 사용자의 투자 성향을 고려하여 적절한 수준의 상세함으로 설명하세요.
한국어로 응답하세요.

AI-A2 요청: [AI_A2_QUERY]
시장 데이터 컨텍스트: [CONTEXT]""",

            "survey-analysis": """당신은 투자자의 성향을 분석하고 평가하는 전문가입니다.

주어진 투자 관련 지표의 점수를 바탕으로 투자자의 성향과 특징을 상세히 분석하고 설명해주세요.

분석해야 할 지표:
1. Risk Tolerance (위험 감수성)
2. Investment Time Horizon (투자 시간 범위)
3. Financial Goal Orientation (재무 목표 지향성)
4. Information Processing Style (정보 처리 스타일)
5. Investment Fear (투자 두려움)
6. Investment Confidence (투자 자신감)

각 지표별 세부 분석을 제공하고, 지표 간 상호작용을 분석하며, 종합적 평가를 통해 투자자에게 적합한 투자 전략이나 상품 유형을 제안해주세요.

분석 결과는 다음 JSON 형식으로 출력해주세요:
{
  "detailed_analysis": {
    "risk_tolerance_analysis": "위험 감수성 분석",
    "investment_time_horizon_analysis": "투자 시간 범위 분석",
    "financial_goal_orientation_analysis": "재무 목표 지향성 분석",
    "information_processing_style_analysis": "정보 처리 스타일 분석",
    "investment_fear_analysis": "투자 두려움 분석",
    "investment_confidence_analysis": "투자 자신감 분석"
  },
  "overall_analysis": "종합 분석",
  "investment_strategy": "적합한 투자 전략",
  "recommended_products": ["추천 상품 리스트"]
}""",

            "survey-score": """당신은 투자자들의 성향을 평가하는 전문가입니다.

주어진 질문과 답변을 분석하여 각 투자 관련 지표에 대한 점수를 0-100 척도로 평가해주세요.

평가할 지표:
1. Risk Tolerance (위험 감수성): 투자 위험을 감수할 의향
2. Investment Time Horizon (투자 시간 범위): 장기 vs 단기 투자 선호
3. Financial Goal Orientation (재무 목표 지향성): 구체적 목표 vs 일반적 수익 추구
4. Information Processing Style (정보 처리 스타일): 분석적 vs 직관적 의사결정
5. Investment Fear (투자 두려움): 투자에 대한 두려움 정도 (낮을수록 높은 점수)
6. Investment Confidence (투자 자신감): 투자 결정에 대한 자신감

각 지표에 대해 0-100점으로 점수를 매기고, JSON 형식으로 출력해주세요:
{
  "scores": {
    "risk_tolerance": 점수,
    "investment_time_horizon": 점수,
    "financial_goal_orientation": 점수,
    "information_processing_style": 점수,
    "investment_fear": 점수,
    "investment_confidence": 점수
  },
  "reasoning": {
    "risk_tolerance": "점수 근거",
    "investment_time_horizon": "점수 근거",
    "financial_goal_orientation": "점수 근거",
    "information_processing_style": "점수 근거",
    "investment_fear": "점수 근거",
    "investment_confidence": "점수 근거"
  }
}"""
        }
        
        return default_prompts.get(prompt_name, f"기본 프롬프트를 찾을 수 없습니다: {prompt_name}")
    
    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate prompt structure and content.
        
        Args:
            prompt: Prompt string to validate
            
        Returns:
            True if prompt is valid
        """
        try:
            # REQUIRED: Basic validation
            if not prompt or len(prompt.strip()) < 10:
                return False
            
            # REQUIRED: Check for required Korean language instruction
            if "한국어" not in prompt:
                logger.warning("Prompt missing Korean language instruction")
                return False
            
            # REQUIRED: Check for proper structure
            required_elements = ["역할", "지시", "응답", "질문"]
            if not any(element in prompt for element in required_elements):
                logger.warning("Prompt missing structural elements")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Prompt validation error: {e}")
            return False
    
    @lru_cache(maxsize=128)
    def get_cached_prompt(self, prompt_name: str, user_profile_hash: Optional[str] = None) -> str:
        """
        Get cached personalized prompt.
        
        Args:
            prompt_name: Name of the prompt template
            user_profile_hash: Hash of user profile for caching
            
        Returns:
            Cached prompt string
        """
        return self.get_prompt(prompt_name)
    
    def reload_prompts(self) -> bool:
        """
        Reload all prompt templates from files.
        
        Returns:
            True if reload successful
        """
        try:
            # REQUIRED: Clear cache
            self.prompt_cache.clear()
            
            # REQUIRED: Reload all prompts
            for prompt_name in self.prompt_files.keys():
                self.get_prompt(prompt_name)
            
            logger.info("All prompts reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload prompts: {e}")
            return False
    
    def get_available_prompts(self) -> list:
        """Get list of available prompt templates."""
        return list(self.prompt_files.keys())
    
    def save_prompt(self, prompt_name: str, content: str) -> bool:
        """
        Save prompt template to file.
        
        Args:
            prompt_name: Name of the prompt
            content: Prompt content
            
        Returns:
            True if saved successfully
        """
        try:
            if prompt_name not in self.prompt_files:
                logger.error(f"Unknown prompt name: {prompt_name}")
                return False
            
            # REQUIRED: Validate prompt before saving
            if not self.validate_prompt(content):
                logger.error(f"Invalid prompt content for {prompt_name}")
                return False
            
            prompt_path = self.prompt_files[prompt_name]
            
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # REQUIRED: Update cache
            self.prompt_cache[prompt_name] = content
            
            logger.info(f"Prompt saved successfully: {prompt_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save prompt {prompt_name}: {e}")
            return False 