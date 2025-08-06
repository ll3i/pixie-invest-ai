#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Investment Advisor 테스트
- 멀티 에이전트 AI 시스템 테스트
- 설문 분석 기능 테스트
- 채팅 기능 테스트
"""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import sys
import os

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.investment_advisor import InvestmentAdvisor

class TestInvestmentAdvisor:
    """투자 어드바이저 테스트 클래스"""
    
    @pytest.fixture
    def advisor(self):
        """테스트용 어드바이저 인스턴스 생성"""
        with patch('src.investment_advisor.LLMService'), \
             patch('src.investment_advisor.MemoryManager'), \
             patch('src.investment_advisor.UserProfileAnalyzer'), \
             patch('src.investment_advisor.FinancialDataProcessor'), \
             patch('src.investment_advisor.get_user_profile_service'), \
             patch('src.investment_advisor.get_chat_history_service'):
            
            advisor = InvestmentAdvisor(api_type="simulation")
            advisor.set_session_id("test_session_123")
            return advisor
    
    def test_initialization(self, advisor):
        """어드바이저 초기화 테스트"""
        assert advisor.api_type == "simulation"
        assert advisor.session_id == "test_session_123"
        assert hasattr(advisor, 'llm_service')
        assert hasattr(advisor, 'memory_manager')
        assert hasattr(advisor, 'user_profile_analyzer')
        assert hasattr(advisor, 'financial_processor')
    
    def test_set_session_id(self, advisor):
        """세션 ID 설정 테스트"""
        new_session_id = "new_session_456"
        advisor.set_session_id(new_session_id)
        assert advisor.session_id == new_session_id
    
    def test_set_callbacks(self, advisor):
        """콜백 함수 설정 테스트"""
        status_callback = MagicMock()
        response_callback = MagicMock()
        
        advisor.set_status_callback(status_callback)
        advisor.set_response_callback(response_callback)
        
        assert advisor.status_callback == status_callback
        assert advisor.response_callback == response_callback
    
    @patch('src.investment_advisor.InvestmentAdvisor._get_user_profile')
    @patch('src.investment_advisor.InvestmentAdvisor._get_market_context')
    def test_chat_no_session(self, mock_market_context, mock_user_profile, advisor):
        """세션 없이 채팅 시도 테스트"""
        advisor.session_id = None
        result = advisor.chat("안녕하세요")
        assert "세션이 설정되지 않았습니다" in result
    
    @patch('src.investment_advisor.InvestmentAdvisor.generate_ai_a_response')
    @patch('src.investment_advisor.InvestmentAdvisor._generate_ai_a2_response')
    @patch('src.investment_advisor.InvestmentAdvisor._generate_ai_b_response')
    @patch('src.investment_advisor.InvestmentAdvisor._generate_final_response')
    @patch('src.investment_advisor.InvestmentAdvisor._get_user_profile')
    @patch('src.investment_advisor.InvestmentAdvisor._get_market_context')
    def test_chat_success_flow(self, mock_market_context, mock_user_profile,
                              mock_final, mock_ai_b, mock_ai_a2, mock_ai_a, advisor):
        """성공적인 채팅 플로우 테스트"""
        # Mock responses
        mock_user_profile.return_value = {"risk_tolerance": "medium"}
        mock_market_context.return_value = "시장 컨텍스트"
        mock_ai_a.return_value = {"success": True, "response": "AI-A 응답"}
        mock_ai_a2.return_value = {"success": True, "response": "AI-A2 응답"}
        mock_ai_b.return_value = {"success": True, "response": "AI-B 응답"}
        mock_final.return_value = {"success": True, "response": "최종 응답"}
        
        result = advisor.chat("투자 조언을 부탁합니다")
        
        assert result == "최종 응답"
        mock_ai_a.assert_called_once()
        mock_ai_a2.assert_called_once()
        mock_ai_b.assert_called_once()
        mock_final.assert_called_once()
    
    def test_analyze_survey_responses_no_session(self, advisor):
        """세션 없이 설문 분석 시도 테스트"""
        advisor.session_id = None
        
        answers = [
            {"question": "투자 경험이 있나요?", "answer": "5년 정도 있습니다"}
        ]
        
        result = advisor.analyze_survey_responses(answers)
        assert "error" in result
    
    @patch('src.investment_advisor.InvestmentAdvisor.user_profile_analyzer')
    def test_analyze_survey_responses_success(self, mock_analyzer, advisor):
        """설문 분석 성공 테스트"""
        mock_analysis_result = {
            "risk_tolerance": "medium",
            "investment_time_horizon": "long",
            "overall_analysis": "안정적인 투자 성향"
        }
        mock_analyzer.analyze_survey_responses.return_value = mock_analysis_result
        
        answers = [
            {"question": "투자 경험이 있나요?", "answer": "5년 정도 있습니다"}
        ]
        
        result = advisor.analyze_survey_responses(answers)
        assert result == mock_analysis_result
    
    def test_get_chat_history_no_session(self, advisor):
        """세션 없이 채팅 기록 조회 테스트"""
        advisor.session_id = None
        result = advisor.get_chat_history()
        assert result == []
    
    def test_test_api_connection(self, advisor):
        """API 연결 테스트"""
        advisor.llm_service.test_api_connection = MagicMock(return_value={"openai": True})
        result = advisor.test_api_connection()
        assert result == {"openai": True}
    
    def test_get_available_models(self, advisor):
        """사용 가능한 모델 조회 테스트"""
        advisor.llm_service.get_available_models = MagicMock(return_value=["gpt-3.5-turbo"])
        result = advisor.get_available_models()
        assert result == ["gpt-3.5-turbo"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 