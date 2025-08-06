#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core Investment Advisor with Multi-Agent AI System.
- Implements exact AI-A → AI-A2 → AI-B → Final response chain
- User profile integration and session management
- Comprehensive error handling and fallback mechanisms
- Status callbacks and progress tracking
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import json

from llm_service import LLMService
from memory_manager import MemoryManager
from user_profile_analyzer import UserProfileAnalyzer
from financial_data_processor import FinancialDataProcessor
from db_client import get_user_profile_service, get_chat_history_service

logger = logging.getLogger(__name__)

class InvestmentAdvisor:
    """Core investment advisor with multi-agent AI coordination."""
    
    def __init__(self, api_type: str = "openai", financial_processor: Optional[FinancialDataProcessor] = None):
        self.api_type = api_type
        self.session_id: Optional[str] = None
        
        # REQUIRED: Initialize core components
        self.llm_service = LLMService(api_type=api_type)
        self.memory_manager = MemoryManager()
        self.user_profile_analyzer = UserProfileAnalyzer()
        
        # Use provided financial_processor or None (for Supabase mode)
        self.financial_processor = financial_processor
        if financial_processor is not None:
            logger.info("Using provided FinancialDataProcessor instance")
        else:
            logger.info("Running without FinancialDataProcessor (Supabase mode)")
        
        # REQUIRED: Database services
        self.user_profile_service = get_user_profile_service()
        self.chat_history_service = get_chat_history_service()
        
        # REQUIRED: Callbacks for status updates
        self.status_callback: Optional[Callable[[str, str], None]] = None
        self.response_callback: Optional[Callable[[str, str], None]] = None
        
        logger.info(f"InvestmentAdvisor initialized with API type: {api_type}")
    
    def set_session_id(self, session_id: str) -> None:
        """
        Set session ID for user identification.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        logger.info(f"Session ID set: {session_id}")
    
    def set_status_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for status updates with agent name and status."""
        self.status_callback = callback
    
    def set_response_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for agent responses."""
        self.response_callback = callback
    
    def _update_status(self, status: str) -> None:
        """Update status via callback."""
        if self.status_callback:
            try:
                # Extract agent name from status if possible
                if ":" in status:
                    agent_name = status.split(":")[0].strip()
                    status_msg = "thinking"
                else:
                    agent_name = "System"
                    status_msg = status
                self.status_callback(agent_name, status_msg)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def _notify_response(self, agent: str, response: str) -> None:
        """Notify agent response via callback."""
        if self.response_callback:
            try:
                self.response_callback(agent, response)
            except Exception as e:
                logger.error(f"Response callback error: {e}")
    
    def chat(self, user_input: str) -> str:
        """
        MANDATORY: Execute complete AI agent chain in correct sequence.
        
        This method implements the exact AI-A → AI-A2 → AI-B → Final chain
        as required by cursor rules. NEVER deviate from this sequence.
        
        Args:
            user_input: User's input message
            
        Returns:
            Final synthesized response
        """
        if not self.session_id:
            return "세션이 설정되지 않았습니다. 다시 시도해주세요."
        
        try:
            # REQUIRED: Save user message
            self.memory_manager.add_user_ai_message(self.session_id, "user", user_input)
            self.chat_history_service.save_chat_message(self.session_id, "user", user_input)
            
            # REQUIRED: Get user profile and context
            self._update_status("사용자 프로필 분석 중...")
            analysis_result = self._get_user_profile()
            
            if analysis_result:
                logger.info(f"User profile loaded successfully for {self.session_id}")
                logger.info(f"Profile top-level keys: {list(analysis_result.keys())}")
                
                # Log the detailed structure
                if 'profile_json' in analysis_result:
                    logger.info(f"profile_json exists with keys: {list(analysis_result['profile_json'].keys())}")
                    if 'detailed_analysis' in analysis_result['profile_json']:
                        logger.info(f"detailed_analysis keys: {list(analysis_result['profile_json']['detailed_analysis'].keys())}")
                        # Log first analysis item to verify content
                        first_key = list(analysis_result['profile_json']['detailed_analysis'].keys())[0]
                        logger.info(f"Sample analysis ({first_key}): {analysis_result['profile_json']['detailed_analysis'][first_key][:200]}...")
                
                # Don't modify analysis_result here - let prompt_manager handle the structure
                logger.info(f"Passing full profile structure to LLM service")
            else:
                logger.warning(f"No user profile found for {self.session_id}")
            
            self._update_status("시장 컨텍스트 수집 중...")
            market_context = self._get_market_context(user_input)
            
            # MANDATORY: Step 1 - AI-A (Initial Response)
            self._update_status("AI-A: 초기 투자 조언 생성 중...")
            
            # Add explicit profile context if available
            profile_context = ""
            if analysis_result:
                logger.info("Adding explicit profile context to AI-A")
                if 'overall_analysis' in analysis_result:
                    profile_context = f"\n\n[사용자 프로필 요약]\n{analysis_result.get('overall_analysis', '')}\n"
                elif 'profile_json' in analysis_result and isinstance(analysis_result['profile_json'], dict):
                    overall = analysis_result['profile_json'].get('overall_analysis', '')
                    if overall:
                        profile_context = f"\n\n[사용자 프로필 요약]\n{overall}\n"
            
            enhanced_context = market_context + profile_context
            ai_a_response = self.generate_ai_a_response(user_input, analysis_result, enhanced_context)
            
            if not ai_a_response.get("success"):
                return self._handle_error("AI-A 응답 생성 실패", ai_a_response.get("error"))
            
            ai_a_text = ai_a_response["response"]
            logger.info(f"AI-A response (first 200 chars): {ai_a_text[:200]}...")
            self._notify_response("AI-A", ai_a_text)
            
            # MANDATORY: Step 2 - AI-A2 (Query Refinement)
            self._update_status("AI-A2: 쿼리 조정 및 데이터 요청 생성 중...")
            ai_a2_response = self._generate_ai_a2_response(user_input, ai_a_text, analysis_result)
            
            if not ai_a2_response.get("success"):
                return self._handle_error("AI-A2 응답 생성 실패", ai_a2_response.get("error"))
            
            ai_a2_text = ai_a2_response["response"]
            self._notify_response("AI-A2", ai_a2_text)
            
            # MANDATORY: Step 3 - AI-B (Data Analysis)
            self._update_status("AI-B: 금융 데이터 분석 중...")
            ai_b_response = self._generate_ai_b_response(ai_a2_text, analysis_result, market_context)
            
            if not ai_b_response.get("success"):
                return self._handle_error("AI-B 응답 생성 실패", ai_b_response.get("error"))
            
            ai_b_text = ai_b_response["response"]
            self._notify_response("AI-B", ai_b_text)
            
            # MANDATORY: Step 4 - Final Synthesis
            self._update_status("최종 응답 합성 중...")
            ai_conversation_history = self.memory_manager.get_ai_ai_history(self.session_id)
            final_response = self._generate_final_response(ai_conversation_history, user_input)
            
            if not final_response.get("success"):
                return self._handle_error("최종 응답 생성 실패", final_response.get("error"))
            
            final_text = final_response["response"]
            
            # REQUIRED: Save final response and cleanup
            self.memory_manager.add_user_ai_message(self.session_id, "assistant", final_text, "Final")
            self.chat_history_service.save_chat_message(self.session_id, "assistant", final_text, "Final")
            
            # REQUIRED: Reset AI-AI conversation after final response
            self.memory_manager.reset_ai_conversation(self.session_id)
            
            self._update_status("완료")
            return final_text
            
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return self._handle_error("채팅 처리 중 오류가 발생했습니다", str(e))
    
    def generate_ai_a_response(
        self, 
        user_input: str, 
        analysis_result: Optional[Dict[str, Any]], 
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate AI-A initial response.
        
        Args:
            user_input: User's question
            analysis_result: User profile analysis
            context: Market context
            
        Returns:
            AI-A response dictionary
        """
        try:
            # REQUIRED: Generate AI-A response
            response = self.llm_service.generate_ai_response(
                prompt_name="AI-A",
                user_message=user_input,
                context=context,
                analysis_result=analysis_result
            )
            
            # REQUIRED: Save to AI-AI history
            if response.get("success"):
                self.memory_manager.add_ai_ai_message(
                    self.session_id, 
                    "assistant", 
                    response["response"], 
                    "AI-A"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"AI-A response generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_ai_a2_response(
        self, 
        user_input: str, 
        ai_a_response: str, 
        analysis_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate AI-A2 query refinement response.
        
        Args:
            user_input: Original user question
            ai_a_response: AI-A's response
            analysis_result: User profile analysis
            
        Returns:
            AI-A2 response dictionary
        """
        try:
            # REQUIRED: Prepare context for AI-A2
            context = f"[AI-A 응답]\n{ai_a_response}\n\n[사용자 원본 질문]\n{user_input}"
            
            # REQUIRED: Generate AI-A2 response
            response = self.llm_service.generate_ai_response(
                prompt_name="AI-A2",
                user_message=user_input,
                context=context,
                analysis_result=analysis_result
            )
            
            # REQUIRED: Save to AI-AI history
            if response.get("success"):
                self.memory_manager.add_ai_ai_message(
                    self.session_id, 
                    "assistant", 
                    response["response"], 
                    "AI-A2"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"AI-A2 response generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_ai_b_response(
        self, 
        ai_a2_query: str, 
        analysis_result: Optional[Dict[str, Any]], 
        market_context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate AI-B data analysis response.
        
        Args:
            ai_a2_query: AI-A2's structured query
            analysis_result: User profile analysis
            market_context: Market data context
            
        Returns:
            AI-B response dictionary
        """
        try:
            # REQUIRED: Prepare enhanced context for AI-B
            enhanced_context = f"{market_context}\n\n[AI-A2 분석 요청]\n{ai_a2_query}"
            
            # Debug: Log context for AI-B
            logger.info(f"AI-B Context Length: {len(enhanced_context)}")
            
            # Log the entire evaluation context if it exists
            if "[주식 평가 데이터" in enhanced_context:
                eval_start = enhanced_context.find("[주식 평가 데이터")
                eval_end = enhanced_context.find("\n\n", eval_start + 100)  # Find end of evaluation section
                if eval_end == -1:
                    eval_end = len(enhanced_context)
                eval_section = enhanced_context[eval_start:eval_end]
                logger.info(f"AI-B Evaluation Context ({len(eval_section)} chars):")
                # Log first 3 stocks to verify current_price and market_cap are included
                stocks = eval_section.split("\n\n")[:3]
                for stock in stocks:
                    if stock.strip():
                        logger.info(f"Stock data:\n{stock[:500]}")
            else:
                logger.warning("No stock evaluation context found in AI-B context")
            
            # Check if stock evaluation context is included
            if "[주식 평가 데이터" in enhanced_context or "[주식 추천 결과]" in enhanced_context:
                logger.info("Stock evaluation context is included in AI-B context")
                # Find and log the stock recommendation part
                if "[주식 평가 데이터" in enhanced_context:
                    start_idx = enhanced_context.find("[주식 평가 데이터")
                else:
                    start_idx = enhanced_context.find("[주식 추천 결과]")
                end_idx = start_idx + 1000
                logger.info(f"Stock evaluation data preview: {enhanced_context[start_idx:end_idx]}...")
                
                # Count how many stock evaluations are included
                eval_count = enhanced_context.count("평가점수:")
                logger.info(f"Total {eval_count} stock evaluations included in context")
            else:
                logger.warning("Stock evaluation context is NOT included in AI-B context")
                logger.warning(f"Market context keywords found: {any(kw in market_context for kw in ['추천', '평가', '분석', '성장', '가치', '수익', '좋은', '우량', '배당', '안정'])}")
            
            # REQUIRED: Generate AI-B response
            response = self.llm_service.generate_ai_response(
                prompt_name="AI-B",
                user_message=ai_a2_query,
                context=enhanced_context,
                analysis_result=analysis_result
            )
            
            # Debug: Log AI-B response
            if response.get("success"):
                logger.info(f"AI-B Response preview (first 1000 chars): {response['response'][:1000]}...")
                
                # Check if current_price is in the response
                if "현재가:" in response['response']:
                    logger.info("AI-B response includes 현재가 (current_price)")
                    # Extract and log some current_price values
                    import re
                    price_matches = re.findall(r'현재가: ([\d,]+)원', response['response'])
                    if price_matches:
                        logger.info(f"Found {len(price_matches)} current_price values: {price_matches[:3]}")
                else:
                    logger.warning("AI-B response does NOT include 현재가 (current_price)")
                    logger.error("AI-B failed to extract current_price from data!")
                    
                if "시가총액:" in response['response']:
                    logger.info("AI-B response includes 시가총액 (market_cap)")
                    # Extract and log some market_cap values
                    cap_matches = re.findall(r'시가총액: ([\d,]+)억원', response['response'])
                    if cap_matches:
                        logger.info(f"Found {len(cap_matches)} market_cap values: {cap_matches[:3]}")
                else:
                    logger.warning("AI-B response does NOT include 시가총액 (market_cap)")
                    logger.error("AI-B failed to extract market_cap from data!")
            
            # REQUIRED: Save to AI-AI history
            if response.get("success"):
                # Log the response for debugging
                logger.info(f"AI-B raw response preview: {response['response'][:500]}...")
                
                # Temporarily disable filtering to test if AI-B can use data correctly
                # try:
                #     from response_filter import filter_ai_response, validate_stock_data
                #     
                #     original_response = response["response"]
                #     
                #     # Check if response uses real data
                #     if not validate_stock_data(original_response, enhanced_context):
                #         logger.warning("AI-B didn't use real data, applying filter")
                #         filtered_response = filter_ai_response(original_response, enhanced_context)
                #         response["response"] = filtered_response
                #         logger.info("Response filtered to use real data")
                #     
                # except Exception as e:
                #     logger.error(f"Response filtering failed: {e}")
                
                self.memory_manager.add_ai_ai_message(
                    self.session_id, 
                    "assistant", 
                    response["response"], 
                    "AI-B"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"AI-B response generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_final_response(
        self, 
        ai_conversation_history: List[Dict[str, Any]], 
        user_query: str
    ) -> Dict[str, Any]:
        """
        Generate final synthesized response.
        
        Args:
            ai_conversation_history: Complete AI-AI conversation
            user_query: Original user question
            
        Returns:
            Final response dictionary
        """
        try:
            # REQUIRED: Prepare conversation summary
            conversation_summary = "\n".join([
                f"[{msg.get('agent', 'Unknown')}]: {msg.get('content', '')}"
                for msg in ai_conversation_history
            ])
            
            # REQUIRED: Prepare final prompt
            final_prompt = f"""다음은 AI 에이전트들 간의 대화 내용입니다:

{conversation_summary}

**답변 작성 규칙 (반드시 준수)**:

1. 종목 추천 시 다음 형식을 정확히 따르세요:

### 추천 종목

1. **종목명(티커)** - 포트폴리오 비중%
   - 현재가: [AI-B가 제공한 실제 가격]원 (예: 15,240원)
   - 시가총액: [AI-B가 제공한 실제 시가총액]억원 (예: 3,450억원)
   - PER: [AI-B가 제공한 실제 값], PBR: [AI-B가 제공한 실제 값]
   - 매출성장률: [AI-B가 제공한 실제 값]%
   - 순이익률: [AI-B가 제공한 실제 값]%
   - 부채비율: [AI-B가 제공한 실제 값]%
   - 평가점수: [AI-B가 제공한 실제 점수]점
   - 추천 이유: [AI-B가 제공한 실제 이유]
   
   **절대로 "정보 없음"이나 "확인 필요"라고 쓰지 마세요**

2. 절대 금지사항:
   - "정보 없음", "실시간 확인 필요" 같은 표현 사용 금지
   - AI-B가 제공하지 않은 종목 추가 금지
   - 임의의 가격이나 수치 생성 금지
   - SK텔레콤, 현대차, KT&G 등 일반적인 주식 추가 금지

3. AI-B의 데이터를 그대로 사용하세요. 절대 수정하지 마세요.

사용자 원본 질문: {user_query}"""
            
            # REQUIRED: Generate final response
            response = self.llm_service.generate_ai_response(
                prompt_name="AI-A",  # Use AI-A for final synthesis
                user_message=final_prompt,
                context="",
                analysis_result=None
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Final response generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_user_profile(self) -> Optional[Dict[str, Any]]:
        """Get user profile for current session."""
        try:
            if not self.session_id:
                return None
            
            # REQUIRED: Try to get from cache first
            cached_profile = self.memory_manager.get_cached_context(self.session_id, "user_profile")
            if cached_profile:
                logger.info(f"Using cached profile for {self.session_id}")
                return cached_profile
            
            # REQUIRED: Get from database
            profile = self.user_profile_service.get_user_profile(self.session_id)
            
            if profile:
                logger.info(f"Retrieved profile from database for {self.session_id}")
                logger.info(f"Profile structure: {list(profile.keys())}")
                
                # If profile has profile_json as a nested structure, use it directly
                if 'profile_json' in profile and isinstance(profile['profile_json'], dict):
                    logger.info(f"Found nested profile_json, using it as the main profile")
                    # Return the profile_json directly as the profile
                    actual_profile = profile['profile_json']
                    self.memory_manager.cache_context(self.session_id, "user_profile", actual_profile)
                    return actual_profile
                else:
                    # Use the profile as is
                    self.memory_manager.cache_context(self.session_id, "user_profile", profile)
                    return profile
            
            logger.warning(f"No profile found for {self.session_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return None
    
    def _get_market_context_from_supabase(self, user_message: str) -> str:
        """
        Get market context directly from Supabase (without FinancialDataProcessor).
        """
        try:
            from db_client import get_supabase_client
            supabase = get_supabase_client()
            
            if not supabase:
                return "실시간 시장 데이터를 가져올 수 없습니다."
            
            # 뉴스 요약
            news_context = ""
            try:
                # 최근 뉴스 가져오기
                news_res = supabase.table("news").select("*").order("created_at", desc=True).limit(5).execute()
                if news_res.data:
                    news_context = "[최신 금융 뉴스]\n"
                    for news in news_res.data[:3]:
                        news_context += f"- {news.get('title', '')}\n"
            except Exception as e:
                logger.error(f"뉴스 조회 오류: {e}")
            
            # 주요 종목 데이터
            stock_context = ""
            try:
                # 주요 종목 가격 정보
                key_tickers = ['005930', '000660', '035420', '035720', '051910']
                stock_context = "\n[주요 종목 현황]\n"
                for ticker in key_tickers:
                    price_res = supabase.table("kor_stock_prices").select("*").eq("ticker", ticker).order("date", desc=True).limit(1).execute()
                    if price_res.data and len(price_res.data) > 0:
                        price = price_res.data[0]
                        stock_context += f"{price.get('name', ticker)}({ticker}): {price.get('close', 0):,}원\n"
            except Exception as e:
                logger.error(f"주가 조회 오류: {e}")
            
            # 종목 평가 정보
            eval_context = ""
            if any(keyword in user_message for keyword in ["추천", "평가", "분석", "성장", "가치", "수익", "좋은", "우량", "배당", "안정", "종목", "투자"]):
                # 안정성 중시 키워드 확인
                is_stable_focused = any(keyword in user_message for keyword in ["안정", "보수", "안전", "리스크", "위험"])
                try:
                    eval_res = supabase.table("kor_stock_evaluations").select(
                        "id,ticker,name,reference_date,current_price,market_cap,revenue_growth,profit_margin,debt_ratio,per,pbr,score,evaluation,reasons,created_at"
                    ).order("score", desc=True).limit(20).execute()
                    
                    if eval_res.data:
                        eval_context = "\n[주식 평가 데이터 (kor_stock_evaluations 테이블 기준)]\n"
                        eval_context += f"기준일: {eval_res.data[0].get('reference_date', 'N/A')}\n\n"
                        
                        for idx, eval_data in enumerate(eval_res.data[:10], 1):
                            eval_context += f"{idx}. 종목명: {eval_data['name']}, 티커: {eval_data['ticker']}\n"
                            # 숫자 포맷팅 개선
                            current_price = eval_data.get('current_price', 0)
                            market_cap = eval_data.get('market_cap', 0)
                            
                            # AI-B가 쉽게 파싱할 수 있도록 명확한 필드명 사용
                            eval_context += f"   - name: {eval_data['name']}\n"
                            eval_context += f"   - ticker: {eval_data['ticker']}\n"
                            eval_context += f"   - current_price: {int(current_price):,}\n" if current_price else f"   - current_price: 0\n"
                            
                            # 시가총액 억원 단위로 변환 및 표시
                            if market_cap and market_cap > 0:
                                market_cap_billion = market_cap / 100000000  # 억원 단위
                                eval_context += f"   - market_cap: {market_cap_billion:,.0f}\n"
                            else:
                                eval_context += f"   - market_cap: 0\n"
                            eval_context += f"   - score: {eval_data.get('score', 0)}\n"
                            eval_context += f"   - evaluation: {eval_data.get('evaluation', 'N/A')}\n"
                            eval_context += f"   - per: {eval_data.get('per', 'N/A')}\n"
                            eval_context += f"   - pbr: {eval_data.get('pbr', 'N/A')}\n"
                            eval_context += f"   - revenue_growth: {eval_data.get('revenue_growth', 'N/A')}\n"
                            eval_context += f"   - profit_margin: {eval_data.get('profit_margin', 'N/A')}\n"
                            eval_context += f"   - debt_ratio: {eval_data.get('debt_ratio', 'N/A')}\n"
                            eval_context += f"   - reasons: {eval_data.get('reasons', 'N/A')}\n"
                            eval_context += f"   - reference_date: {eval_data.get('reference_date', 'N/A')}\n\n"
                        
                        logger.info(f"kor_stock_evaluations 테이블에서 {len(eval_res.data)}개 종목 평가 정보 로드")
                        
                        # 디버그: 첫 번째 종목 데이터 확인
                        if eval_res.data:
                            first_stock = eval_res.data[0]
                            logger.info(f"첫 번째 종목 데이터 확인: {first_stock['name']}({first_stock['ticker']})")
                            logger.info(f"  - 현재가: {first_stock.get('current_price', 'None')}")
                            logger.info(f"  - 시가총액: {first_stock.get('market_cap', 'None')}")
                            logger.info(f"  - PER: {first_stock.get('per', 'None')}, PBR: {first_stock.get('pbr', 'None')}")
                except Exception as e:
                    logger.error(f"평가 정보 조회 오류: {e}")
            
            # 전체 컨텍스트 조합
            context = f"{news_context}\n{stock_context}\n{eval_context}".strip()
            
            # 디버그 로그 추가
            if eval_context:
                logger.info(f"종목 평가 컨텍스트 포함됨 (길이: {len(eval_context)}자)")
                logger.debug(f"평가 컨텍스트 미리보기: {eval_context[:500]}...")
            else:
                logger.warning("종목 평가 컨텍스트가 비어있음")
            
            return context if context else "실시간 시장 데이터를 가져올 수 없습니다."
            
        except Exception as e:
            logger.error(f"Supabase 시장 컨텍스트 수집 오류: {e}")
            return "실시간 시장 데이터를 가져올 수 없습니다."
    
    def _get_market_context(self, user_message: str) -> str:
        """
        Get market context for user query.
        
        Args:
            user_message: User's message for context relevance
            
        Returns:
            Combined market context string
        """
        try:
            # Check if we have financial_processor
            if self.financial_processor is None:
                # Use Supabase directly
                return self._get_market_context_from_supabase(user_message)
            
            # REQUIRED: Get news context (max 5 items)
            news_context = self.financial_processor.get_news_context(user_message)
            if news_context and news_context.count('\n') > 5:
                news_context = '\n'.join(news_context.split('\n')[:6])
            
            # 뉴스 감정 분석 요약 추가
            news_sentiment_summary = self._get_news_sentiment_summary(user_message)
            if news_sentiment_summary:
                news_context = f"{news_context}\n\n{news_sentiment_summary}"
            
            # REQUIRED: Get risk alerts context (max 5 items)  
            alerts_context = self.financial_processor.get_risk_alerts_context(user_message)
            if alerts_context and alerts_context.count('\n') > 5:
                alerts_context = '\n'.join(alerts_context.split('\n')[:6])
            
            # REQUIRED: Get portfolio context for relevant queries
            portfolio_context = ""
            portfolio_keywords = ["포트폴리오", "주식", "투자", "종목", "배분", "추천"]
            if any(keyword in user_message for keyword in portfolio_keywords):
                recent_recs = self._get_recent_portfolio_recommendations(limit=1)
                if recent_recs:
                    portfolio_context = f"\n\n[최근 추천 포트폴리오]\n{json.dumps(recent_recs[0], ensure_ascii=False)}"
            
            # 실제 주식 데이터 추가
            stock_data_context = self._get_stock_data_context(user_message)
            if stock_data_context:
                portfolio_context += f"\n\n{stock_data_context}"
            
            # notebook 기반 주식 평가 컨텍스트 추가
            stock_evaluation_context = ""
            evaluation_keywords = ["추천", "평가", "분석", "성장", "가치", "수익", "좋은", "우량", "배당", "안정", "포트폴리오", "구성"]
            logger.info(f"Checking evaluation keywords in user message: {user_message}")
            if any(keyword in user_message for keyword in evaluation_keywords):
                logger.info("Evaluation keywords found, getting stock evaluation context")
                stock_evaluation_context = self.financial_processor.get_stock_evaluation_context(user_message)
                if stock_evaluation_context:
                    logger.info(f"Stock evaluation context retrieved: {len(stock_evaluation_context)} chars")
                    # Log first few lines to verify data format
                    preview_lines = stock_evaluation_context.split('\n')[:10]
                    logger.info(f"Stock evaluation preview: {preview_lines}")
                    portfolio_context += f"\n\n{stock_evaluation_context}"
                else:
                    logger.warning("No stock evaluation context returned")
            
            # REQUIRED: Combine all contexts
            full_context = f"{news_context}\n\n{alerts_context}{portfolio_context}"
            
            return full_context.strip()
            
        except Exception as e:
            logger.error(f"Failed to get market context: {e}")
            return ""
    
    def _get_news_sentiment_summary(self, user_message: str) -> str:
        """뉴스 감정 분석 요약 생성
        
        Args:
            user_message: 사용자 메시지
            
        Returns:
            뉴스 감정 분석 요약 문자열
        """
        try:
            # If no financial_processor, skip
            if self.financial_processor is None:
                return ""
                
            # 관련 뉴스 검색
            news_docs = self.financial_processor.search_relevant_documents(
                user_message, top_k=10, category_filter="news"
            )
            
            if not news_docs:
                return ""
            
            # 감정 점수 기반 통계
            positive_count = sum(1 for doc in news_docs if doc.get("sentiment_score", 0.5) > 0.7)
            negative_count = sum(1 for doc in news_docs if doc.get("sentiment_score", 0.5) < 0.3)
            neutral_count = len(news_docs) - positive_count - negative_count
            
            avg_sentiment = sum(doc.get("sentiment_score", 0.5) for doc in news_docs) / len(news_docs)
            
            # 전반적인 시장 분위기 판단
            if avg_sentiment > 0.6:
                overall_sentiment = "긍정적"
                market_advice = "시장 심리가 개선되고 있어 적극적인 투자를 고려해볼 수 있습니다."
            elif avg_sentiment < 0.4:
                overall_sentiment = "부정적"
                market_advice = "시장 불안이 확대되고 있어 리스크 관리에 주의가 필요합니다."
            else:
                overall_sentiment = "중립적"
                market_advice = "시장이 방향성을 모색하고 있어 선별적 투자가 유효합니다."
            
            summary = f"[뉴스 감정 분석 요약]\n"
            summary += f"전체 {len(news_docs)}개 뉴스 중: 긍정 {positive_count}개, 부정 {negative_count}개, 중립 {neutral_count}개\n"
            summary += f"전반적인 시장 분위기: {overall_sentiment} (평균 점수: {avg_sentiment:.2f})\n"
            summary += f"투자 조언: {market_advice}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get news sentiment summary: {e}")
            return ""
    
    def _get_stock_data_context(self, user_message: str) -> str:
        """주요 주식의 실제 데이터를 가져옵니다."""
        try:
            # 주요 종목 리스트
            key_stocks = ["005930", "005380", "051910", "000660", "035420", "035720", "006400", "000270"]
            stock_names = {
                "005930": "삼성전자",
                "005380": "현대차", 
                "051910": "LG화학",
                "000660": "SK하이닉스",
                "035420": "NAVER",
                "035720": "카카오",
                "006400": "삼성SDI",
                "000270": "기아"
            }
            
            stock_data_text = "[주요 종목 최신 데이터]\n"
            
            # 캐시에서 주식 데이터 가져오기
            for ticker in key_stocks:
                if ticker in self.financial_processor.market_data_cache:
                    data = self.financial_processor.market_data_cache[ticker]
                    name = stock_names.get(ticker, data.get('name', ticker))
                    
                    stock_info = f"{name}({ticker}): "
                    
                    # 가격 정보
                    if 'close' in data:
                        stock_info += f"현재가 {data['close']:,.0f}원"
                    
                    # 밸류에이션 정보
                    if 'valuations' in data:
                        vals = data['valuations']
                        if 'PER' in vals:
                            stock_info += f", PER {vals['PER']:.1f}"
                        if 'PBR' in vals:
                            stock_info += f", PBR {vals['PBR']:.1f}"
                        if 'DY' in vals:
                            stock_info += f", 배당수익률 {vals['DY']:.1f}%"
                    
                    stock_data_text += stock_info + "\n"
            
            # 주식 관련 문서 검색
            stock_docs = self.financial_processor.search_relevant_documents(
                user_message, top_k=5, category_filter="stock_price"
            )
            
            if stock_docs:
                stock_data_text += "\n[관련 종목 정보]\n"
                for doc in stock_docs[:3]:
                    stock_data_text += f"- {doc['title']}: {doc['content']}\n"
            
            return stock_data_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to get stock data context: {e}")
            return ""
    
    def _get_recent_portfolio_recommendations(self, limit: int = 1) -> List[Dict[str, Any]]:
        """Get recent portfolio recommendations."""
        try:
            # PLACEHOLDER: Implement portfolio recommendation retrieval
            # This would query the database for recent recommendations
            return []
            
        except Exception as e:
            logger.error(f"Failed to get portfolio recommendations: {e}")
            return []
    
    def _handle_error(self, message: str, error_details: str) -> str:
        """
        Handle errors with graceful fallback.
        
        Args:
            message: User-friendly error message
            error_details: Technical error details
            
        Returns:
            Fallback response text
        """
        logger.error(f"{message}: {error_details}")
        
        # REQUIRED: Update status
        self._update_status(f"오류: {message}")
        
        # REQUIRED: Return user-friendly error message
        return f"죄송합니다. {message} 기본적인 분산투자 원칙을 권장드리며, 잠시 후 다시 시도해주세요."
    
    def analyze_survey_responses(self, answers: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze survey responses to generate user profile.
        
        Args:
            answers: List of survey question-answer pairs
            
        Returns:
            Analysis result with scores and recommendations
        """
        try:
            if not self.session_id:
                raise ValueError("Session ID not set")
            
            # REQUIRED: Analyze responses
            result = self.user_profile_analyzer.analyze_survey_responses(answers)
            
            # REQUIRED: Save profile to database
            if result:
                self.user_profile_service.save_user_profile(self.session_id, result)
                
                # REQUIRED: Cache profile
                self.memory_manager.cache_context(self.session_id, "user_profile", result)
                
                # REQUIRED: Update session metadata
                self.memory_manager.update_session_metadata(self.session_id, {
                    "profile_analyzed": True,
                    "profile_created_at": datetime.now().isoformat()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Survey analysis failed: {e}")
            return {"error": str(e)}
    
    def get_chat_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get chat history for current session.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of chat messages
        """
        if not self.session_id:
            return []
        
        try:
            return self.memory_manager.get_user_ai_history(self.session_id, limit)
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []
    
    def test_api_connection(self) -> Dict[str, bool]:
        """Test API connections."""
        return self.llm_service.test_api_connection()
    
    def get_available_models(self) -> List[str]:
        """Get available AI models."""
        return self.llm_service.get_available_models() 