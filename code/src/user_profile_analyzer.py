#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
User Profile Analyzer for Investment Recommendations.
- Survey response analysis and scoring
- User investment profile generation
- Embedding generation using sentence transformers
- Profile persistence and caching
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from llm_service import LLMService
from config import config

logger = logging.getLogger(__name__)

class UserProfileAnalyzer:
    """Analyzes user investment profiles and generates personalized recommendations."""
    
    def __init__(self):
        # REQUIRED: Initialize LLM service
        self.llm_service = LLMService()
        
        # REQUIRED: Initialize embedding model
        self.embedding_model = self._load_embedding_model()
        
        # REQUIRED: Cache for user profiles
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        
        logger.info("UserProfileAnalyzer initialized successfully")
    
    def _load_embedding_model(self) -> Optional[SentenceTransformer]:
        """
        Load sentence transformer model for embeddings.
        
        Returns:
            SentenceTransformer model or None if not available
        """
        if not SentenceTransformer:
            logger.warning("sentence-transformers not available, embeddings disabled")
            return None
        
        try:
            # REQUIRED: Use Korean-optimized model
            model = SentenceTransformer('jhgan/ko-sbert-multitask')
            logger.info("Embedding model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None
    
    def analyze_survey_responses(self, answers: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze survey responses to calculate investment preference scores.
        
        Args:
            answers: List of survey question-answer pairs
                    Example: [{"question": "Risk preference?", "answer": "High risk"}]
        
        Returns:
            Dictionary of investment scores (0-100 scale)
            Keys: risk_tolerance, investment_time_horizon, financial_goal_orientation,
                  information_processing_style, investment_fear, investment_confidence
                  
        Raises:
            ValueError: If answers list is empty or malformed
            
        Example:
            scores = analyzer.analyze_survey_responses([
                {"question": "Risk tolerance?", "answer": "I prefer safe investments"}
            ])
            # Returns: {"risk_tolerance": 25, "investment_time_horizon": 50, ...}
        """
        try:
            # REQUIRED: Input validation
            if not answers or not isinstance(answers, list):
                raise ValueError("Answers list cannot be empty")
            
            if len(answers) < 3:
                raise ValueError("Minimum 3 survey responses required")
            
            # REQUIRED: Validate answer format
            for i, answer in enumerate(answers):
                if not isinstance(answer, dict) or 'answer' not in answer:
                    raise ValueError(f"Answer {i} must be a dictionary with 'answer' key")
            
            # REQUIRED: Step 1 - Generate scores using LLM
            scores = self._generate_investment_scores(answers)
            
            # REQUIRED: Step 2 - Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(scores, answers)
            
            # REQUIRED: Step 3 - Generate embeddings
            profile_embedding = self._generate_profile_embedding(answers, scores)
            
            # REQUIRED: Step 4 - Create comprehensive profile
            profile = {
                "scores": scores,
                "detailed_analysis": detailed_analysis,
                "embedding": profile_embedding,
                "survey_responses": answers,
                "created_at": datetime.now().isoformat(),
                "overall_analysis": detailed_analysis.get("overall_analysis", ""),
                "investment_strategy": detailed_analysis.get("investment_strategy", ""),
                "recommended_products": detailed_analysis.get("recommended_products", [])
            }
            
            return profile
            
        except Exception as e:
            logger.error(f"Survey analysis failed: {e}")
            raise
    
    def _generate_investment_scores(self, answers: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Generate investment scores using LLM analysis.
        
        Args:
            answers: Survey responses
            
        Returns:
            Dictionary of scores (0-100)
        """
        try:
            # REQUIRED: Prepare survey data for LLM
            survey_text = "\n".join([
                f"질문: {answer.get('question', '질문 없음')}\n답변: {answer['answer']}\n"
                for answer in answers
            ])
            
            # REQUIRED: Generate scores using LLM
            response = self.llm_service.generate_ai_response(
                prompt_name="survey-score",
                user_message=survey_text,
                context="",
                analysis_result=None
            )
            
            if not response.get("success"):
                logger.warning("LLM score generation failed, using fallback")
                return self._generate_fallback_scores(answers)
            
            # REQUIRED: Parse JSON response
            try:
                scores_data = json.loads(response["response"])
                scores = scores_data.get("scores", {})
                
                # REQUIRED: Validate score ranges
                validated_scores = {}
                required_keys = [
                    "risk_tolerance", "investment_time_horizon", 
                    "financial_goal_orientation", "information_processing_style",
                    "investment_fear", "investment_confidence"
                ]
                
                for key in required_keys:
                    score = scores.get(key, 50)  # Default to middle score
                    validated_scores[key] = max(0, min(100, float(score)))
                
                return validated_scores
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM scores: {e}")
                return self._generate_fallback_scores(answers)
                
        except Exception as e:
            logger.error(f"Score generation failed: {e}")
            return self._generate_fallback_scores(answers)
    
    def _generate_detailed_analysis(
        self, 
        scores: Dict[str, float], 
        answers: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Generate detailed profile analysis using LLM.
        
        Args:
            scores: Investment scores
            answers: Survey responses
            
        Returns:
            Detailed analysis dictionary
        """
        try:
            # REQUIRED: Prepare analysis input
            analysis_input = {
                "scores": scores,
                "survey_responses": answers
            }
            
            analysis_text = json.dumps(analysis_input, ensure_ascii=False, indent=2)
            
            # REQUIRED: Generate detailed analysis
            response = self.llm_service.generate_ai_response(
                prompt_name="survey-analysis",
                user_message=analysis_text,
                context="",
                analysis_result=None
            )
            
            if not response.get("success"):
                logger.warning("LLM analysis generation failed, using fallback")
                return self._generate_fallback_analysis(scores)
            
            # REQUIRED: Parse JSON response
            try:
                analysis_data = json.loads(response["response"])
                return analysis_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM analysis: {e}")
                return self._generate_fallback_analysis(scores)
                
        except Exception as e:
            logger.error(f"Detailed analysis generation failed: {e}")
            return self._generate_fallback_analysis(scores)
    
    def _generate_profile_embedding(
        self, 
        answers: List[Dict[str, str]], 
        scores: Dict[str, float]
    ) -> Optional[List[float]]:
        """
        Generate profile embedding using sentence transformers.
        
        Args:
            answers: Survey responses
            scores: Investment scores
            
        Returns:
            Profile embedding vector or None
        """
        if not self.embedding_model:
            logger.warning("Embedding model not available")
            return None
        
        try:
            # REQUIRED: Combine text for embedding
            answer_texts = [answer['answer'] for answer in answers]
            combined_text = " ".join(answer_texts)
            
            # REQUIRED: Add score information
            score_text = " ".join([f"{key}_{value}" for key, value in scores.items()])
            full_text = f"{combined_text} {score_text}"
            
            # REQUIRED: Generate embedding
            embedding = self.embedding_model.encode(full_text)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _generate_fallback_scores(self, answers: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Generate fallback scores when LLM fails.
        
        Args:
            answers: Survey responses
            
        Returns:
            Basic scores based on keyword analysis
        """
        try:
            scores = {
                "risk_tolerance": 50.0,
                "investment_time_horizon": 50.0,
                "financial_goal_orientation": 50.0,
                "information_processing_style": 50.0,
                "investment_fear": 50.0,
                "investment_confidence": 50.0
            }
            
            # REQUIRED: Simple keyword-based scoring
            combined_text = " ".join([answer['answer'].lower() for answer in answers])
            
            # Risk tolerance keywords
            if any(word in combined_text for word in ["안전", "보수", "위험하지", "확실"]):
                scores["risk_tolerance"] = 25.0
            elif any(word in combined_text for word in ["위험", "모험", "공격적", "높은 수익"]):
                scores["risk_tolerance"] = 75.0
            
            # Time horizon keywords
            if any(word in combined_text for word in ["단기", "빠른", "즉시", "1년"]):
                scores["investment_time_horizon"] = 25.0
            elif any(word in combined_text for word in ["장기", "오랫동안", "10년", "은퇴"]):
                scores["investment_time_horizon"] = 75.0
            
            # Confidence keywords
            if any(word in combined_text for word in ["확신", "자신", "경험", "알고"]):
                scores["investment_confidence"] = 75.0
            elif any(word in combined_text for word in ["잘 모르", "처음", "어려워", "힘들"]):
                scores["investment_confidence"] = 25.0
            
            logger.info("Generated fallback scores using keyword analysis")
            return scores
            
        except Exception as e:
            logger.error(f"Fallback scoring failed: {e}")
            # REQUIRED: Return default middle scores
            return {
                "risk_tolerance": 50.0,
                "investment_time_horizon": 50.0,
                "financial_goal_orientation": 50.0,
                "information_processing_style": 50.0,
                "investment_fear": 50.0,
                "investment_confidence": 50.0
            }
    
    def _generate_fallback_analysis(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate fallback analysis when LLM fails.
        
        Args:
            scores: Investment scores
            
        Returns:
            Basic analysis based on scores
        """
        try:
            risk_level = scores.get("risk_tolerance", 50)
            timeframe = scores.get("investment_time_horizon", 50)
            confidence = scores.get("investment_confidence", 50)
            
            # REQUIRED: Generate basic analysis
            if risk_level < 30:
                risk_analysis = "안전 지향적인 투자 성향으로 원금 보존을 중시합니다."
                recommended_products = ["예금", "적금", "국채", "안전자산"]
            elif risk_level > 70:
                risk_analysis = "위험 감수 의향이 높아 적극적인 투자를 선호합니다."
                recommended_products = ["성장주", "해외주식", "섹터ETF", "개별주식"]
            else:
                risk_analysis = "균형잡힌 투자 성향으로 안정성과 수익성을 함께 추구합니다."
                recommended_products = ["혼합형펀드", "인덱스펀드", "ETF", "균형포트폴리오"]
            
            if timeframe < 30:
                timeframe_analysis = "단기 투자를 선호하며 빠른 성과를 기대합니다."
            elif timeframe > 70:
                timeframe_analysis = "장기 투자 관점으로 복리 효과를 중시합니다."
            else:
                timeframe_analysis = "중기적 투자 목표를 가지고 있습니다."
            
            investment_strategy = f"위험도 {risk_level:.0f}%, 투자기간 {timeframe:.0f}%에 맞는 포트폴리오 구성을 권장합니다."
            
            return {
                "detailed_analysis": {
                    "risk_tolerance_analysis": risk_analysis,
                    "investment_time_horizon_analysis": timeframe_analysis,
                    "financial_goal_orientation_analysis": "재무 목표에 따른 투자 계획이 필요합니다.",
                    "information_processing_style_analysis": "투자 정보 처리 방식에 맞는 조언이 필요합니다.",
                    "investment_fear_analysis": "투자에 대한 두려움을 고려한 접근이 필요합니다.",
                    "investment_confidence_analysis": f"투자 자신감 수준({confidence:.0f}%)에 맞는 단계적 접근을 권장합니다."
                },
                "overall_analysis": f"위험 감수성 {risk_level:.0f}%, 투자 기간 {timeframe:.0f}%의 특성을 가진 투자자입니다.",
                "investment_strategy": investment_strategy,
                "recommended_products": recommended_products
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis generation failed: {e}")
            return {
                "detailed_analysis": {},
                "overall_analysis": "기본적인 분산투자를 권장합니다.",
                "investment_strategy": "안정적인 포트폴리오 구성",
                "recommended_products": ["인덱스펀드", "ETF"]
            }
    
    def get_profile_similarity(
        self, 
        profile1: Dict[str, Any], 
        profile2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two user profiles.
        
        Args:
            profile1: First user profile
            profile2: Second user profile
            
        Returns:
            Similarity score (0-1)
        """
        if not self.embedding_model:
            return 0.0
        
        try:
            embedding1 = profile1.get("embedding")
            embedding2 = profile2.get("embedding")
            
            if not embedding1 or not embedding2:
                return 0.0
            
            # REQUIRED: Calculate cosine similarity
            import numpy as np
            
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(cosine_sim)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def update_profile(
        self, 
        user_id: str, 
        profile_updates: Dict[str, Any]
    ) -> bool:
        """
        Update existing user profile.
        
        Args:
            user_id: User identifier
            profile_updates: Updates to apply
            
        Returns:
            True if update successful
        """
        try:
            if user_id in self.user_profiles:
                self.user_profiles[user_id].update(profile_updates)
                self.user_profiles[user_id]["updated_at"] = datetime.now().isoformat()
                logger.info(f"Profile updated for user {user_id}")
                return True
            else:
                logger.warning(f"Profile not found for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Profile update failed: {e}")
            return False
    
    def get_profile_summary(self, profile: Dict[str, Any]) -> str:
        """
        Get human-readable profile summary.
        
        Args:
            profile: User profile data
            
        Returns:
            Profile summary text
        """
        try:
            scores = profile.get("scores", {})
            
            risk_level = scores.get("risk_tolerance", 50)
            timeframe = scores.get("investment_time_horizon", 50)
            confidence = scores.get("investment_confidence", 50)
            
            # REQUIRED: Generate readable summary
            if risk_level < 30:
                risk_desc = "보수적"
            elif risk_level > 70:
                risk_desc = "적극적"
            else:
                risk_desc = "균형잡힌"
            
            if timeframe < 30:
                time_desc = "단기"
            elif timeframe > 70:
                time_desc = "장기"
            else:
                time_desc = "중기"
            
            if confidence < 30:
                conf_desc = "초보"
            elif confidence > 70:
                conf_desc = "경험있는"
            else:
                conf_desc = "중급"
            
            summary = f"{risk_desc} 성향의 {time_desc} 투자자 ({conf_desc} 수준)"
            
            return summary
            
        except Exception as e:
            logger.error(f"Profile summary generation failed: {e}")
            return "투자 성향 분석 필요"
    
    def validate_profile(self, profile: Dict[str, Any]) -> bool:
        """
        Validate profile data structure.
        
        Args:
            profile: Profile data to validate
            
        Returns:
            True if profile is valid
        """
        try:
            # REQUIRED: Check required fields
            required_fields = ["scores", "detailed_analysis", "created_at"]
            for field in required_fields:
                if field not in profile:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # REQUIRED: Validate scores
            scores = profile["scores"]
            required_scores = [
                "risk_tolerance", "investment_time_horizon",
                "financial_goal_orientation", "information_processing_style",
                "investment_fear", "investment_confidence"
            ]
            
            for score_key in required_scores:
                if score_key not in scores:
                    logger.warning(f"Missing score: {score_key}")
                    return False
                
                score_value = scores[score_key]
                if not isinstance(score_value, (int, float)) or not (0 <= score_value <= 100):
                    logger.warning(f"Invalid score value for {score_key}: {score_value}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Profile validation failed: {e}")
            return False 