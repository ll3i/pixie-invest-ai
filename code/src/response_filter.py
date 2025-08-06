#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI 응답 필터링 모듈
실제 데이터만 사용하도록 강제
"""

import re
import logging

logger = logging.getLogger(__name__)

# 금지된 종목명과 가격
FORBIDDEN_STOCKS = {
    "SK텔레콤": ["200,000", "200000", "190,000", "210,000"],
    "KT&G": ["80,000", "80000", "75,000", "85,000"],
    "현대차": ["200,000", "200000", "190,000", "210,000"],
    "기아": ["80,000", "80000", "75,000", "85,000"],
    "LG전자": ["100,000", "100000", "95,000", "105,000"],
    "포스코": ["300,000", "300000", "290,000", "310,000"],
    "KB금융": ["50,000", "50000", "45,000", "55,000"],
    "신한금융": ["40,000", "40000", "35,000", "45,000"],
    "삼성전자": ["70,000", "70000", "69,000", "71,000"],
    "SK하이닉스": ["100,000", "100000", "95,000", "105,000"],
}

def filter_ai_response(response: str, actual_stock_data: str) -> str:
    """
    AI 응답에서 금지된 종목과 가격을 제거하고 실제 데이터로 교체
    
    Args:
        response: AI-B의 원본 응답
        actual_stock_data: 실제 주식 데이터 컨텍스트
        
    Returns:
        필터링된 응답
    """
    try:
        # 금지된 종목이 언급되었는지 확인
        forbidden_found = []
        for stock, prices in FORBIDDEN_STOCKS.items():
            if stock in response:
                for price in prices:
                    if price in response:
                        forbidden_found.append((stock, price))
                        logger.warning(f"Forbidden stock found: {stock} with price {price}")
        
        if forbidden_found:
            logger.warning(f"AI-B used forbidden stocks: {forbidden_found}")
            
            # 실제 데이터에서 종목 정보 추출
            if "[주식 추천 결과]" in actual_stock_data:
                # 실제 데이터만 추출하여 반환
                start_idx = actual_stock_data.find("[주식 추천 결과]")
                end_idx = actual_stock_data.find("\n\n", start_idx + 100)
                if end_idx == -1:
                    end_idx = len(actual_stock_data)
                
                real_data = actual_stock_data[start_idx:end_idx]
                
                # 응답을 실제 데이터로 교체
                filtered_response = f"""제공된 데이터베이스를 기반으로 답변드립니다.

{real_data}

위 종목들은 실제 평가 데이터를 기반으로 선정된 종목들입니다.
각 종목의 현재가, PER, PBR 등은 최신 데이터베이스 기준입니다."""
                
                return filtered_response
        
        # 금지된 종목이 없으면 원본 반환
        return response
        
    except Exception as e:
        logger.error(f"Response filtering failed: {e}")
        return response

def validate_stock_data(response: str, context: str) -> bool:
    """
    응답이 실제 컨텍스트의 데이터를 사용했는지 검증
    
    Args:
        response: AI 응답
        context: 제공된 컨텍스트
        
    Returns:
        검증 성공 여부
    """
    try:
        # 컨텍스트에서 실제 종목 추출
        real_stocks = []
        if "피에이치씨" in context:
            real_stocks.append("피에이치씨")
        if "동화약품" in context:
            real_stocks.append("동화약품")
        if "셀트리온" in context:
            real_stocks.append("셀트리온")
        
        # 응답에 실제 종목이 포함되어 있는지 확인
        used_real_data = any(stock in response for stock in real_stocks)
        
        # 금지된 종목이 포함되어 있는지 확인
        used_forbidden = any(stock in response for stock in FORBIDDEN_STOCKS.keys())
        
        return used_real_data and not used_forbidden
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False