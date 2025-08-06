"""
네이버 뉴스 수집 및 감정 분석 서비스
naver-search-mcp를 활용한 실시간 뉴스 수집 및 분석
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import os
from collections import Counter

# MCP import (설치 필요: pip install mcp)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    NAVER_MCP_AVAILABLE = True
except ImportError:
    NAVER_MCP_AVAILABLE = False
    print("Warning: naver-search-mcp not available. Install with: pip install mcp")

logger = logging.getLogger(__name__)

class NewsCollectorService:
    """네이버 뉴스 수집 및 감정 분석 서비스"""
    
    def __init__(self):
        self.mcp_client = None
        self.session = None
        
        # 감정 분석용 키워드
        self.positive_words = [
            "상승", "급등", "호재", "신고가", "흑자", "개선", "회복", "성장", 
            "돌파", "강세", "낙관", "긍정", "호조", "증가", "확대", "상향"
        ]
        
        self.negative_words = [
            "하락", "급락", "악재", "신저가", "적자", "악화", "위축", "감소",
            "붕괴", "약세", "비관", "부정", "부진", "축소", "하향", "우려"
        ]
        
        # 트렌드 키워드 추출용
        self.trend_keywords = []
        
    async def initialize_mcp(self):
        """MCP 클라이언트 초기화"""
        if not NAVER_MCP_AVAILABLE:
            logger.error("naver-search-mcp not available")
            return False
            
        try:
            # naver-search-mcp 서버 실행
            server_params = StdioServerParameters(
                command="npx",
                args=["naver-search-mcp"]
            )
            
            self.mcp_client = stdio_client(server_params)
            self.session = await self.mcp_client.__aenter__()
            
            logger.info("Successfully initialized naver-search-mcp")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
            return False
    
    async def search_news(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """네이버 뉴스 검색"""
        if not self.session:
            logger.error("MCP session not initialized")
            return []
            
        try:
            # MCP 도구 호출
            result = await self.session.call_tool(
                "naver-search-news",
                arguments={
                    "query": query,
                    "display": max_results,
                    "sort": "date"  # 최신순
                }
            )
            
            # 결과 파싱
            news_items = []
            if result and "items" in result:
                for item in result["items"]:
                    news_items.append({
                        "title": self._clean_html(item.get("title", "")),
                        "content": self._clean_html(item.get("description", "")),
                        "link": item.get("link", ""),
                        "pub_date": item.get("pubDate", ""),
                        "source": item.get("source", ""),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Failed to search news: {e}")
            return []
    
    def _clean_html(self, text: str) -> str:
        """HTML 태그 제거"""
        text = re.sub('<.*?>', '', text)
        text = text.replace('&quot;', '"').replace('&amp;', '&')
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        return text.strip()
    
    def analyze_sentiment(self, news: Dict[str, Any]) -> Dict[str, Any]:
        """뉴스 감정 분석"""
        text = f"{news.get('title', '')} {news.get('content', '')}"
        
        # 긍정/부정 단어 카운트
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        
        # 감정 점수 계산 (0~1 사이)
        if positive_count + negative_count == 0:
            sentiment_score = 0.5
        else:
            sentiment_score = positive_count / (positive_count + negative_count)
        
        # 감정 레이블
        if sentiment_score > 0.7:
            sentiment = "긍정"
        elif sentiment_score < 0.3:
            sentiment = "부정"
        else:
            sentiment = "중립"
        
        news["sentiment_score"] = sentiment_score
        news["sentiment"] = sentiment
        news["positive_count"] = positive_count
        news["negative_count"] = negative_count
        
        return news
    
    async def collect_daily_news(self, keywords: List[str]) -> Dict[str, Any]:
        """일일 뉴스 수집 및 분석"""
        all_news = []
        
        for keyword in keywords:
            logger.info(f"Collecting news for keyword: {keyword}")
            news_items = await self.search_news(keyword, max_results=20)
            
            # 감정 분석 적용
            for news in news_items:
                news["keyword"] = keyword
                self.analyze_sentiment(news)
            
            all_news.extend(news_items)
        
        # 중복 제거 (제목 기준)
        unique_news = {}
        for news in all_news:
            title = news["title"]
            if title not in unique_news:
                unique_news[title] = news
        
        news_list = list(unique_news.values())
        
        # 감정 분석 요약
        sentiment_summary = self._calculate_sentiment_summary(news_list)
        
        # 트렌드 키워드 추출
        trend_keywords = self._extract_trend_keywords(news_list)
        
        # 카테고리별 분류
        categorized_news = self._categorize_news(news_list)
        
        return {
            "news_list": news_list,
            "sentiment_summary": sentiment_summary,
            "trend_keywords": trend_keywords,
            "categorized_news": categorized_news,
            "collection_time": datetime.now().isoformat()
        }
    
    def _calculate_sentiment_summary(self, news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """감정 분석 요약 통계"""
        if not news_list:
            return {
                "total_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "average_score": 0.5,
                "overall_sentiment": "중립"
            }
        
        positive_count = sum(1 for news in news_list if news["sentiment"] == "긍정")
        negative_count = sum(1 for news in news_list if news["sentiment"] == "부정")
        neutral_count = sum(1 for news in news_list if news["sentiment"] == "중립")
        
        average_score = sum(news["sentiment_score"] for news in news_list) / len(news_list)
        
        if average_score > 0.6:
            overall_sentiment = "긍정적"
        elif average_score < 0.4:
            overall_sentiment = "부정적"
        else:
            overall_sentiment = "중립적"
        
        return {
            "total_count": len(news_list),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "average_score": round(average_score, 3),
            "overall_sentiment": overall_sentiment
        }
    
    def _extract_trend_keywords(self, news_list: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """트렌드 키워드 추출"""
        # 모든 뉴스에서 명사 추출 (간단한 방법)
        all_words = []
        
        for news in news_list:
            text = f"{news.get('title', '')} {news.get('content', '')}"
            # 2글자 이상의 한글 단어 추출
            words = re.findall(r'[가-힣]{2,}', text)
            all_words.extend(words)
        
        # 불용어 제거
        stopwords = ["있다", "있는", "없다", "하는", "한다", "그리고", "하지만", "위해", "대한", "통해", "따라"]
        filtered_words = [word for word in all_words if word not in stopwords]
        
        # 빈도수 계산
        word_counts = Counter(filtered_words)
        
        # 상위 키워드 추출
        trend_keywords = []
        for word, count in word_counts.most_common(top_k):
            # 감정 가중치 계산 (긍정/부정 키워드는 가중치 부여)
            weight = 1.0
            if word in self.positive_words or word in self.negative_words:
                weight = 1.5
            
            trend_keywords.append({
                "keyword": word,
                "count": count,
                "weight": weight,
                "score": count * weight
            })
        
        # 점수 기준으로 정렬
        trend_keywords.sort(key=lambda x: x["score"], reverse=True)
        
        return trend_keywords[:top_k]
    
    def _categorize_news(self, news_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """뉴스 카테고리 분류"""
        categories = {
            "today_pick": [],      # 오늘의 픽
            "portfolio": [],       # 보유 종목 관련
            "popular": [],         # 조회수 Top (감정 점수 높은 순)
            "recommend": []        # 추천 많은 (긍정적인 뉴스)
        }
        
        # 오늘의 픽: 최신 + 긍정적인 뉴스
        today_news = [news for news in news_list if self._is_today(news["timestamp"])]
        positive_today = [news for news in today_news if news["sentiment_score"] > 0.6]
        categories["today_pick"] = sorted(positive_today, key=lambda x: x["sentiment_score"], reverse=True)[:4]
        
        # 인기 뉴스: 전체 뉴스 중 감정 점수 높은 순
        categories["popular"] = sorted(news_list, key=lambda x: x["sentiment_score"], reverse=True)[:4]
        
        # 추천 뉴스: 긍정적인 뉴스
        categories["recommend"] = [news for news in news_list if news["sentiment"] == "긍정"][:4]
        
        return categories
    
    def _is_today(self, timestamp: str) -> bool:
        """오늘 날짜인지 확인"""
        try:
            news_date = datetime.fromisoformat(timestamp).date()
            return news_date == datetime.now().date()
        except:
            return False
    
    def generate_pixie_insights(self, news_data: Dict[str, Any]) -> Dict[str, str]:
        """픽시의 인사이트 생성"""
        sentiment_summary = news_data["sentiment_summary"]
        trend_keywords = news_data["trend_keywords"][:5]  # 상위 5개
        
        # 픽시의 인사이트
        if sentiment_summary["overall_sentiment"] == "긍정적":
            insight_title = "기회의 시장, 모멘텀을 잡아라"
            insight_content = f"오늘 시장은 {sentiment_summary['positive_count']}개의 긍정적 뉴스로 활기를 띠고 있어요. " \
                            f"특히 '{trend_keywords[0]['keyword']}'와 '{trend_keywords[1]['keyword']}' 관련 " \
                            f"종목들이 주목받고 있죠. 긍정적인 시장 분위기를 활용한 투자 전략을 고려해보세요."
        elif sentiment_summary["overall_sentiment"] == "부정적":
            insight_title = "신중한 접근, 리스크 관리가 핵심"
            insight_content = f"시장에 {sentiment_summary['negative_count']}개의 부정적 신호가 감지되고 있어요. " \
                            f"'{trend_keywords[0]['keyword']}' 이슈가 시장 불안을 키우고 있습니다. " \
                            f"포트폴리오 방어 전략과 현금 비중 확대를 고려해보세요."
        else:
            insight_title = "관망세 시장, 선별적 접근이 답"
            insight_content = f"시장이 방향성을 찾지 못하고 있어요. 긍정 {sentiment_summary['positive_count']}개, " \
                            f"부정 {sentiment_summary['negative_count']}개로 혼재된 상황입니다. " \
                            f"개별 종목의 펀더멘털에 집중한 선별 투자가 유효할 것 같아요."
        
        # 픽시의 한마디
        if sentiment_summary["average_score"] > 0.7:
            pixie_quote = "상승장에선 대담하게,\n하지만 원칙은 지켜라!"
        elif sentiment_summary["average_score"] < 0.3:
            pixie_quote = "하락장은 기회의 시작,\n준비된 자만이 웃는다!"
        else:
            pixie_quote = "시장이 조용할 때,\n미래를 준비하라!"
        
        return {
            "insight_title": insight_title,
            "insight_content": insight_content,
            "pixie_quote": pixie_quote,
            "trend_summary": f"오늘의 핵심 키워드: {', '.join([kw['keyword'] for kw in trend_keywords[:3]])}"
        }
    
    async def close(self):
        """MCP 클라이언트 종료"""
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)


# 사용 예시
async def main():
    collector = NewsCollectorService()
    
    # MCP 초기화
    if await collector.initialize_mcp():
        # 주요 키워드로 뉴스 수집
        keywords = ["코스피", "삼성전자", "SK하이닉스", "금리", "환율", "AI", "반도체"]
        news_data = await collector.collect_daily_news(keywords)
        
        # 픽시 인사이트 생성
        insights = collector.generate_pixie_insights(news_data)
        
        # 결과 출력
        print(f"수집된 뉴스: {len(news_data['news_list'])}개")
        print(f"감정 분석: {news_data['sentiment_summary']}")
        print(f"트렌드 키워드: {[kw['keyword'] for kw in news_data['trend_keywords'][:10]]}")
        print(f"픽시의 인사이트: {insights['insight_title']}")
        print(f"픽시의 한마디: {insights['pixie_quote']}")
        
        await collector.close()

if __name__ == "__main__":
    asyncio.run(main())