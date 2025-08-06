#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
향상된 뉴스 수집기
- naver-search MCP와 RSS 피드 통합
- 중복 제거 및 품질 필터링
- 키워드 기반 관련성 점수
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
import re
import feedparser
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedNewsCollector")

# MCP 클라이언트 임포트
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    logger.warning("MCP 라이브러리가 설치되지 않았습니다. RSS만 사용합니다.")
    MCP_AVAILABLE = False

class EnhancedNewsCollector:
    """향상된 뉴스 수집기"""
    
    def __init__(self):
        self.raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # 확장된 금융 관련 검색 키워드
        self.search_keywords = {
            # 주요 지수
            "indices": ["코스피", "코스닥", "다우존스", "나스닥", "S&P500"],
            # 주요 종목
            "major_stocks": ["삼성전자", "SK하이닉스", "LG에너지솔루션", "현대차", "카카오", "네이버"],
            # 시장 동향
            "market": ["증시", "주식시장", "증권", "투자", "상장", "공모주"],
            # 경제 지표
            "economy": ["금리", "환율", "인플레이션", "경제성장", "실업률", "GDP"],
            # 산업/섹터
            "sectors": ["반도체", "배터리", "바이오", "IT", "금융", "부동산"],
            # 글로벌
            "global": ["미국증시", "중국증시", "일본증시", "유럽증시", "연준", "FOMC"]
        }
        
        # RSS 피드 소스
        self.rss_feeds = [
            # 국내 경제/증권 뉴스
            {'url': 'https://www.sedaily.com/RSS/Stock.xml', 'source': '서울경제'},
            {'url': 'https://www.mk.co.kr/rss/40300001/', 'source': '매일경제'},
            {'url': 'https://rss.hankyung.com/feed/finance.xml', 'source': '한국경제'},
            {'url': 'http://rss.edaily.co.kr/stock_news.xml', 'source': '이데일리'},
            {'url': 'https://www.fnnews.com/rss/r20/fn_realnews_stock.xml', 'source': '파이낸셜뉴스'},
            # 글로벌 경제
            {'url': 'https://feeds.bloomberg.com/markets/news.rss', 'source': 'Bloomberg'},
            {'url': 'https://feeds.finance.yahoo.com/rss/2.0/headline', 'source': 'Yahoo Finance'}
        ]
    
    async def search_news_with_mcp(self, query: str, display: int = 30) -> List[Dict]:
        """MCP를 통해 네이버 뉴스 검색"""
        if not MCP_AVAILABLE:
            return []
            
        try:
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@iamgabrielooo/naver-search-mcp"],
                env=None
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.call_tool(
                        "search-news",
                        arguments={
                            "query": query,
                            "display": display,
                            "sort": "date"  # 최신순
                        }
                    )
                    
                    if result and hasattr(result, 'content'):
                        items = json.loads(result.content[0].text)
                        return items if isinstance(items, list) else []
                    return []
                    
        except Exception as e:
            logger.error(f"MCP 검색 오류 ({query}): {e}")
            return []
    
    async def collect_mcp_news(self, days: int = 1) -> List[Dict]:
        """MCP로 뉴스 수집"""
        all_news = []
        
        # 카테고리별로 순차 검색
        for category, keywords in self.search_keywords.items():
            for keyword in keywords:
                logger.info(f"MCP 검색 중: {keyword} (카테고리: {category})")
                
                try:
                    news_items = await self.search_news_with_mcp(keyword, display=30)
                    
                    for item in news_items:
                        # 날짜 파싱
                        pub_date = self.parse_date(item.get('pubDate', ''))
                        if not pub_date:
                            continue
                            
                        # 기간 필터링
                        if (datetime.now() - pub_date).days > days:
                            continue
                        
                        news_data = {
                            'title': self.clean_html(item.get('title', '')),
                            'summary': self.clean_html(item.get('description', ''))[:500],
                            'link': item.get('link', ''),
                            'published': pub_date.isoformat(),
                            'source': '네이버뉴스',
                            'source_url': 'naver.com',
                            'is_korean': True,
                            'category': category,
                            'keyword': keyword,
                            'collect_method': 'MCP'
                        }
                        all_news.append(news_data)
                        
                except Exception as e:
                    logger.error(f"뉴스 항목 처리 오류: {e}")
                    continue
                
                # API 호출 간격
                await asyncio.sleep(0.2)
        
        return all_news
    
    def collect_rss_news(self, days: int = 1) -> List[Dict]:
        """RSS 피드로 뉴스 수집"""
        all_news = []
        
        for feed_info in self.rss_feeds:
            try:
                logger.info(f"RSS 수집 중: {feed_info['source']}")
                feed = feedparser.parse(feed_info['url'])
                
                for entry in feed.entries[:50]:  # 최대 50개
                    # 날짜 파싱
                    pub_date = None
                    if hasattr(entry, 'published_parsed'):
                        pub_date = datetime.fromtimestamp(entry.published_parsed.timestamp())
                    elif hasattr(entry, 'updated_parsed'):
                        pub_date = datetime.fromtimestamp(entry.updated_parsed.timestamp())
                    else:
                        continue
                    
                    # 기간 필터링
                    if (datetime.now() - pub_date).days > days:
                        continue
                    
                    # 카테고리 매칭
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    category = self.match_category(title + ' ' + summary)
                    
                    news_data = {
                        'title': self.clean_html(title),
                        'summary': self.clean_html(summary)[:500],
                        'link': entry.get('link', ''),
                        'published': pub_date.isoformat(),
                        'source': feed_info['source'],
                        'source_url': feed_info['url'],
                        'is_korean': 'korea' not in feed_info['source'].lower(),
                        'category': category,
                        'keyword': '',
                        'collect_method': 'RSS'
                    }
                    all_news.append(news_data)
                    
            except Exception as e:
                logger.error(f"RSS 피드 오류 ({feed_info['source']}): {e}")
                continue
        
        return all_news
    
    def match_category(self, text: str) -> str:
        """텍스트에서 카테고리 매칭"""
        text = text.lower()
        
        for category, keywords in self.search_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    return category
        
        return 'general'
    
    def calculate_relevance_score(self, news: Dict) -> float:
        """뉴스 관련성 점수 계산"""
        score = 0.0
        text = (news.get('title', '') + ' ' + news.get('summary', '')).lower()
        
        # 카테고리별 가중치
        category_weights = {
            'major_stocks': 1.0,
            'indices': 0.9,
            'market': 0.8,
            'economy': 0.7,
            'sectors': 0.8,
            'global': 0.6,
            'general': 0.3
        }
        
        # 카테고리 점수
        score += category_weights.get(news.get('category', 'general'), 0.3)
        
        # 키워드 매칭 점수
        keyword_count = 0
        for keywords in self.search_keywords.values():
            for keyword in keywords:
                if keyword.lower() in text:
                    keyword_count += 1
        
        score += min(keyword_count * 0.1, 0.5)
        
        # 최신성 점수
        try:
            pub_date = datetime.fromisoformat(news.get('published', ''))
            hours_ago = (datetime.now() - pub_date).total_seconds() / 3600
            if hours_ago < 6:
                score += 0.3
            elif hours_ago < 24:
                score += 0.2
            elif hours_ago < 48:
                score += 0.1
        except:
            pass
        
        return min(score, 2.0)
    
    def deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """중복 뉴스 제거"""
        seen_titles = set()
        unique_news = []
        
        # 관련성 점수로 정렬
        news_list.sort(key=lambda x: self.calculate_relevance_score(x), reverse=True)
        
        for news in news_list:
            # 제목 정규화
            normalized_title = re.sub(r'[^\w\s]', '', news['title'].lower())
            normalized_title = ' '.join(normalized_title.split())
            
            # 유사도 체크
            is_duplicate = False
            for seen in seen_titles:
                if self.calculate_similarity(normalized_title, seen) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_titles.add(normalized_title)
                news['relevance_score'] = self.calculate_relevance_score(news)
                unique_news.append(news)
        
        return unique_news
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """간단한 텍스트 유사도 계산"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """다양한 날짜 형식 파싱"""
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.replace('+0900', '+09:00'), fmt).replace(tzinfo=None)
            except:
                continue
        
        return datetime.now()
    
    def clean_html(self, text: str) -> str:
        """HTML 태그 및 특수문자 제거"""
        # HTML 태그 제거
        text = re.sub('<.*?>', '', text)
        
        # HTML 엔티티 변환
        replacements = {
            '&quot;': '"',
            '&nbsp;': ' ',
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&#39;': "'",
            '&ldquo;': '"',
            '&rdquo;': '"'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # 여러 공백을 하나로
        text = ' '.join(text.split())
        
        return text.strip()
    
    async def collect_all_news_async(self, days: int = 1) -> pd.DataFrame:
        """비동기로 모든 뉴스 수집"""
        all_news = []
        
        # MCP 뉴스 수집
        if MCP_AVAILABLE:
            logger.info("MCP를 통한 뉴스 수집 시작...")
            mcp_news = await self.collect_mcp_news(days)
            all_news.extend(mcp_news)
            logger.info(f"MCP 뉴스 수집 완료: {len(mcp_news)}개")
        
        # RSS 뉴스 수집
        logger.info("RSS 피드 뉴스 수집 시작...")
        rss_news = self.collect_rss_news(days)
        all_news.extend(rss_news)
        logger.info(f"RSS 뉴스 수집 완료: {len(rss_news)}개")
        
        # 중복 제거 및 정렬
        logger.info("중복 제거 및 정렬 중...")
        unique_news = self.deduplicate_news(all_news)
        
        # 데이터프레임 변환
        if unique_news:
            df = pd.DataFrame(unique_news)
            df = df.sort_values(['relevance_score', 'published'], ascending=[False, False])
            return df
        
        return pd.DataFrame()
    
    def collect_news(self, days: int = 1) -> Optional[pd.DataFrame]:
        """동기 방식으로 뉴스 수집 (메인 인터페이스)"""
        try:
            # 비동기 함수를 동기적으로 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            df = loop.run_until_complete(self.collect_all_news_async(days))
            loop.close()
            
            if not df.empty:
                # 파일 저장
                today = datetime.now().strftime('%Y%m%d')
                output_file = os.path.join(self.raw_data_dir, f"news_{today}.csv")
                
                # 필요한 컬럼만 저장
                save_columns = ['title', 'summary', 'link', 'published', 'source', 
                               'source_url', 'is_korean', 'category', 'keyword', 
                               'relevance_score']
                df_save = df[save_columns]
                df_save.to_csv(output_file, index=False, encoding='utf-8')
                
                logger.info(f"뉴스 수집 완료: 총 {len(df)}개 기사")
                logger.info(f"저장 경로: {output_file}")
                
                # 카테고리별 통계
                category_stats = df['category'].value_counts()
                logger.info("카테고리별 뉴스 수:")
                for cat, count in category_stats.items():
                    logger.info(f"  - {cat}: {count}개")
                
                return df
            else:
                logger.warning("수집된 뉴스가 없습니다.")
                return None
                
        except Exception as e:
            logger.error(f"뉴스 수집 실패: {e}", exc_info=True)
            return None


def main():
    """메인 함수"""
    collector = EnhancedNewsCollector()
    
    # 뉴스 수집 (최근 1일)
    logger.info("향상된 뉴스 수집 시작...")
    news_df = collector.collect_news(days=1)
    
    if news_df is not None and not news_df.empty:
        # 상위 10개 뉴스 출력
        print("\n[ 오늘의 주요 뉴스 TOP 10 ]")
        print("-" * 80)
        
        for idx, row in news_df.head(10).iterrows():
            print(f"\n{idx+1}. [{row['category']}] {row['title']}")
            print(f"   출처: {row['source']} | 관련성: {row['relevance_score']:.2f}")
            print(f"   {row['summary'][:100]}...")
    else:
        print("수집된 뉴스가 없습니다.")


if __name__ == "__main__":
    main()