"""
네이버 검색 MCP를 활용한 뉴스 수집기
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio

# MCP 클라이언트 임포트
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("MCP 라이브러리가 설치되지 않았습니다. pip install mcp를 실행하세요.")
    MCP_AVAILABLE = False

class NaverNewsMCPCollector:
    """네이버 검색 MCP를 활용한 뉴스 수집기"""
    
    def __init__(self):
        self.raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # 금융 관련 검색 키워드
        self.search_keywords = [
            "증시", "주식", "코스피", "코스닥", 
            "금융", "투자", "경제", "증권",
            "삼성전자", "SK하이닉스", "LG에너지솔루션",
            "금리", "환율", "미국증시", "나스닥"
        ]
        
    async def search_news_with_mcp(self, query: str, display: int = 20) -> List[Dict]:
        """MCP를 통해 네이버 뉴스 검색"""
        if not MCP_AVAILABLE:
            print("MCP가 사용 불가능합니다.")
            return []
            
        try:
            # MCP 서버 파라미터 설정
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@iamgabrielooo/naver-search-mcp"],
                env=None
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # 서버 초기화
                    await session.initialize()
                    
                    # 뉴스 검색 도구 호출
                    result = await session.call_tool(
                        "search-news",
                        arguments={
                            "query": query,
                            "display": display,
                            "sort": "date"  # 최신순 정렬
                        }
                    )
                    
                    # 결과 파싱
                    if result and hasattr(result, 'content'):
                        return json.loads(result.content[0].text)
                    return []
                    
        except Exception as e:
            print(f"MCP 검색 오류 ({query}): {e}")
            return []
    
    async def collect_all_news(self, days: int = 1) -> pd.DataFrame:
        """모든 키워드에 대해 뉴스 수집"""
        all_news = []
        
        for keyword in self.search_keywords:
            print(f"'{keyword}' 검색 중...")
            news_items = await self.search_news_with_mcp(keyword, display=20)
            
            for item in news_items:
                try:
                    # 날짜 파싱
                    pub_date = datetime.now()
                    if 'pubDate' in item:
                        try:
                            pub_date = datetime.strptime(item['pubDate'], '%a, %d %b %Y %H:%M:%S %z').replace(tzinfo=None)
                        except:
                            pub_date = datetime.now()
                    
                    # 최근 n일 이내의 뉴스만
                    if (datetime.now() - pub_date).days <= days:
                        news_data = {
                            'title': self.clean_html(item.get('title', '')),
                            'summary': self.clean_html(item.get('description', ''))[:500],
                            'link': item.get('link', ''),
                            'published': pub_date.isoformat(),
                            'source': '네이버뉴스',
                            'source_url': 'naver.com',
                            'is_korean': True,
                            'keyword': keyword
                        }
                        all_news.append(news_data)
                        
                except Exception as e:
                    print(f"뉴스 항목 처리 오류: {e}")
                    continue
            
            # API 호출 간격
            await asyncio.sleep(0.1)
        
        # 중복 제거
        seen_titles = set()
        unique_news = []
        for news in all_news:
            if news['title'] not in seen_titles:
                seen_titles.add(news['title'])
                unique_news.append(news)
        
        # 데이터프레임 변환
        if unique_news:
            df = pd.DataFrame(unique_news)
            # 시간순 정렬
            df = df.sort_values('published', ascending=False)
            return df
        
        return pd.DataFrame()
    
    def clean_html(self, text: str) -> str:
        """HTML 태그 제거"""
        import re
        clean_text = re.sub('<.*?>', '', text)
        clean_text = clean_text.replace('&quot;', '"')
        clean_text = clean_text.replace('&nbsp;', ' ')
        clean_text = clean_text.replace('&lt;', '<')
        clean_text = clean_text.replace('&gt;', '>')
        clean_text = clean_text.replace('&amp;', '&')
        return clean_text.strip()
    
    def collect_news(self, days: int = 1) -> Optional[pd.DataFrame]:
        """동기 방식으로 뉴스 수집"""
        try:
            # 비동기 함수를 동기적으로 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            df = loop.run_until_complete(self.collect_all_news(days))
            loop.close()
            
            if not df.empty:
                # 파일 저장
                today = datetime.now().strftime('%Y%m%d')
                output_file = os.path.join(self.raw_data_dir, f"news_mcp_{today}.csv")
                df.to_csv(output_file, index=False, encoding='utf-8')
                
                print(f"네이버 뉴스 수집 완료: {len(df)}개 기사")
                print(f"저장 경로: {output_file}")
                return df
            else:
                print("수집된 뉴스가 없습니다.")
                return None
                
        except Exception as e:
            print(f"뉴스 수집 실패: {e}")
            return None
    
    def merge_with_rss_news(self, mcp_df: pd.DataFrame, rss_file: str) -> pd.DataFrame:
        """MCP 뉴스와 RSS 뉴스 병합"""
        try:
            # RSS 뉴스 로드
            rss_df = pd.read_csv(rss_file, encoding='utf-8')
            
            # 병합
            merged_df = pd.concat([mcp_df, rss_df], ignore_index=True)
            
            # 중복 제거 (제목 기준)
            merged_df = merged_df.drop_duplicates(subset=['title'], keep='first')
            
            # 시간순 정렬
            merged_df = merged_df.sort_values('published', ascending=False)
            
            return merged_df
            
        except Exception as e:
            print(f"뉴스 병합 오류: {e}")
            return mcp_df


def main():
    """메인 함수"""
    collector = NaverNewsMCPCollector()
    
    # MCP로 뉴스 수집
    print("네이버 검색 MCP를 통한 뉴스 수집 시작...")
    mcp_news = collector.collect_news(days=1)
    
    if mcp_news is not None and not mcp_news.empty:
        # 기존 RSS 뉴스와 병합
        today = datetime.now().strftime('%Y%m%d')
        rss_file = os.path.join(collector.raw_data_dir, f"news_{today}.csv")
        
        if os.path.exists(rss_file):
            print("기존 RSS 뉴스와 병합 중...")
            merged_news = collector.merge_with_rss_news(mcp_news, rss_file)
            
            # 병합된 뉴스 저장
            merged_news.to_csv(rss_file, index=False, encoding='utf-8')
            print(f"최종 뉴스 수: {len(merged_news)}개")
        else:
            # RSS 파일이 없으면 MCP 뉴스를 메인 뉴스 파일로 저장
            mcp_news.to_csv(rss_file, index=False, encoding='utf-8')
            print(f"MCP 뉴스를 메인 뉴스 파일로 저장: {len(mcp_news)}개")


if __name__ == "__main__":
    main()