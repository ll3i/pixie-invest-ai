#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MINERVA 투자 챗봇 시스템 메인 모듈
- 서비스 알고리즘 흐름도에 따라 구현된 전체 시스템 통합
- 모든 모듈 초기화 및 실행
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
import time # Added for daily collection

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"minerva_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MINERVA")

# 환경 변수 로드
load_dotenv()

# src 디렉토리를 Python 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# 모듈 임포트
try:
    from user_profile_analyzer import UserProfileAnalyzer
    from financial_data_processor import FinancialDataProcessor
    from llm_service import LLMService
    from memory_manager import MemoryManager
    from investment_advisor import InvestmentAdvisor
    from data_collector import DataCollector
    from data_processor import DataProcessor
    from data_update_scheduler import DataUpdateScheduler
    from api_service import APIService
    from prompt_manager import PromptManager
    
    modules_imported = True
    logger.info("모든 모듈 임포트 성공")
except ImportError as e:
    modules_imported = False
    logger.error(f"모듈 임포트 오류: {e}")

class MinervaSystem:
    """MINERVA 투자 챗봇 시스템 클래스"""
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)
        
        # API 유형 결정
        self.api_type = "openai" if os.environ.get('OPENAI_API_KEY') else "clova"
        if not os.environ.get('OPENAI_API_KEY') and not os.environ.get('CLOVA_API_KEY'):
            logger.warning("API 키가 설정되지 않았습니다. 시뮬레이션 모드로 실행합니다.")
            self.api_type = "simulation"
        
        # 모듈 초기화
        self.initialize_modules()
        
        logger.info(f"MINERVA 시스템 초기화 완료 (API 유형: {self.api_type})")
    
    def initialize_modules(self):
        """모든 모듈 초기화"""
        if not modules_imported:
            logger.error("모듈 임포트 실패로 시스템을 초기화할 수 없습니다.")
            return False
        
        try:
            # 프롬프트 관리자 초기화
            self.prompt_manager = PromptManager()
            logger.info("프롬프트 관리자 초기화 완료")
            
            # API 서비스 초기화
            self.api_service = APIService(api_type=self.api_type)
            logger.info("API 서비스 초기화 완료")
            
            # 메모리 관리자 초기화
            self.memory_manager = MemoryManager()
            logger.info("메모리 관리자 초기화 완료")
            
            # 사용자 프로필 분석기 초기화
            self.profile_analyzer = UserProfileAnalyzer()
            logger.info("사용자 프로필 분석기 초기화 완료")
            
            # 금융 데이터 프로세서 초기화
            self.financial_processor = FinancialDataProcessor()
            logger.info("금융 데이터 프로세서 초기화 완료")
            
            # LLM 서비스 초기화
            self.llm_service = LLMService(api_type=self.api_type)
            logger.info("LLM 서비스 초기화 완료")
            
            # 투자 어드바이저 초기화
            self.investment_advisor = InvestmentAdvisor(api_type=self.api_type)
            logger.info("투자 어드바이저 초기화 완료")
            
            # 데이터 수집기 초기화
            self.data_collector = DataCollector()
            logger.info("데이터 수집기 초기화 완료")
            
            # 데이터 프로세서 초기화
            self.data_processor = DataProcessor()
            logger.info("데이터 프로세서 초기화 완료")
            
            # 데이터 업데이트 스케줄러 초기화
            self.data_update_scheduler = DataUpdateScheduler()
            logger.info("데이터 업데이트 스케줄러 초기화 완료")
            
            return True
        except Exception as e:
            logger.error(f"모듈 초기화 오류: {e}")
            return False
    
    def start_data_update_scheduler(self):
        """데이터 업데이트 스케줄러 시작"""
        if hasattr(self, 'data_update_scheduler'):
            return self.data_update_scheduler.start()
        return False
    
    def stop_data_update_scheduler(self):
        """데이터 업데이트 스케줄러 중지"""
        if hasattr(self, 'data_update_scheduler'):
            return self.data_update_scheduler.stop()
        return False
    
    def run_immediate_data_update(self, update_type='all'):
        """즉시 데이터 업데이트 실행"""
        if hasattr(self, 'data_update_scheduler'):
            return self.data_update_scheduler.run_immediate_update(update_type)
        return False
    
    def test_api_connection(self):
        """API 연결 테스트"""
        if hasattr(self, 'api_service'):
            return self.api_service.test_api_connection()
        return False, "API 서비스가 초기화되지 않았습니다."
    
    def get_system_status(self):
        """시스템 상태 정보 반환"""
        status = {
            "api_type": self.api_type,
            "modules": {
                "prompt_manager": hasattr(self, 'prompt_manager'),
                "api_service": hasattr(self, 'api_service'),
                "memory_manager": hasattr(self, 'memory_manager'),
                "profile_analyzer": hasattr(self, 'profile_analyzer'),
                "financial_processor": hasattr(self, 'financial_processor'),
                "llm_service": hasattr(self, 'llm_service'),
                "investment_advisor": hasattr(self, 'investment_advisor'),
                "data_collector": hasattr(self, 'data_collector'),
                "data_processor": hasattr(self, 'data_processor'),
                "data_update_scheduler": hasattr(self, 'data_update_scheduler')
            },
            "timestamp": datetime.now().isoformat()
        }
        return status

    def get_data_status(self):
        """데이터 파일 상태 정보 반환"""
        data_status = {}
        data_dir = os.path.join(self.base_dir, "data")
        if os.path.exists(data_dir):
            for item_name in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item_name)
                if os.path.isdir(item_path):
                    data_type = item_name
                    if data_type == "prices":
                        price_files = [f for f in os.listdir(item_path) if f.endswith(".csv")]
                        data_status[data_type] = {"count": len(price_files), "latest": max(price_files) if price_files else "N/A"}
                    elif data_type == "news":
                        news_files = [f for f in os.listdir(item_path) if f.endswith(".json")]
                        data_status[data_type] = {"count": len(news_files), "latest": max(news_files) if news_files else "N/A"}
                    elif data_type == "financials":
                        financial_files = [f for f in os.listdir(item_path) if f.endswith(".csv")]
                        data_status[data_type] = {"count": len(financial_files), "latest": max(financial_files) if financial_files else "N/A"}
                    elif data_type == "valuations":
                        valuation_files = [f for f in os.listdir(item_path) if f.endswith(".csv")]
                        data_status[data_type] = {"count": len(valuation_files), "latest": max(valuation_files) if valuation_files else "N/A"}
                    elif data_type == "tickers":
                        ticker_files = [f for f in os.listdir(item_path) if f.endswith(".csv")]
                        data_status[data_type] = {"count": len(ticker_files), "latest": max(ticker_files) if ticker_files else "N/A"}
                    elif data_type == "sectors":
                        sector_files = [f for f in os.listdir(item_path) if f.endswith(".csv")]
                        data_status[data_type] = {"count": len(sector_files), "latest": max(sector_files) if sector_files else "N/A"}
                    elif data_type == "vector_db":
                        vector_files = [f for f in os.listdir(item_path) if f.endswith(".pkl")]
                        data_status[data_type] = {"count": len(vector_files), "latest": max(vector_files) if vector_files else "N/A"}
                    elif data_type == "historical_setup":
                        historical_files = [f for f in os.listdir(item_path) if f.endswith(".pkl")]
                        data_status[data_type] = {"count": len(historical_files), "latest": max(historical_files) if historical_files else "N/A"}
        return data_status

    def run_initial_data_setup(self):
        """최초 3년간 데이터 수집 및 일일 자동 업데이트 스케줄러 설정"""
        if not self.initialize_modules():
            return False
        
        # 데이터 수집기 실행
        logger.info("최초 3년간 데이터 수집 시작")
        success = self.data_collector.run_initial_data_setup()
        if not success:
            logger.error("최초 3년간 데이터 수집 실패")
            return False
        logger.info("최초 3년간 데이터 수집 완료")

        # 데이터 업데이트 스케줄러 설정
        logger.info("일일 데이터 업데이트 스케줄러 설정 시작")
        self.data_update_scheduler.schedule_jobs()
        self.data_update_scheduler.start()
        logger.info("일일 데이터 업데이트 스케줄러 설정 완료")

        return True

    def run_daily_data_collection(self):
        """일일 데이터 수집 실행"""
        if not self.initialize_modules():
            return False
        
        logger.info("일일 데이터 수집 시작")
        success = self.data_collector.run_daily_update()
        if not success:
            logger.error("일일 데이터 수집 실패")
            return False
        logger.info("일일 데이터 수집 완료")

        return True

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='MINERVA 투자 챗봇 시스템')
    parser.add_argument('--update-data', choices=['all', 'prices', 'news', 'financials', 'valuations', 'tickers', 'sectors', 'vector_db', 'historical_setup'],
                        help='즉시 데이터 업데이트 실행')
    parser.add_argument('--start-scheduler', action='store_true',
                        help='데이터 업데이트 스케줄러 시작')
    parser.add_argument('--test-api', action='store_true',
                        help='API 연결 테스트')
    parser.add_argument('--status', action='store_true',
                        help='시스템 상태 정보 출력')
    parser.add_argument('--initial-setup', action='store_true',
                        help='최초 3년간 데이터 수집 실행')
    parser.add_argument('--collect-daily', action='store_true',
                        help='일일 데이터 수집 실행')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_arguments()
    
    # MINERVA 시스템 초기화
    minerva = MinervaSystem()
    
    # 명령행 인수에 따라 작업 수행
    if args.initial_setup:
        logger.info("최초 3년간 데이터 수집 시작")
        success = minerva.run_initial_data_setup()
        if success:
            print("✅ 3년간 데이터 수집이 완료되었습니다.")
        else:
            print("❌ 데이터 수집 중 오류가 발생했습니다.")
    
    if args.collect_daily:
        logger.info("일일 데이터 수집 실행")
        success = minerva.run_daily_data_collection()
        if success:
            print("✅ 일일 데이터 수집이 완료되었습니다.")
        else:
            print("❌ 일일 데이터 수집 중 오류가 발생했습니다.")
    
    if args.update_data:
        logger.info(f"데이터 업데이트 실행: {args.update_data}")
        minerva.run_immediate_data_update(args.update_data)
    
    if args.start_scheduler:
        logger.info("데이터 업데이트 스케줄러 시작")
        success = minerva.start_data_update_scheduler()
        if success:
            print("✅ 자동 업데이트 스케줄러가 시작되었습니다.")
            print("📅 매일 오전 6시: 전날 주가 데이터 수집")
            print("📰 매일 오전 7시: 뉴스 데이터 수집")
            print("�� 평일 오전 8시: 재무지표 업데이트")
            print("🔄 매주 월요일 오전 5시: 종목 코드 업데이트")
            print("\n스케줄러가 백그라운드에서 실행됩니다. Ctrl+C로 중지할 수 있습니다.")
            
            try:
                # 메인 스레드 유지
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 스케줄러 중지 중...")
                minerva.stop_data_update_scheduler()
                print("✅ 스케줄러가 정상적으로 중지되었습니다.")
        else:
            print("❌ 스케줄러 시작에 실패했습니다.")
    
    if args.test_api:
        logger.info("API 연결 테스트")
        success, message = minerva.test_api_connection()
        print(f"API 연결 테스트 결과: {'✅ 성공' if success else '❌ 실패'} - {message}")
    
    if args.status:
        logger.info("시스템 상태 정보 출력")
        status = minerva.get_system_status()
        print(f"🤖 API 유형: {status['api_type']}")
        print("📦 모듈 상태:")
        for module, initialized in status['modules'].items():
            status_icon = "✅" if initialized else "❌"
            print(f"  {status_icon} {module}: {'초기화됨' if initialized else '초기화되지 않음'}")
        print(f"⏰ 타임스탬프: {status['timestamp']}")
        
        # 데이터 파일 상태 확인
        data_status = minerva.get_data_status()
        print("\n📊 데이터 상태:")
        for data_type, info in data_status.items():
            print(f"  📁 {data_type}: {info['count']}개 파일, 최신: {info['latest']}")
    
    # 아무 인수도 없으면 도움말 출력
    if not any(vars(args).values()):
        logger.info("도움말 출력")
        print("🤖 MINERVA 투자 챗봇 시스템")
        print("=" * 50)
        print("사용 방법: python main.py [옵션]")
        print("\n📋 옵션:")
        print("  --initial-setup       최초 3년간 데이터 수집 실행")
        print("  --collect-daily       일일 데이터 수집 실행")
        print("  --start-scheduler     자동 업데이트 스케줄러 시작")
        print("  --update-data TYPE    즉시 데이터 업데이트 실행")
        print("                        TYPE: all, prices, news, financials, valuations,")
        print("                              tickers, sectors, vector_db, historical_setup")
        print("  --test-api           API 연결 테스트")
        print("  --status             시스템 상태 정보 출력")
        print("\n🚀 빠른 시작:")
        print("  1. 최초 설정: python main.py --initial-setup")
        print("  2. 스케줄러 시작: python main.py --start-scheduler")
        print("  3. 웹 서비스 실행: cd web && python app.py")

if __name__ == "__main__":
    main()
