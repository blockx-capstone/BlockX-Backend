"""
내부자 대규모 지분 거래(블록딜) 탐지 시스템 실행 모듈
"""
import datetime
import numpy as np
import logging
import os
import sys
import argparse
from data_extraction import get_disclosure_list
from model import BlockDealDetector
from api_keys import DEFAULT_TICKER, DEFAULT_WATCHLIST

# 로깅 설정
if not os.path.exists('logs'):
    os.makedirs('logs')

log_file = f"logs/block_deal_detector_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def verify_api_connections():
    """
    KRX 및 DART API 연결 상태 확인
    """
    import dart_fss as dart
    from pykrx import stock
    
    logger.info("KRX 및 DART API 연결 상태 확인 중...")
    
    # 1. KRX API 연결 확인
    try:
        today = datetime.datetime.now().strftime("%Y%m%d")
        df = stock.get_market_ticker_list(today)
        logger.info(f"KRX API 연결 성공: {len(df)}개 종목 정보 확인")
    except Exception as e:
        logger.error(f"KRX API 연결 실패: {e}")
        return False
    
    # 2. DART API 연결 확인
    try:
        corp_list = dart.get_corp_list()
        logger.info(f"DART API 연결 성공: {len(corp_list.corps)} 기업 정보 확인")
        return True
    except Exception as e:
        logger.error(f"DART API 연결 실패: {e}")
        return False

def run_block_deal_detection_system(ticker=DEFAULT_TICKER, start_date=None, end_date=None):
    """
    블록딜 탐지 모델을 학습하고 테스트 데이터로 평가
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
    """
    # 날짜 기본값 설정
    if start_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 모델 객체 생성
    detector = BlockDealDetector()
    
    # 모델 학습 시작
    logger.info(f"\n{ticker} 종목에 대한 블록딜 탐지 모델 학습 시작 ({start_date} ~ {end_date})...")
    
    # 저장 디렉토리 생성
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 모델 학습
    training_success = detector.train(ticker, start_date, end_date)
    
    if training_success:
        # 모델 저장
        detector.save_model(f"models/block_deal_detector_{ticker}")
        
        # 새로운 거래 데이터로 예측 테스트
        logger.info("\n새로운 내부자 거래에 대한 블록딜 예측 테스트:")
        
        # 예시 데이터 (실제로는 API에서 실시간으로 가져올 수 있음)
        # 주요 가격 및 거래량 지표 추가
        new_trade_data = {
            # 기본 거래 정보
            'reporter': 'CEO Name',
            'quantity': 50000,  
            'price': 70000,
            'stock_type': '보통주', 
            'cal_quantity': 50000,
            
            # OHLCV 데이터
            'open_price': 68000,
            'high_price': 72000,
            'low_price': 67500,
            'close_price': 69500,
            'volume': 1200000,
            'trading_value': 83475000000,
            'price_change': 1.2,
            
            # 추가 지표
            'volume_ma5': 900000,
            'volume_ma20': 850000,
            'volume_ratio': 1.33,
            'daily_return': 0.015,
            'volatility': 0.022,
            'ma5': 68500,
            'ma20': 67000,
            'ma60': 65500,
            'trading_value_ma5': 62000000000,
            
            # 투자자 데이터
            'individual_investor_buy_amount': 1500000000,
            'individual_investor_sell_amount': 1200000000,
            'institutional_investor_buy_amount': 900000000,
            'institutional_investor_sell_amount': 800000000,
            'foreign_investor_buy_amount': 700000000,
            'foreign_investor_sell_amount': 900000000,
            
            # 재무 데이터
            'revenue': 500000000000,
            'operating_profit': 50000000000, 
            'net_income': 35000000000,
            'total_assets': 2000000000000,
            'total_liabilities': 1000000000000,
            'equity': 1000000000000,
            'operating_cashflow': 70000000000,
            'investing_cashflow': -30000000000,
            'financing_cashflow': -20000000000,
            'debt_ratio': 0.5,
            'roe': 0.035,
            'operating_margin': 0.1,
            
            # 공시 데이터
            'disclosure_실적': 1,
            'disclosure_주주총회': 0,
            'disclosure_M&A': 0,
            'disclosure_투자': 1,
            'disclosure_기타': 2,
            
            # 뉴스 데이터
            'news_sentiment': 0.2,
            'news_실적': 2,
            'news_투자': 1,
            'news_인수': 0,
            'news_합병': 0,
            'news_매각': 0,
            'news_주주총회': 0,
            'news_배당': 1,
            'news_감자': 0,
            'news_증자': 0,
            
            # 시장 지표
            'PER': 15.2,
            'PBR': 1.8,
            'KOSPI_index': 2450,
            'KOSDAQ_index': 850,
            'KOSPI200_index': 320,
            'market_cap': 3500000000000,
            
            # 기타 필요한 시계열 특성
            'weekday_0': 0, 'weekday_1': 1, 'weekday_2': 0, 'weekday_3': 0, 'weekday_4': 0,
            'month_1': 0, 'month_2': 0, 'month_3': 0, 'month_4': 0, 'month_5': 1,
            'month_6': 0, 'month_7': 0, 'month_8': 0, 'month_9': 0, 'month_10': 0,
            'month_11': 0, 'month_12': 0,
            'quarter_1': 0, 'quarter_2': 1, 'quarter_3': 0, 'quarter_4': 0,
            'price_trend': 0.035,
            'volume_trend': 0.12
        }
        
        # 예측 실행
        prediction = detector.predict_block_deal(new_trade_data)
        
        if prediction:
            logger.info(f"\n예측 결과:")
            logger.info(f"블록딜 확률: {prediction['probability']:.4f} ({prediction['confidence']})")
            logger.info(f"판단: {'블록딜로 판단됨' if prediction['is_block_deal'] == 1 else '일반 거래로 판단됨'}")
        else:
            logger.error("예측 실패")
    else:
        logger.error("모델 학습 실패")

def setup_monitoring_system(watchlist=DEFAULT_WATCHLIST, model_path=None):
    """
    실시간 내부자 거래 모니터링 시스템 설정
    
    Args:
        watchlist (list): 모니터링할 종목 리스트
        model_path (str): 불러올 모델 경로
    """
    # 모델 로드
    detector = BlockDealDetector()
    
    # 모델 경로 기본값 설정
    if model_path is None:
        model_path = f"models/block_deal_detector_{DEFAULT_TICKER}"
    
    # 모델 불러오기
    if os.path.exists(model_path):
        load_success = detector.load_model(model_path)
        if not load_success:
            logger.error("모델 로드 실패, 새 모델을 학습합니다.")
            run_block_deal_detection_system(DEFAULT_TICKER)
            detector.load_model(model_path)
    else:
        logger.warning(f"모델 파일이 존재하지 않음: {model_path}, 새 모델을 학습합니다.")
        run_block_deal_detection_system(DEFAULT_TICKER)
        detector.load_model(model_path)
    
    # 현재 날짜 설정
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"\n{today} 내부자 거래 모니터링 시작...")
    
    for ticker in watchlist:
        logger.info(f"\n{ticker} 종목 모니터링 중...")
        
        # 당일 공시 데이터 확인
        disclosures = get_disclosure_list(ticker, today, today)
        
        if not disclosures.empty:
            logger.info(f"{ticker} 종목 당일 공시 {len(disclosures)}건 발견")
            
            # 각 공시에 대해 블록딜 여부 판단
            for _, disclosure in disclosures.iterrows():
                rcept_no = disclosure['rcept_no']
                
                # 이 부분에서 실제로 공시 상세 내용을 가져와서 분석해야 함
                # 현재는 임의 데이터로 테스트
                
                np.random.seed(int(rcept_no[-4:]))  # 리셉션 번호 마지막 4자리로 시드 설정하여 결과 재현성 확보
                
                # 임시 데이터 생성
                detailed_data = {
                    'reporter': 'Sample Name',
                    'quantity': np.random.randint(10000, 100000),  
                    'price': np.random.randint(50000, 80000),
                    'stock_type': '보통주', 
                    'cal_quantity': np.random.randint(10000, 100000),
                    'open_price': np.random.randint(50000, 70000),
                    'high_price': np.random.randint(70000, 80000),
                    'low_price': np.random.randint(50000, 60000),
                    'close_price': np.random.randint(60000, 75000),
                    'volume': np.random.randint(500000, 2000000),
                    'trading_value': np.random.randint(1000000000, 5000000000),
                    'price_change': np.random.uniform(-2.0, 2.0),
                    'individual_investor_buy_amount': np.random.randint(500000000, 2000000000),
                    'individual_investor_sell_amount': np.random.randint(500000000, 2000000000),
                    'institutional_investor_buy_amount': np.random.randint(300000000, 1500000000),
                    'institutional_investor_sell_amount': np.random.randint(300000000, 1500000000),
                    'foreign_investor_buy_amount': np.random.randint(200000000, 1000000000),
                    'foreign_investor_sell_amount': np.random.randint(200000000, 1000000000)
                }
                
                # 블록딜 예측
                prediction = detector.predict_block_deal(detailed_data)
                
                # 결과 출력 및 알림
                if prediction and prediction['probability'] > 0.7:  # 높은 확률로 블록딜 예상
                    logger.info("\n!!! 블록딜 의심 거래 탐지 !!!")
                    logger.info(f"종목: {ticker}")
                    logger.info(f"공시제목: {disclosure['report_nm']}")
                    logger.info(f"공시일자: {disclosure['report_dt']}")
                    logger.info(f"블록딜 확률: {prediction['probability']:.4f} ({prediction['confidence']})")
                    
                    # 실제 환경에서는 여기에 알림 기능 추가 (이메일, 메시지 등)
                    # send_alert(ticker, disclosure, prediction)
        else:
            logger.info(f"{ticker} 종목 당일 공시 없음")

def main():
    """
    메인 함수 - 명령행 인수 처리 및 시스템 실행
    """
    parser = argparse.ArgumentParser(description='내부자 대규모 지분 거래(블록딜) 탐지 시스템')
    parser.add_argument('--mode', type=str, choices=['train', 'monitor'], default='monitor',
                       help='실행 모드 (train: 모델 학습, monitor: 실시간 모니터링)')
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER,
                       help='종목코드 (기본값: 005930 삼성전자)')
    parser.add_argument('--start_date', type=str,
                       help='시작일자 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str,
                       help='종료일자 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # 시스템 초기화 메시지
    logger.info("내부자 대규모 지분 거래(블록딜) 탐지 시스템 시작")
    logger.info(f"현재 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # API 연결 확인
    api_connection = verify_api_connections()
    
    if not api_connection:
        logger.error("API 연결 실패. 프로그램을 종료합니다.")
        return
    
    # 모드에 따른 실행
    if args.mode == "train":
        run_block_deal_detection_system(args.ticker, args.start_date, args.end_date)
    elif args.mode == "monitor":
        setup_monitoring_system()

if __name__ == "__main__":
    main()
