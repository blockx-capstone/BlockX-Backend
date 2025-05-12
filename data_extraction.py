"""
KRX 및 DART API를 통한 데이터 추출 모듈
"""
import requests
import pandas as pd
import numpy as np
import dart_fss as dart
from pykrx import stock
from datetime import datetime, timedelta
import time
import logging
from bs4 import BeautifulSoup
import FinanceDataReader as fdr
from dart_fss import get_corp_list
import math

from api_keys import DART_API_KEY, KRX_API_KEY

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("block_deal_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# DART API 초기화
try:
    dart.set_api_key(api_key=DART_API_KEY)
    logger.info("DART API 키 설정 완료")
except Exception as e:
    logger.error(f"DART API 키 설정 오류: {e}")

# KRX API 헤더 설정
KRX_HEADERS = {
    'AUTH_KEY': KRX_API_KEY,
    'content-type': 'application/json'
}

def get_stock_ohlcv_from_pykrx(ticker, start_date, end_date):
    """
    PyKRX 라이브러리를 통해 주식 OHLCV 데이터 추출
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
        
    Returns:
        DataFrame: OHLCV 데이터
    """
    try:
        # 날짜 형식 변환 (YYYY-MM-DD -> YYYYMMDD)
        start = start_date.replace('-', '')
        end = end_date.replace('-', '')
        
        # API 호출 시 요청 제한 방지를 위한 대기
        time.sleep(0.5)
        
        # OHLCV 데이터 가져오기
        df = stock.get_market_ohlcv(start, end, ticker)
        
        # 거래대금 계산
        df['trading_value'] = df['종가'] * df['거래량']
        
        # 컬럼명 영문으로 변경
        df.columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'trading_value', 'price_change']
        
        # 인덱스(날짜)를 컬럼으로 변환
        df.reset_index(inplace=True)
        df.rename(columns={'날짜': 'date'}, inplace=True)
        
        logger.info(f"OHLCV 데이터 추출 완료: {ticker}, {len(df)}개 레코드")
        return df
    
    except Exception as e:
        logger.error(f"OHLCV 데이터 추출 오류: {e}")
        # 오류 발생시 빈 데이터프레임 반환
        return pd.DataFrame()

def get_investor_trends_from_krx(ticker, date):
    """
    KRX API를 통해 투자자별 매매 동향 추출
    
    Args:
        ticker (str): 종목코드
        date (str): 날짜 (YYYY-MM-DD)
        
    Returns:
        DataFrame: 투자자별 매매 동향 데이터
    """
    try:
        # API 호출 시 요청 제한 방지를 위한 대기
        time.sleep(0.5)
        
        # KRX API URL 설정
        url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        
        # 요청 파라미터 설정
        params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT02402',
            'trdDd': date.replace('-', ''),
            'mktId': 'ALL',  # ALL: 전체시장, STK: 유가증권시장, KSQ: 코스닥, KNX: 코넥스
            'isuCd': ticker,
            'isuCd2': ticker,
            'share': '1',
            'money': '1'
        }
        
        # API 요청 및 응답 처리
        response = requests.post(url, headers=KRX_HEADERS, data=params)
        
        if response.status_code == 200:
            result = response.json()
            
            # 투자자별 매매 동향 데이터 추출
            investor_data = result.get('output', [])
            
            if investor_data:
                df = pd.DataFrame(investor_data)
                
                # 필요한 컬럼만 선택
                if 'invstTpNm' in df.columns:
                    df = df[['invstTpNm', 'sellQty', 'sellAmt', 'buyQty', 'buyAmt', 'netQty', 'netAmt']]
                    
                    # 컬럼명 영문으로 변경
                    df.columns = ['investor_type', 'sell_qty', 'sell_amount', 'buy_qty', 'buy_amount', 'net_qty', 'net_amount']
                    
                    # 날짜 컬럼 추가
                    df['date'] = date
                    
                    # 숫자형 변환
                    for col in ['sell_qty', 'sell_amount', 'buy_qty', 'buy_amount', 'net_qty', 'net_amount']:
                        df[col] = df[col].str.replace(',', '').astype(float)
                    
                    logger.info(f"투자자 매매 동향 데이터 추출 완료: {ticker}, {date}")
                    return df
                else:
                    logger.warning(f"투자자 매매 동향 데이터 컬럼 오류: {ticker}, {date}")
            else:
                logger.warning(f"투자자 매매 동향 데이터 없음: {ticker}, {date}")
                
            return pd.DataFrame()
        else:
            logger.error(f"KRX API 요청 오류: {response.status_code}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"투자자 매매 동향 데이터 추출 오류: {e}")
        return pd.DataFrame()

def get_insider_trading(ticker, start_date, end_date):
    """
    DART API를 통해 임원/주요주주 거래 정보 추출
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
        
    Returns:
        DataFrame: 내부자 거래 정보
    """
    try:
        # API 호출 시 요청 제한 방지를 위한 대기
        time.sleep(1)
        
        # 종목코드로 회사 정보 조회
        corp_list = dart.get_corp_list()
        corp_code = corp_list.find_by_stock_code(ticker).corp_code
        
        # API 요청 URL 및 파라미터 설정
        url = "https://opendart.fss.or.kr/api/elestock.json"
        params = {
            'crtfc_key': DART_API_KEY,
            'corp_code': corp_code,
            'start_date': start_date.replace('-', ''),
            'end_date': end_date.replace('-', '')
        }
        
        # API 요청 및 응답 처리
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            result = response.json()
            if result['status'] == '000':
                # 응답 데이터를 DataFrame으로 변환
                data = result.get('list', [])
                df = pd.DataFrame(data)
                
                # 데이터 전처리
                if not df.empty:
                    # 'remark' 컬럼에서 블록딜 거래 여부 파악
                    df['is_block_deal'] = 0
                    if 'remark' in df.columns:
                        df['is_block_deal'] = df['remark'].str.contains('블록|블럭|시간외대량|대량매매|협의매매', case=False).astype(int)
                    
                    # 거래 가치 계산
                    if 'price' in df.columns and 'quantity' in df.columns:
                        # 콤마(,) 제거 후 숫자형으로 변환
                        if df['price'].dtype == 'object':
                            df['price'] = df['price'].str.replace(',', '').astype(float)
                        if df['quantity'].dtype == 'object':
                            df['quantity'] = df['quantity'].str.replace(',', '').astype(float)
                            
                        df['transaction_value'] = df['price'].astype(float) * df['quantity'].astype(float)
                
                logger.info(f"내부자 거래 정보 추출 완료: {ticker}, {len(df)}개 레코드")
                return df
            else:
                logger.warning(f"DART API 응답 오류: {result['message']}")
                return pd.DataFrame()
        else:
            logger.error(f"DART API 요청 오류: {response.status_code}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"내부자 거래 정보 추출 오류: {e}")
        return pd.DataFrame()

def get_disclosure_list(ticker, start_date, end_date):
    """
    DART API를 통해 공시 목록 추출
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
        
    Returns:
        DataFrame: 공시 목록
    """
    try:
        # API 호출 시 요청 제한 방지를 위한 대기
        time.sleep(1)
        
        # 종목코드로 회사 정보 조회
        corp_list = dart.get_corp_list()
        corp_code = corp_list.find_by_stock_code(ticker).corp_code
        
        # API 요청 URL 및 파라미터 설정
        url = "https://opendart.fss.or.kr/api/list.json"
        
        # 날짜 형식 변환 (YYYY-MM-DD -> YYYYMMDD)
        bgn_de = start_date.replace('-', '')
        end_de = end_date.replace('-', '')
        
        params = {
            'crtfc_key': DART_API_KEY,
            'corp_code': corp_code,
            'bgn_de': bgn_de,
            'end_de': end_de,
            'page_no': 1,
            'page_count': 100
        }
        
        all_data = []
        
        # 첫 페이지 요청
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            result = response.json()
            if result['status'] == '000':
                # 전체 페이지 수 계산
                total_count = int(result.get('total_count', 0))
                total_page = math.ceil(total_count / 100)
                
                # 첫 페이지 데이터 저장
                data = result.get('list', [])
                all_data.extend(data)
                
                # 2페이지부터 순차적으로 요청
                for page in range(2, total_page + 1):
                    # API 요청 제한 방지를 위한 대기
                    time.sleep(0.5)
                    
                    params['page_no'] = page
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result['status'] == '000':
                            data = result.get('list', [])
                            all_data.extend(data)
                
                if not all_data:
                    logger.info(f"{ticker} 종목 해당 기간 공시 정보 없음")
                    return pd.DataFrame()
                
                # 데이터프레임 생성
                df = pd.DataFrame(all_data)
                
                # 날짜 형식 변환
                if 'rcept_dt' in df.columns:
                    df['date'] = pd.to_datetime(df['rcept_dt']).dt.strftime('%Y-%m-%d')
                
                # 대량 지분 변동 관련 공시 필터링
                keywords = ['지분', '대량', '블록딜', '주식', '처분', '취득', '변동', '임원', '주요주주']
                if 'report_nm' in df.columns:
                    # 키워드 포함 여부 확인을 위한 컬럼 추가
                    df['keyword_match'] = df['report_nm'].str.contains('|'.join(keywords), case=False, na=False)
                    
                    # 필터링된 데이터 추출
                    filtered_df = df[df['keyword_match']]
                    logger.info(f"공시 목록 추출 완료: {ticker}, 전체 {len(df)}개 중 {len(filtered_df)}개 관련 공시 발견")
                    
                    # 필터링 컬럼 제거
                    if not filtered_df.empty:
                        filtered_df = filtered_df.drop('keyword_match', axis=1)
                    
                    return filtered_df
                else:
                    logger.warning(f"{ticker} 종목 공시 정보 컬럼 오류")
                    return df
            else:
                logger.warning(f"DART API 응답 오류: {result.get('message', '')}")
                return pd.DataFrame()
        else:
            logger.error(f"DART API 요청 오류: {response.status_code}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"공시 목록 추출 오류: {e}")
        return pd.DataFrame()

def get_financial_statements(ticker, start_date, end_date):
    """
    재무제표 데이터 수집
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
        
    Returns:
        DataFrame: 재무제표 데이터
    """
    try:
        # 종목코드로 회사 정보 조회
        corp = dart.get_corp_list().find_by_stock_code(ticker)
        if not corp:
            logger.error(f"{ticker} 기업 정보 조회 실패")
            return pd.DataFrame()
        
        # 날짜 형식 변환 (YYYY-MM-DD -> YYYYMMDD)
        bgn_de = start_date.replace('-', '')
        end_de = end_date.replace('-', '')
        
        # 재무제표 데이터 수집
        try:
            # DART-FSS 라이브러리를 이용한 재무제표 추출
            # 참고: https://dart-fss.readthedocs.io/en/latest/_modules/dart_fss/fs/extract.html
            fs = corp.extract_fs(bgn_de=bgn_de, end_de=end_de, 
                                fs_tp=('bs', 'is', 'cis', 'cf'),  # 재무상태표, 손익계산서, 포괄손익계산서, 현금흐름표
                                report_tp='annual',  # 연간보고서 기준
                                lang='ko',  # 한글 재무제표
                                separator=True,  # 1000단위 구분자 표시
                                dataset='xbrl')  # XBRL 데이터셋 사용
        except Exception as fs_error:
            # 연간 보고서가 없을 경우 분기 보고서 시도
            try:
                logger.warning(f"{ticker} 연간 재무제표 추출 오류: {fs_error}. 분기 보고서로 대체합니다.")
                fs = corp.extract_fs(bgn_de=bgn_de, end_de=end_de, 
                                    fs_tp=('bs', 'is', 'cis', 'cf'),
                                    report_tp='quarter',  # 분기보고서 포함
                                    lang='ko',
                                    separator=True,
                                    dataset='xbrl')
            except Exception as qtr_error:
                logger.warning(f"{ticker} 분기 재무제표 추출 오류: {qtr_error}. 기본 재무 데이터로 대체합니다.")
                # 기본 재무 데이터 생성
                financial_data = pd.DataFrame({
                    'revenue': [0],
                    'operating_profit': [0], 
                    'net_income': [0],
                    'total_assets': [0],
                    'total_liabilities': [0],
                    'equity': [0],
                    'operating_cashflow': [0],
                    'investing_cashflow': [0],
                    'financing_cashflow': [0],
                    'debt_ratio': [0],
                    'roe': [0],
                    'operating_margin': [0]
                })
                return financial_data
        
        # 주요 재무지표 추출
        financial_data = pd.DataFrame()
        
        # 손익계산서
        if hasattr(fs, 'is_') and fs.is_ is not None:
            try:
                income = fs.is_
                if '매출액' in income.columns:
                    financial_data['revenue'] = income['매출액']
                elif '영업수익' in income.columns:
                    financial_data['revenue'] = income['영업수익']
                elif '수익(매출)' in income.columns:
                    financial_data['revenue'] = income['수익(매출)']
                
                if '영업이익' in income.columns:
                    financial_data['operating_profit'] = income['영업이익']
                elif '영업이익(손실)' in income.columns:
                    financial_data['operating_profit'] = income['영업이익(손실)']
                
                if '당기순이익' in income.columns:
                    financial_data['net_income'] = income['당기순이익']
                elif '당기순이익(손실)' in income.columns:
                    financial_data['net_income'] = income['당기순이익(손실)']
            except Exception as e:
                logger.warning(f"{ticker} 손익계산서 데이터 추출 오류: {e}")
                financial_data['revenue'] = 0
                financial_data['operating_profit'] = 0
                financial_data['net_income'] = 0
        else:
            financial_data['revenue'] = 0
            financial_data['operating_profit'] = 0
            financial_data['net_income'] = 0
        
        # 재무상태표
        if hasattr(fs, 'bs') and fs.bs is not None:
            try:
                balance = fs.bs
                if '자산총계' in balance.columns:
                    financial_data['total_assets'] = balance['자산총계']
                elif '자산' in balance.columns:
                    financial_data['total_assets'] = balance['자산']
                
                if '부채총계' in balance.columns:
                    financial_data['total_liabilities'] = balance['부채총계']
                elif '부채' in balance.columns:
                    financial_data['total_liabilities'] = balance['부채']
                
                if '자본총계' in balance.columns:
                    financial_data['equity'] = balance['자본총계']
                elif '자본' in balance.columns:
                    financial_data['equity'] = balance['자본']
            except Exception as e:
                logger.warning(f"{ticker} 대차대조표 데이터 추출 오류: {e}")
                financial_data['total_assets'] = 0
                financial_data['total_liabilities'] = 0
                financial_data['equity'] = 0
        else:
            financial_data['total_assets'] = 0
            financial_data['total_liabilities'] = 0
            financial_data['equity'] = 0
        
        # 현금흐름표
        if hasattr(fs, 'cf') and fs.cf is not None:
            try:
                cashflow = fs.cf
                # 다양한 열 이름 대응
                for col_name, alt_names in {
                    'operating_cashflow': ['영업활동현금흐름', '영업활동으로 인한 현금흐름', '영업활동 순현금흐름'],
                    'investing_cashflow': ['투자활동현금흐름', '투자활동으로 인한 현금흐름', '투자활동 순현금흐름'],
                    'financing_cashflow': ['재무활동현금흐름', '재무활동으로 인한 현금흐름', '재무활동 순현금흐름']
                }.items():
                    for alt_name in alt_names:
                        if alt_name in cashflow.columns:
                            financial_data[col_name] = cashflow[alt_name]
                            break
                    else:
                        financial_data[col_name] = 0
            except Exception as e:
                logger.warning(f"{ticker} 현금흐름표 데이터 추출 오류: {e}")
                financial_data['operating_cashflow'] = 0
                financial_data['investing_cashflow'] = 0
                financial_data['financing_cashflow'] = 0
        else:
            financial_data['operating_cashflow'] = 0
            financial_data['investing_cashflow'] = 0
            financial_data['financing_cashflow'] = 0
        
        # 재무비율 계산
        if not financial_data.empty:
            try:
                if 'total_assets' in financial_data and 'total_liabilities' in financial_data:
                    financial_data['debt_ratio'] = financial_data['total_liabilities'] / financial_data['total_assets']
                else:
                    financial_data['debt_ratio'] = 0
                    
                if 'net_income' in financial_data and 'equity' in financial_data:
                    financial_data['roe'] = financial_data['net_income'] / financial_data['equity']
                else:
                    financial_data['roe'] = 0
                    
                if 'operating_profit' in financial_data and 'revenue' in financial_data:
                    financial_data['operating_margin'] = financial_data['operating_profit'] / financial_data['revenue']
                else:
                    financial_data['operating_margin'] = 0
            except Exception as e:
                logger.warning(f"{ticker} 재무비율 계산 오류: {e}")
                financial_data['debt_ratio'] = 0
                financial_data['roe'] = 0
                financial_data['operating_margin'] = 0
        
        # NaN 값을 0으로 대체
        financial_data = financial_data.fillna(0)
        
        return financial_data
    
    except Exception as e:
        logger.error(f"재무제표 데이터 수집 오류: {e}")
        return pd.DataFrame()

def get_disclosure_info(ticker, start_date, end_date):
    """
    공시 정보 수집
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
        
    Returns:
        DataFrame: 공시 정보
    """
    try:
        # get_disclosure_list 함수를 사용하여 공시 정보 수집
        disclosure_data = get_disclosure_list(ticker, start_date, end_date)
        
        if disclosure_data.empty:
            logger.warning(f"{ticker} 공시 정보 없음")
            return pd.DataFrame()
        
        # 공시 유형 분류
        if 'report_nm' in disclosure_data.columns:
            disclosure_data['disclosure_type'] = disclosure_data['report_nm'].apply(
                lambda x: '실적' if '실적' in x else
                         '주주총회' if '주주총회' in x else
                         'M&A' if any(keyword in x for keyword in ['인수', '합병', '매각']) else
                         '투자' if '투자' in x else
                         '기타'
            )
        else:
            # report_nm 컬럼이 없는 경우 기본값 설정
            disclosure_data['disclosure_type'] = '기타'
        
        return disclosure_data
    
    except Exception as e:
        logger.error(f"공시 정보 수집 오류: {e}")
        return pd.DataFrame()

def get_news_data(ticker, start_date, end_date):
    """
    뉴스 데이터 수집
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
        
    Returns:
        DataFrame: 뉴스 데이터
    """
    try:
        # 네이버 금융 뉴스 수집
        url = f"https://finance.naver.com/item/news_news.naver?code={ticker}&page=1"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_data = []
        news_items = soup.select('table.type5 tr')
        
        for item in news_items:
            try:
                date = item.select_one('td.date')
                title = item.select_one('td.title')
                if date and title:
                    news_date = datetime.strptime(date.text.strip(), '%Y.%m.%d')
                    if start_date <= news_date <= end_date:
                        news_data.append({
                            'date': news_date.strftime('%Y-%m-%d'),
                            'title': title.text.strip(),
                            'link': title.find('a')['href'] if title.find('a') else None
                        })
            except Exception as e:
                continue
        
        return pd.DataFrame(news_data)
    
    except Exception as e:
        logger.error(f"뉴스 데이터 수집 오류: {e}")
        return pd.DataFrame()

def get_market_indicators(ticker, start_date, end_date):
    """
    시장 지표 데이터 수집
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
        
    Returns:
        DataFrame: 시장 지표 데이터
    """
    try:
        # 데이터 통합을 위한 날짜 인덱스 생성
        date_range = pd.date_range(start=start_date, end=end_date)
        market_indicators = pd.DataFrame(index=date_range)
        market_indicators.index.name = 'date'
        market_indicators.reset_index(inplace=True)
        market_indicators['date'] = market_indicators['date'].dt.strftime('%Y-%m-%d')
        
        # 1. 섹터 지수 수집 (FinanceDataReader 사용)
        sector_indices = {
            'KOSPI': '^KS11',
            'KOSDAQ': '^KQ11',
            'KOSPI200': '^KS200'
        }
        
        for sector_name, sector_code in sector_indices.items():
            try:
                # FinanceDataReader를 이용한 지수 데이터 수집
                sector_df = fdr.DataReader(sector_code, start_date, end_date)
                
                if not sector_df.empty and 'Close' in sector_df.columns:
                    # 날짜 형식 통일
                    sector_data = pd.DataFrame({
                        'date': pd.to_datetime(sector_df.index).strftime('%Y-%m-%d'),
                        f'{sector_name}_index': sector_df['Close']
                    })
                    
                    # 왼쪽 조인 (날짜 기준)
                    market_indicators = market_indicators.merge(sector_data, on='date', how='left')
                    logger.info(f"{sector_name} 지수 데이터 수집 완료: {len(sector_data)}개 레코드")
            except Exception as e:
                logger.warning(f"{sector_name} 지수 데이터 수집 실패: {e}")
                market_indicators[f'{sector_name}_index'] = 0
        
        # 2. 종목별 시장 지표 수집 (PyKRX 사용)
        try:
            # 날짜 형식 변환 (YYYY-MM-DD -> YYYYMMDD)
            start = start_date.replace('-', '')
            end = end_date.replace('-', '')
            
            # 시간 간격을 두고 요청 (API 제한 방지)
            time.sleep(0.5)
            
            # PyKRX를 이용한 시장 지표 수집
            stock_data = stock.get_market_fundamental_by_date(start, end, ticker)
            
            if not stock_data.empty:
                # 데이터프레임 변환
                stock_df = stock_data.reset_index()
                
                # 날짜 형식 변환
                stock_df['날짜'] = pd.to_datetime(stock_df['날짜']).dt.strftime('%Y-%m-%d')
                
                # 필요한 컬럼만 선택 및 이름 변경
                cols_map = {'날짜': 'date'}
                selected_cols = ['날짜']
                
                # 존재하는 컬럼만 처리
                for col, new_name in [('PER', 'PER'), ('PBR', 'PBR'), ('EPS', 'EPS'), 
                                    ('BPS', 'BPS'), ('DIV', 'DIV'), ('DPS', 'DPS')]:
                    if col in stock_df.columns:
                        selected_cols.append(col)
                        cols_map[col] = new_name
                
                # 컬럼 선택 및 이름 변경
                if len(selected_cols) > 1:  # 날짜 외에 다른 컬럼이 있는 경우
                    stock_df = stock_df[selected_cols]
                    stock_df = stock_df.rename(columns=cols_map)
                    
                    # 지표 데이터와 병합
                    market_indicators = market_indicators.merge(stock_df, on='date', how='left')
                    logger.info(f"{ticker} 시장 지표 데이터 수집 완료: {len(stock_df)}개 레코드")
            else:
                logger.warning(f"{ticker} 시장 지표 데이터 없음")
                # 기본 컬럼 추가
                market_indicators['PER'] = 0
                market_indicators['PBR'] = 0
        except Exception as e:
            logger.warning(f"{ticker} 시장 지표 데이터 수집 오류: {e}")
            # 기본 컬럼 추가
            market_indicators['PER'] = 0
            market_indicators['PBR'] = 0
        
        # 3. 추가 시장 데이터 수집 (FDR 사용)
        try:
            # 종목 주가 데이터 (추가 정보)
            stock_price_df = fdr.DataReader(ticker, start_date, end_date)
            
            if not stock_price_df.empty:
                # 날짜 형식 변환
                stock_price_data = pd.DataFrame({
                    'date': pd.to_datetime(stock_price_df.index).strftime('%Y-%m-%d'),
                    'market_cap': stock_price_df['Close'] * stock_price_df['Volume']  # 시가총액 근사치
                })
                
                # 왼쪽 조인 (날짜 기준)
                market_indicators = market_indicators.merge(stock_price_data, on='date', how='left')
                logger.info(f"{ticker} 시가총액 데이터 수집 완료")
        except Exception as e:
            logger.warning(f"{ticker} 시가총액 데이터 수집 오류: {e}")
            market_indicators['market_cap'] = 0
        
        # 결측치를 ffill 방식으로 채우고 남은 결측치는 0으로 대체
        market_indicators = market_indicators.ffill().fillna(0)
        
        return market_indicators
    
    except Exception as e:
        logger.error(f"시장 지표 데이터 수집 오류: {e}")
        # 오류 발생 시 최소한의 데이터프레임 반환 (날짜 컬럼 포함)
        date_range = pd.date_range(start=start_date, end=end_date)
        default_df = pd.DataFrame(index=date_range)
        default_df.index.name = 'date'
        default_df.reset_index(inplace=True)
        default_df['date'] = default_df['date'].dt.strftime('%Y-%m-%d')
        default_df['PER'] = 0
        default_df['PBR'] = 0
        
        for sector in ['KOSPI', 'KOSDAQ', 'KOSPI200']:
            default_df[f'{sector}_index'] = 0
            
        default_df['market_cap'] = 0
        return default_df

def get_block_deal_history(ticker, start_date, end_date):
    """
    과거 블록딜 발생 이력 수집
    DART API의 '임원/주요주주 특정증권 등 소유상황보고서' API 사용
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
        
    Returns:
        DataFrame: 블록딜 발생 이력
    """
    try:
        # DART API를 통해 대량주식거래 내역 수집
        corp_list = dart.get_corp_list()
        corp_code = corp_list.find_by_stock_code(ticker).corp_code
        
        # 날짜 형식 변환 (YYYY-MM-DD -> YYYYMMDD)
        # 하이픈(-) 제거하여 API에서 요구하는 YYYYMMDD 형태로 변환
        start_dt = start_date.replace('-', '')
        end_dt = end_date.replace('-', '')
        
        # 임원/주요주주 특정증권 등 소유상황보고서 API 호출
        url = "https://opendart.fss.or.kr/api/elestock.json"
        params = {
            'crtfc_key': DART_API_KEY,
            'corp_code': corp_code,
            'start_dt': start_dt,  # YYYYMMDD 형식 사용
            'end_dt': end_dt       # YYYYMMDD 형식 사용
        }
        
        logger.info(f"DART API 요청: {url}, 시작일: {start_dt}, 종료일: {end_dt}, 기업코드: {corp_code}")
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"DART API 응답 코드: {result.get('status')}, 메시지: {result.get('message', '')}")
            
            if result.get('status') == '000':
                data = result.get('list', [])
                
                if not data:
                    logger.info(f"{ticker} 종목 해당 기간 내 대량주식거래 내역 없음")
                    return pd.DataFrame()
                
                # 데이터프레임 생성
                df = pd.DataFrame(data)
                logger.info(f"대량주식거래 데이터 수집 완료: {ticker}, {len(df)}개 레코드")
                logger.info(f"수집된 데이터 컬럼: {df.columns.tolist()}")
                
                # 컬럼 존재 여부 확인 및 데이터 변환
                # 참고: DART API elestock.json 응답 형식
                # - rcept_no: 접수번호
                # - rcept_dt: 접수일자
                # - stock_code: 종목코드
                # - report_tp: 보고서 유형
                # - repror: 보고자
                # - report_resn: 보고사유 (이 필드가 없을 수 있음)
                # - corp_cls: 법인구분
                # - corp_code: 고유번호
                # - corp_name: 회사명
                # - stkqy: 주식 등의 수
                # - stkqy_irds: 주식 등의 수 증감
                # - ctrqty: 주식 등의 수 비율
                
                # 응답 필드 목록 확인 및 로깅
                required_fields = ['rcept_no', 'rcept_dt', 'stock_code']
                missing_fields = [field for field in required_fields if field not in df.columns]
                if missing_fields:
                    logger.warning(f"필수 필드 누락: {missing_fields}")
                
                # 블록딜 여부 판단 (보고사유 또는 대체 로직 활용)
                if 'report_resn' in df.columns:
                    block_deal_keywords = ['매매', '장외매매', '장외매도', '장외매수', '시간외대량매매', '대량매매', '블록딜']
                    df['is_block_deal'] = df['report_resn'].str.contains('|'.join(block_deal_keywords), case=False, na=False).astype(int)
                    logger.info(f"'report_resn' 컬럼 사용하여 블록딜 여부 판단")
                else:
                    # 보고사유 컬럼이 없는 경우 대체 로직
                    logger.warning(f"'report_resn' 컬럼이 없습니다. 대체 로직으로 블록딜 여부 판단")
                    
                    # report_tp(보고서 유형) 컬럼이 있는 경우 활용
                    if 'report_tp' in df.columns:
                        block_report_types = ['주식등의대량보유상황보고서', '임원주요주주소유상황보고서']
                        df['is_block_deal'] = df['report_tp'].str.contains('|'.join(block_report_types), case=False, na=False).astype(int)
                        logger.info(f"'report_tp' 컬럼 사용하여 블록딜 여부 판단")
                    # stkqy_irds(주식 등의 수 증감) 컬럼이 있는 경우 활용
                    elif 'stkqy_irds' in df.columns:
                        # 문자열로 되어 있는 수량 정보를 숫자로 변환
                        df['quantity'] = df['stkqy_irds'].str.replace(',', '').astype(float)
                        # 거래량 상위 30%를 블록딜로 간주
                        threshold = df['quantity'].quantile(0.7)
                        df['is_block_deal'] = (df['quantity'].abs() > threshold).astype(int)
                        logger.info(f"'stkqy_irds' 컬럼 사용하여 블록딜 여부 판단, 임계값: {threshold}")
                    else:
                        # 수량 정보도 없는 경우 모든 건을 블록딜로 간주
                        df['is_block_deal'] = 1
                        logger.warning("블록딜 여부 판단을 위한 컬럼이 없어 모든 건을 블록딜로 간주합니다.")
                
                # 블록딜 데이터만 필터링
                block_deals = df[df['is_block_deal'] == 1].copy()
                
                if block_deals.empty:
                    logger.info(f"{ticker} 종목 해당 기간 내 블록딜 없음")
                    return pd.DataFrame()
                
                # 거래일자 설정
                if 'rcept_dt' in block_deals.columns:
                    # 접수일자 형식: YYYYMMDD
                    try:
                        block_deals['transaction_date'] = pd.to_datetime(block_deals['rcept_dt'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
                        logger.info("접수일자를 거래일자로 설정")
                    except Exception as date_error:
                        logger.error(f"날짜 변환 오류: {date_error}, 샘플 데이터: {block_deals['rcept_dt'].iloc[0] if not block_deals.empty else 'N/A'}")
                        # 오류 발생 시 현재 날짜 사용
                        block_deals['transaction_date'] = datetime.now().strftime('%Y-%m-%d')
                else:
                    logger.warning(f"거래일자 컬럼을 찾을 수 없습니다. 현재 날짜를 사용합니다.")
                    block_deals['transaction_date'] = datetime.now().strftime('%Y-%m-%d')
                
                # 거래 정보 추가
                if 'stkqy_irds' in block_deals.columns:
                    # 주식수 및 변동량
                    block_deals['quantity'] = block_deals['stkqy_irds'].str.replace(',', '').astype(float)
                
                # 추가 정보 계산 (가격 정보가 없는 경우 대체 데이터 생성)
                try:
                    # 해당 일자의 종가 정보 가져오기
                    ohlcv_data = get_stock_ohlcv_from_pykrx(ticker, start_date, end_date)
                    if not ohlcv_data.empty:
                        # 거래일자별 종가 매핑
                        price_dict = dict(zip(ohlcv_data['date'].astype(str), ohlcv_data['close_price']))
                        
                        # 거래일자별 가격 정보 할당
                        block_deals['price'] = block_deals['transaction_date'].map(price_dict)
                        
                        # 거래대금 계산 (가격 * 수량)
                        if 'quantity' in block_deals.columns and 'price' in block_deals.columns:
                            block_deals['transaction_value'] = block_deals['quantity'].abs() * block_deals['price']
                
                except Exception as e:
                    logger.warning(f"가격 정보 매핑 오류: {e}")
                
                logger.info(f"블록딜 이력 수집 완료: {ticker}, {len(block_deals)}개 발생")
                return block_deals
            else:
                error_msg = result.get('message', '알 수 없는 오류')
                logger.warning(f"DART API 응답 오류: {error_msg}")
                return pd.DataFrame()
        else:
            logger.error(f"DART API 요청 오류: {response.status_code}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"블록딜 이력 수집 오류: {e}")
        return pd.DataFrame()

def get_daily_market_data(ticker, start_date, end_date):
    """
    일별 시장 데이터 수집 (OHLCV + 추가 지표)
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
        
    Returns:
        DataFrame: 일별 시장 데이터
    """
    try:
        # 기본 OHLCV 데이터
        market_data = get_stock_ohlcv_from_pykrx(ticker, start_date, end_date)
        
        if market_data.empty:
            return pd.DataFrame()
        
        # 거래량 관련 지표 추가
        market_data['volume_ma5'] = market_data['volume'].rolling(window=5).mean()
        market_data['volume_ma20'] = market_data['volume'].rolling(window=20).mean()
        market_data['volume_ratio'] = market_data['volume'] / market_data['volume_ma5']
        
        # 가격 변동성 지표
        market_data['daily_return'] = market_data['close_price'].pct_change()
        market_data['volatility'] = market_data['daily_return'].rolling(window=20).std()
        
        # 이동평균선
        market_data['ma5'] = market_data['close_price'].rolling(window=5).mean()
        market_data['ma20'] = market_data['close_price'].rolling(window=20).mean()
        market_data['ma60'] = market_data['close_price'].rolling(window=60).mean()
        
        # 거래대금
        market_data['trading_value'] = market_data['close_price'] * market_data['volume']
        market_data['trading_value_ma5'] = market_data['trading_value'].rolling(window=5).mean()
        
        # 결측치 처리 - FutureWarning 해결
        market_data = market_data.ffill().fillna(0)
        
        return market_data
    
    except Exception as e:
        logger.error(f"일별 시장 데이터 수집 오류: {e}")
        return pd.DataFrame()
