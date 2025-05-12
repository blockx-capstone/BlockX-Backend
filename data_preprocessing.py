"""
데이터 통합 및 전처리 모듈
"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import re
from datetime import datetime, timedelta

from data_extraction import (
    get_daily_market_data,
    get_block_deal_history,
    get_financial_statements,
    get_disclosure_info,
    get_news_data,
    get_market_indicators
)

logger = logging.getLogger(__name__)

def prepare_data_for_prediction(ticker, start_date, end_date, prediction_window=5):
    """
    블록딜 발생 가능성 예측을 위한 데이터 준비
    
    Args:
        ticker (str): 종목코드
        start_date (str): 시작일자 (YYYY-MM-DD)
        end_date (str): 종료일자 (YYYY-MM-DD)
        prediction_window (int): 예측 기간 (일)
        
    Returns:
        tuple: (X, y, scaler) - 특성 데이터, 타겟 데이터, 스케일러
    """
    try:
        logger.info(f"\n{ticker} 데이터 준비 시작 ({start_date} ~ {end_date})...")
        
        # 1. 일별 시장 데이터 수집
        market_data = get_daily_market_data(ticker, start_date, end_date)
        if market_data.empty:
            logger.error(f"{ticker} 시장 데이터 수집 실패")
            return None, None, None
        
        # 'date' 컬럼 형식 통일 및 확인
        if 'date' not in market_data.columns:
            logger.error(f"{ticker} 시장 데이터에 'date' 컬럼이 없습니다.")
            return None, None, None
        
        # 날짜 형식 통일
        market_data['date'] = pd.to_datetime(market_data['date']).dt.strftime('%Y-%m-%d')
        
        # 2. 블록딜 이력 수집
        block_deals = get_block_deal_history(ticker, start_date, end_date)
        
        # 3. 추가 데이터 수집
        financial_data = get_financial_statements(ticker, start_date, end_date)
        disclosure_data = get_disclosure_info(ticker, start_date, end_date)
        news_data = get_news_data(ticker, start_date, end_date)
        market_indicators = get_market_indicators(ticker, start_date, end_date)
        
        # 4. 데이터 전처리
        logger.info(f"{ticker} 데이터 전처리 중...")
        
        # 4-1. 블록딜 발생 여부 레이블 생성
        market_data['has_block_deal'] = 0
        has_block_deal_data = False
        
        if not block_deals.empty:
            # transaction_date 컬럼이 없는 경우 대체 로직
            if 'transaction_date' not in block_deals.columns:
                logger.warning(f"{ticker} 블록딜 데이터에 'transaction_date' 컬럼이 없습니다. 'rcept_dt' 컬럼을 사용합니다.")
                # 공시 접수일을 거래일로 대체
                if 'rcept_dt' in block_deals.columns:
                    block_deals['transaction_date'] = pd.to_datetime(block_deals['rcept_dt'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
                else:
                    logger.warning(f"{ticker} 블록딜 데이터에 날짜 관련 컬럼이 없습니다. 레이블 생성을 건너뜁니다.")
                    block_deals = pd.DataFrame()  # 빈 데이터프레임으로 설정하여 이후 처리 건너뜀
            
            # 정상적인 블록딜 데이터가 있는 경우 레이블 생성
            if not block_deals.empty and 'transaction_date' in block_deals.columns:
                # is_block_deal 필드가 없는 경우, report_resn 컬럼으로 블록딜 여부 판단
                if 'is_block_deal' not in block_deals.columns:
                    if 'report_resn' in block_deals.columns:
                        block_deal_keywords = ['매매', '장외매매', '장외매도', '장외매수', '시간외대량매매', '대량매매', '블록딜']
                        block_deals['is_block_deal'] = block_deals['report_resn'].str.contains('|'.join(block_deal_keywords), case=False, na=False).astype(int)
                    else:
                        # 보고 사유 필드도 없는 경우 모두 블록딜로 간주
                        logger.warning(f"{ticker} 블록딜 데이터에 'is_block_deal' 및 'report_resn' 컬럼이 없습니다. 모두 블록딜로 간주합니다.")
                        block_deals['is_block_deal'] = 1
                
                # 실제 블록딜 데이터만 필터링
                block_deals_filtered = block_deals[block_deals['is_block_deal'] == 1]
                
                if not block_deals_filtered.empty:
                    has_block_deal_data = True
                    for _, deal in block_deals_filtered.iterrows():
                        try:
                            deal_date = pd.to_datetime(deal['transaction_date'])
                            # 블록딜 발생일로부터 prediction_window일 이내의 날짜에 레이블 1 부여
                            for i in range(prediction_window):
                                target_date = (deal_date - timedelta(days=i)).strftime('%Y-%m-%d')
                                market_data.loc[market_data['date'] == target_date, 'has_block_deal'] = 1
                        except Exception as e:
                            logger.warning(f"블록딜 레이블 생성 오류: {e}")
                            continue
        
        # 블록딜 레이블 확인 및 로깅
        block_deal_count = market_data['has_block_deal'].sum()
        total_count = len(market_data)
        logger.info(f"{ticker} 블록딜 레이블 생성: 전체 {total_count}일 중 {block_deal_count}일 발생 (비율: {block_deal_count/total_count:.2%})")
        
        # 블록딜 발생 데이터가 없거나 매우 적은 경우 경고
        if block_deal_count == 0:
            logger.warning(f"{ticker} 블록딜 발생 이력이 없습니다. 학습에 어려움이 있을 수 있습니다.")
        elif block_deal_count / total_count < 0.05:  # 5% 미만인 경우
            logger.warning(f"{ticker} 블록딜 발생 비율이 매우 낮습니다({block_deal_count/total_count:.2%}). 학습 시 클래스 불균형 문제가 있을 수 있습니다.")
        
        # 4-2. 재무 데이터 통합
        if not financial_data.empty:
            try:
                # 가장 최근 분기 데이터 사용
                latest_financial = financial_data.iloc[-1]
                for col in financial_data.columns:
                    market_data[col] = latest_financial[col]
            except Exception as e:
                logger.warning(f"재무 데이터 통합 오류: {e}")
                # 기본 재무 데이터 컬럼 추가
                for col in ['revenue', 'operating_profit', 'net_income', 'total_assets', 
                           'total_liabilities', 'equity', 'operating_cashflow', 
                           'investing_cashflow', 'financing_cashflow', 'debt_ratio', 
                           'roe', 'operating_margin']:
                    if col not in market_data.columns:
                        market_data[col] = 0
        
        # 4-3. 공시 데이터 통합
        if not disclosure_data.empty and 'date' in disclosure_data.columns and 'disclosure_type' in disclosure_data.columns:
            try:
                # 최근 30일 공시 집계
                for date_str in market_data['date'].unique():
                    date = pd.to_datetime(date_str)
                    thirty_days_ago = (date - timedelta(days=30)).strftime('%Y-%m-%d')
                    
                    recent_disclosures = disclosure_data[
                        (pd.to_datetime(disclosure_data['date']) <= date) & 
                        (pd.to_datetime(disclosure_data['date']) >= pd.to_datetime(thirty_days_ago))
                    ]
                    
                    if not recent_disclosures.empty:
                        for dtype in recent_disclosures['disclosure_type'].unique():
                            count = len(recent_disclosures[recent_disclosures['disclosure_type'] == dtype])
                            col_name = f'disclosure_{dtype}'
                            market_data.loc[market_data['date'] == date_str, col_name] = count
            except Exception as e:
                logger.warning(f"공시 데이터 통합 오류: {e}")
        
        # 기본 공시 데이터 컬럼 추가
        for dtype in ['실적', '주주총회', 'M&A', '투자', '기타']:
            col_name = f'disclosure_{dtype}'
            if col_name not in market_data.columns:
                market_data[col_name] = 0
        
        # 4-4. 뉴스 데이터 통합
        if not news_data.empty and 'date' in news_data.columns and 'title' in news_data.columns:
            try:
                # 뉴스 감성 분석 및 키워드 추출
                news_data['sentiment'] = news_data['title'].apply(
                    lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
                )
                
                keywords = ['실적', '투자', '인수', '합병', '매각', '주주총회', '배당', '감자', '증자']
                for keyword in keywords:
                    news_data[f'has_{keyword}'] = news_data['title'].apply(
                        lambda x: 1 if pd.notna(x) and keyword in str(x) else 0
                    )
                
                # 최근 30일 뉴스 집계
                for date_str in market_data['date'].unique():
                    date = pd.to_datetime(date_str)
                    thirty_days_ago = (date - timedelta(days=30)).strftime('%Y-%m-%d')
                    
                    recent_news = news_data[
                        (pd.to_datetime(news_data['date']) <= date) & 
                        (pd.to_datetime(news_data['date']) >= pd.to_datetime(thirty_days_ago))
                    ]
                    
                    if not recent_news.empty:
                        market_data.loc[market_data['date'] == date_str, 'news_sentiment'] = recent_news['sentiment'].mean()
                        for keyword in keywords:
                            market_data.loc[market_data['date'] == date_str, f'news_{keyword}'] = \
                                recent_news[f'has_{keyword}'].sum()
            except Exception as e:
                logger.warning(f"뉴스 데이터 통합 오류: {e}")
        
        # 기본 뉴스 데이터 컬럼 추가
        if 'news_sentiment' not in market_data.columns:
            market_data['news_sentiment'] = 0
        
        keywords = ['실적', '투자', '인수', '합병', '매각', '주주총회', '배당', '감자', '증자']
        for keyword in keywords:
            col_name = f'news_{keyword}'
            if col_name not in market_data.columns:
                market_data[col_name] = 0
        
        # 4-5. 시장 지표 통합
        if not market_indicators.empty and 'date' in market_indicators.columns:
            try:
                # 날짜 형식이 일치하는지 확인
                market_indicators['date'] = pd.to_datetime(market_indicators['date']).dt.strftime('%Y-%m-%d')
                
                # 시장 지표 통합
                market_data = market_data.merge(market_indicators, on='date', how='left')
            except Exception as e:
                logger.warning(f"시장 지표 통합 오류: {e}")
        else:
            logger.warning(f"{ticker} 시장 지표 데이터가 비어있거나 'date' 컬럼이 없습니다. 기본값으로 대체합니다.")
            # 기본 시장 지표 컬럼 추가
            market_data['PER'] = 0
            market_data['PBR'] = 0
            # ROE가 없을 수 있으므로 조건부로 추가
            if 'roe' not in market_data.columns:  # 재무 데이터에서 이미 ROE 컬럼이 추가되었을 수 있음
                market_data['roe'] = 0
            # 주요 섹터 지수 추가
            for sector in ['KOSPI', 'KOSDAQ', 'KOSPI200']:
                market_data[f'{sector}_index'] = 0
            # 시가총액 정보 추가
            market_data['market_cap'] = 0
        
        # 5. 시계열 특성 추가
        try:
            # 5-1. 시간 관련 특성
            market_data['weekday'] = pd.to_datetime(market_data['date']).dt.weekday
            market_data['month'] = pd.to_datetime(market_data['date']).dt.month
            market_data['quarter'] = pd.to_datetime(market_data['date']).dt.quarter
            
            # 5-2. 블록딜 발생 패턴
            market_data['days_since_last_block_deal'] = market_data['has_block_deal'].cumsum()
            market_data['days_since_last_block_deal'] = market_data.groupby('days_since_last_block_deal').cumcount()
            
            # 5-3. 가격 추세
            market_data['price_trend'] = market_data['close_price'].pct_change(5)  # 5일 가격 추세
            market_data['volume_trend'] = market_data['volume'].pct_change(5)      # 5일 거래량 추세
        except Exception as e:
            logger.warning(f"시계열 특성 추가 오류: {e}")
        
        # 6. 결측치 처리
        market_data = market_data.ffill().fillna(0)
        
        # 7. 범주형 변수 처리
        try:
            categorical_cols = ['weekday', 'month', 'quarter']
            for col in categorical_cols:
                if col in market_data.columns:
                    dummies = pd.get_dummies(market_data[col], prefix=col)
                    market_data = pd.concat([market_data, dummies], axis=1)
                    market_data.drop(col, axis=1, inplace=True)
        except Exception as e:
            logger.warning(f"범주형 변수 처리 오류: {e}")
        
        # 8. 스케일링
        try:
            scaler = StandardScaler()
            numeric_cols = market_data.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col not in ['has_block_deal', 'days_since_last_block_deal']]
            
            if len(numeric_cols) > 0:
                # 결측치가 있으면 0으로 대체
                market_data[numeric_cols] = market_data[numeric_cols].fillna(0)
                
                # 모든 숫자 컬럼이 유효한 값인지 확인
                for col in numeric_cols:
                    # 무한값이나 NaN 값 처리
                    market_data[col] = pd.to_numeric(market_data[col], errors='coerce')
                    market_data[col] = market_data[col].replace([np.inf, -np.inf], 0)
                    market_data[col] = market_data[col].fillna(0)
                
                # 스케일링 적용
                scaled_features = scaler.fit_transform(market_data[numeric_cols].astype('float32'))
                scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols, index=market_data.index)
                
                # 스케일링된 열과 스케일링되지 않은 열 합치기
                non_numeric_cols = [col for col in market_data.columns if col not in numeric_cols]
                market_data = pd.concat([market_data[non_numeric_cols], scaled_df], axis=1)
        except Exception as e:
            logger.warning(f"스케일링 오류: {e}")
            scaler = None
        
        # 9. 학습 데이터와 타겟 분리
        try:
            # date 컬럼이 존재하는지 확인
            if 'date' not in market_data.columns:
                logger.error(f"최종 데이터에 'date' 컬럼이 없습니다.")
                return None, None, None
                
            X = market_data.drop(['has_block_deal', 'date'], axis=1)
            y = market_data['has_block_deal']
            
            logger.info(f"{ticker} 데이터 준비 완료: {X.shape}")
            return X, y, scaler
        except Exception as e:
            logger.error(f"학습 데이터 분리 오류: {e}")
            return None, None, None
    
    except Exception as e:
        logger.error(f"{ticker} 데이터 준비 오류: {e}")
        return None, None, None
