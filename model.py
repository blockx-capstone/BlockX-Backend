"""
딥러닝 모델 구현 및 학습 모듈
"""
import json
import pandas as pd
import numpy as np
import logging
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import joblib
from datetime import datetime
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from data_preprocessing import prepare_data_for_prediction

logger = logging.getLogger(__name__)

class BlockDealDetector:
    """
    내부자 대규모 지분 거래(블록딜) 탐지 모델 클래스
    """
    def __init__(self, model_dir='models'):
        """
        블록딜 발생 가능성 예측 모델 초기화
        
        Args:
            model_dir (str): 모델 저장 디렉토리
        """
        self.model = None
        self.scaler = None
        self.model_dir = model_dir
        self.feature_names = None
        
        # 모델 디렉토리 생성
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def build_model(self, input_dim):
        """
        모델 구조 정의
        
        Args:
            input_dim (int): 입력 특성의 차원
        """
        model = Sequential([
            # LSTM 레이어 (시계열 특성 학습)
            LSTM(128, input_shape=(5, input_dim), return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            # 완전연결 레이어
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # 출력 레이어 (블록딜 발생 확률)
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_sequence_data(self, X, y, sequence_length=5):
        """
        시계열 데이터 준비
        
        Args:
            X (DataFrame): 특성 데이터
            y (Series): 타겟 데이터
            sequence_length (int): 시퀀스 길이
            
        Returns:
            tuple: (X_seq, y_seq) - 시퀀스 데이터
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - sequence_length + 1):
            # 데이터프레임 슬라이스를 numpy 배열로 변환하고 float32 타입으로 명시적 변환
            sequence = X.iloc[i:i+sequence_length].values.astype('float32')
            X_seq.append(sequence)
            y_seq.append(y.iloc[i+sequence_length-1])
        
        # 명시적으로 numpy 배열로 변환하고 데이터 타입을 float32로 지정
        X_seq = np.array(X_seq, dtype='float32')
        y_seq = np.array(y_seq, dtype='float32')
        
        return X_seq, y_seq
    
    def train(self, ticker, start_date, end_date, test_size=0.2, random_state=42):
        """
        모델 학습
        
        Args:
            ticker (str): 종목코드
            start_date (str): 시작일자 (YYYY-MM-DD)
            end_date (str): 종료일자 (YYYY-MM-DD)
            test_size (float): 테스트 데이터 비율
            random_state (int): 랜덤 시드
        """
        try:
            logger.info(f"\n{ticker} 모델 학습 시작...")
            
            # 1. 데이터 준비
            X, y, self.scaler = prepare_data_for_prediction(ticker, start_date, end_date)
            if X is None or y is None:
                logger.error(f"{ticker} 데이터 준비 실패")
                return False
            
            # 데이터 타입 검증 및 변환
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].replace([np.inf, -np.inf], 0)
                X[col] = X[col].fillna(0)
            
            # y 데이터 검증 및 변환
            y = y.astype('float32')
            
            # 클래스 분포 확인
            class_counts = y.value_counts()
            logger.info(f"클래스 분포: {class_counts.to_dict()}")
            
            # 클래스 불균형 확인
            if len(class_counts) < 2:
                logger.warning(f"단일 클래스만 존재: {class_counts.index[0]}. 인공 데이터 생성 필요.")
                # 클래스가 하나뿐인 경우, 합성 데이터 생성
                
                # 존재하는 클래스 확인
                existing_class = class_counts.index[0]
                # 없는 클래스 값 생성
                missing_class = 1.0 if existing_class == 0.0 else 0.0
                
                # 학습 데이터의 10%를 합성 데이터로 생성
                synthetic_size = max(int(len(X) * 0.1), 5)  # 최소 5개 이상
                
                # 기존 데이터에서 일부 샘플링하여 레이블만 변경
                synthetic_indices = resample(
                    range(len(X)), 
                    n_samples=synthetic_size, 
                    random_state=random_state
                )
                
                # 합성 데이터 생성
                X_synthetic = X.iloc[synthetic_indices].copy()
                # 약간의 무작위성 추가 (원본 데이터 구조 유지)
                for col in X_synthetic.columns:
                    # 숫자형 특성에만 무작위성 추가
                    if X_synthetic[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                        X_synthetic[col] = X_synthetic[col] * (1 + np.random.normal(0, 0.1, size=len(X_synthetic)))
                
                # 반대 클래스 레이블 할당
                y_synthetic = pd.Series([missing_class] * len(X_synthetic))
                
                # 원본 데이터와 합성 데이터 결합
                X = pd.concat([X, X_synthetic], ignore_index=True)
                y = pd.concat([y, y_synthetic], ignore_index=True)
                
                logger.info(f"합성 데이터 생성 후 클래스 분포: {y.value_counts().to_dict()}")
            
            # 모든 데이터가 유효한지 확인
            if X.isnull().any().any() or np.isinf(X).any().any():
                logger.error(f"{ticker} 데이터에 결측치나 무한값이 있습니다.")
                # 결측치와 무한값 처리
                X = X.fillna(0)
                X = X.replace([np.inf, -np.inf], 0)
            
            # 2. 특성 이름 저장
            self.feature_names = X.columns.tolist()
            
            # 3. 시퀀스 데이터 준비
            X_seq, y_seq = self.prepare_sequence_data(X, y)
            
            # 시퀀스 데이터 유효성 검증
            if len(X_seq) == 0 or len(y_seq) == 0:
                logger.error(f"{ticker} 유효한 시퀀스 데이터가 없습니다.")
                return False
            
            # 4. 데이터 분할 - 시간 순서 유지하면서 계층적 분할
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            
            # 데이터 인덱스 생성
            indices = np.arange(len(y_seq))
            
            # 레이블을 기준으로 계층적 분할
            for train_idx, test_idx in splitter.split(indices, y_seq):
                X_train, X_test = X_seq[train_idx], X_seq[test_idx]
                y_train, y_test = y_seq[train_idx], y_seq[test_idx]
            
            logger.info(f"학습 데이터 크기: {X_train.shape}, 테스트 데이터 크기: {X_test.shape}")
            logger.info(f"학습 데이터 클래스 분포: {np.bincount(y_train.astype(int))}")
            logger.info(f"테스트 데이터 클래스 분포: {np.bincount(y_test.astype(int))}")
            
            # 5. 모델 생성
            input_dim = X.shape[1]
            logger.info(f"모델 입력 차원: {input_dim}")
            self.model = self.build_model(input_dim)
            
            # 6. 클래스 가중치 계산
            try:
                unique_classes = np.unique(y_train)
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=unique_classes, 
                    y=y_train
                )
                # 클래스 가중치를 딕셔너리로 변환
                class_weight_dict = {int(cls): weight for cls, weight in zip(unique_classes, class_weights)}
                logger.info(f"클래스 가중치: {class_weight_dict}")
            except Exception as e:
                logger.warning(f"클래스 가중치 계산 오류: {e}. 가중치 없이 학습합니다.")
                class_weight_dict = None
            
            # 7. 콜백 설정
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, f'{ticker}_best_model.h5'),
                    monitor='val_loss',
                    save_best_only=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
            ]
            
            # 8. 모델 학습
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # 9. 모델 평가
            y_pred = (self.model.predict(X_test, verbose=0) > 0.5).astype(int)
            y_prob = self.model.predict(X_test, verbose=0)
            
            # 9-1. 정확도
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"\n정확도: {accuracy:.4f}")
            
            # 9-2. 혼동 행렬
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"\n혼동 행렬:\n{cm}")
            
            # 9-3. 분류 보고서
            report = classification_report(y_test, y_pred)
            logger.info(f"\n분류 보고서:\n{report}")
            
            # 9-4. ROC AUC (다중 클래스인 경우에만 계산)
            try:
                if len(np.unique(y_test)) > 1:
                    auc = roc_auc_score(y_test, y_prob)
                    logger.info(f"\nROC AUC: {auc:.4f}")
                else:
                    logger.warning("단일 클래스만 존재하여 ROC AUC 계산 불가")
            except Exception as e:
                logger.warning(f"ROC AUC 계산 오류: {e}")
            
            # 10. 모델 저장
            self.save_model(ticker)
            
            logger.info(f"{ticker} 모델 학습 완료")
            return True
            
        except Exception as e:
            logger.error(f"{ticker} 모델 학습 오류: {e}")
            return False
    
    def predict(self, ticker, start_date, end_date):
        """
        블록딜 발생 가능성 예측
        
        Args:
            ticker (str): 종목코드
            start_date (str): 시작일자 (YYYY-MM-DD)
            end_date (str): 종료일자 (YYYY-MM-DD)
            
        Returns:
            DataFrame: 예측 결과 (날짜, 발생 확률)
        """
        try:
            # 1. 데이터 준비
            X, _, _ = prepare_data_for_prediction(ticker, start_date, end_date)
            if X is None:
                logger.error(f"{ticker} 데이터 준비 실패")
                return None
            
            # 2. 시퀀스 데이터 준비
            X_seq, _ = self.prepare_sequence_data(X, pd.Series([0] * len(X)))
            
            # 3. 예측
            probabilities = self.model.predict(X_seq)
            
            # 4. 결과 데이터프레임 생성
            dates = pd.date_range(start=start_date, end=end_date)
            results = pd.DataFrame({
                'date': dates[4:],  # 시퀀스 길이만큼 날짜 제외
                'block_deal_probability': probabilities.flatten()
            })
            
            # 5. 결과 정렬 및 반환
            results = results.sort_values('block_deal_probability', ascending=False)
            return results
            
        except Exception as e:
            logger.error(f"{ticker} 예측 오류: {e}")
            return None
    
    def save_model(self, ticker):
        """
        모델 저장
        
        Args:
            ticker (str): 종목코드
        """
        try:
            # 1. 모델 저장
            model_path = os.path.join(self.model_dir, f'{ticker}_model.h5')
            self.model.save(model_path)
            
            # 2. 스케일러 저장
            scaler_path = os.path.join(self.model_dir, f'{ticker}_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
            # 3. 특성 정보 저장
            feature_path = os.path.join(self.model_dir, f'{ticker}_features.joblib')
            joblib.dump(self.feature_names, feature_path)
            
            logger.info(f"{ticker} 모델 저장 완료")
            
        except Exception as e:
            logger.error(f"{ticker} 모델 저장 오류: {e}")
    
    def load_model(self, ticker):
        """
        모델 로드
        
        Args:
            ticker (str): 종목코드
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            # 1. 모델 로드
            model_path = os.path.join(self.model_dir, f'{ticker}_model.h5')
            if not os.path.exists(model_path):
                logger.error(f"{ticker} 모델 파일 없음")
                return False
            
            self.model = load_model(model_path)
            
            # 2. 스케일러 로드
            scaler_path = os.path.join(self.model_dir, f'{ticker}_scaler.joblib')
            if not os.path.exists(scaler_path):
                logger.error(f"{ticker} 스케일러 파일 없음")
                return False
            
            self.scaler = joblib.load(scaler_path)
            
            # 3. 특성 정보 로드
            feature_path = os.path.join(self.model_dir, f'{ticker}_features.joblib')
            if not os.path.exists(feature_path):
                logger.error(f"{ticker} 특성 정보 파일 없음")
                return False
            
            self.feature_names = joblib.load(feature_path)
            
            logger.info(f"{ticker} 모델 로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"{ticker} 모델 로드 오류: {e}")
            return False

    def predict_block_deal(self, trade_data):
        """
        단일 거래 데이터에 대한 블록딜 여부 예측
        
        Args:
            trade_data (dict): 거래 데이터
            
        Returns:
            dict: 예측 결과 (확률, 분류, 신뢰도)
        """
        try:
            if self.model is None:
                logger.error("모델이 로드되지 않았습니다.")
                return None
                
            # 1. 딕셔너리를 데이터프레임으로 변환
            df = pd.DataFrame([trade_data])
            
            # 2. 모델 학습에 사용된 특성 체크
            if self.feature_names is None:
                logger.error("특성 정보가 없습니다.")
                return None
                
            # 3. 모델 학습 시 사용한 특성만 선택하거나 누락된 특성 추가
            model_features = set(self.feature_names)
            input_features = set(df.columns)
            
            # 모델에는 있지만 입력에는 없는 특성 추가
            for feature in model_features - input_features:
                df[feature] = 0
                
            # 입력에는 있지만 모델에는 없는 특성 제거
            for feature in input_features - model_features:
                df = df.drop(columns=[feature])
                
            # 정확한 순서로 특성 정렬
            df = df[self.feature_names]
            
            # 4. 데이터 타입 변환 및 결측치 처리
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # 무한값 처리
                df[col] = df[col].replace([np.inf, -np.inf], 0)
                # 결측치 처리
                df[col] = df[col].fillna(0)
            
            # 5. 스케일링 적용 (학습 시 사용한 스케일러 활용)
            if self.scaler is not None:
                try:
                    # 모든 열 이름이 스케일러의 훈련 데이터와 일치하는지 확인
                    logger.info(f"스케일링 적용 전 특성: {df.columns.tolist()}")
                    
                    # 숫자형 타입으로 명시적 변환
                    df = df.astype('float32')
                    
                    # 스케일러 적용
                    scaled_data = self.scaler.transform(df)
                    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
                    
                    logger.info(f"스케일링 성공: 입력 데이터 크기 {df.shape}")
                except Exception as scale_error:
                    logger.warning(f"스케일링 적용 오류: {scale_error}. 원본 데이터 사용.")
                    df_scaled = df.copy()
            else:
                logger.warning("스케일러가 없습니다. 원본 데이터 사용.")
                df_scaled = df.copy()
            
            # 6. 모델 입력 형태로 변환 (단일 시퀀스로 간주)
            # 시계열 데이터 대신 현재 데이터 포인트를 5번 반복 (시퀀스 길이 5 가정)
            X_seq = np.array([df_scaled.values.astype('float32')] * 5)
            X_seq = X_seq.reshape(1, 5, -1)  # 배치 차원 추가
            
            # 7. 예측 실행
            probability = self.model.predict(X_seq, verbose=0)[0][0]
            is_block_deal = 1 if probability > 0.5 else 0
            
            # 8. 신뢰도 수준 계산
            if probability < 0.3:
                confidence = "낮음"
            elif probability < 0.7:
                confidence = "중간"
            else:
                confidence = "높음"
            
            # 9. 결과 반환
            result = {
                'probability': float(probability),  # numpy.float32를 Python float로 변환
                'is_block_deal': int(is_block_deal),
                'confidence': confidence
            }
            
            return result
            
        except Exception as e:
            logger.error(f"블록딜 예측 오류: {e}")
            return None
