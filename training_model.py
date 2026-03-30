import pandas as pd
import numpy as np 
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def train_three_models():
    base_path = "MBC_final_project_parking_prediction/features"
    file_path = os.path.join(base_path, "merged_features.csv")
    
    print("⏳ 마스터 데이터 로딩 중...")
    df = pd.read_csv(file_path)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek 
    # 🔥 [수정 1] minute(분) 데이터 살려두기! (30분 단위 예측용)
    df['minute'] = df['datetime'].dt.minute
    
    cols_to_drop_init = ['datetime', 'date', 'minute', '최종_주차대수']
    X_master = df.drop(columns=[c for c in cols_to_drop_init if c in df.columns])
    X_master = pd.get_dummies(X_master, dtype=int)
    y = df['최종_주차대수']

    # =========================================================
    # 🎯 [수정 2] 중기(mid) 모델을 위한 "운영시간 30분 슬롯" 데이터셋 생성
    # =========================================================
    print("📅 중기 예측용 데이터 (운영시간 30분 슬롯) 준비 중...")

    # 요일별 운영시간 필터
    def is_operating(row):
        if row['is_holiday'] == 1 or row['dayofweek'] == 6:  # 일요일/공휴일: 24시간
            return True
        elif row['dayofweek'] == 5:                           # 토요일: 08:00~13:00
            return 8 <= row['hour'] < 13
        else:                                                  # 평일: 08:00~19:00
            return 8 <= row['hour'] < 19

    df_mid = df[df.apply(is_operating, axis=1)].copy()

    # 중기 API는 일 단위 값만 제공 → 관련 피처를 일 단위로 통일 (예측과 일관성 확보)
    df_mid['temp'] = df_mid.groupby('date')['temp'].transform('mean')               # 일 평균 기온
    df_mid['rainfall_mm'] = df_mid.groupby('date')['rainfall_mm'].transform('sum')  # 일 합산 강수량
    if 'snowfall_cm' in df_mid.columns:
        df_mid['snowfall_cm'] = df_mid.groupby('date')['snowfall_cm'].transform('sum')  # 일 합산 적설량

    # 중기 API가 제공하지 않는 피처 제거 (예측 시 항상 기본값이 들어가므로 학습에서 제외)
    cols_to_drop_mid = ['datetime', 'date', '최종_주차대수',
                        'wind_speed', 'humidity', 'pm10', 'pm25', 'pm10_grade', 'pm25_grade']

    X_mid = df_mid.drop(columns=[c for c in cols_to_drop_mid if c in df_mid.columns])
    X_mid = pd.get_dummies(X_mid, dtype=int)
    y_mid = df_mid['최종_주차대수']

    # =========================================================
    # 🚀 모델 3개 학습 (각각 다른 데이터 사용)
    # =========================================================
    print("\n🚀 [AI 공장 가동] 3개의 앙상블 모델 학습을 시작합니다...\n")

    # 1. 초단기 (0~6시간) - 30분 단위 (분 단위 살리고 등급 가림)
    print("🟢 [초단기] 30분 단위 정밀 모델 학습 중...")
    X_vshort = X_master.drop(columns=['pm10_grade', 'pm25_grade'])
    model_vshort = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_vshort.fit(X_vshort, y)
    joblib.dump(model_vshort, os.path.join(base_path, "parking_rf_vshort.pkl"))
    joblib.dump(X_vshort.columns.tolist(), os.path.join(base_path, "columns_vshort.pkl"))

    # 2. 단기 (6시간~3일) - 30분 단위 (수치 가림)
    print("🟡 [단기] 30분 단위 예측 모델 학습 중...")
    X_short = X_master.drop(columns=['pm10', 'pm25'])
    model_short = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_short.fit(X_short, y)
    joblib.dump(model_short, os.path.join(base_path, "parking_rf_short.pkl"))
    joblib.dump(X_short.columns.tolist(), os.path.join(base_path, "columns_short.pkl"))

    # 3. 중기 (3일~11일) - 운영시간 30분 슬롯 단위
    print("🔴 [중기] 운영시간 30분 슬롯 단위 모델 학습 중...")
    model_mid = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_mid.fit(X_mid, y_mid)
    joblib.dump(model_mid, os.path.join(base_path, "parking_rf_mid.pkl"))
    joblib.dump(X_mid.columns.tolist(), os.path.join(base_path, "columns_mid.pkl"))

    print("\n🎉 대성공! 프론트 UI(30분 선그래프 & 일별 막대그래프)에 완벽 대응하는 뇌 3개가 완성되었습니다!")

if __name__ == "__main__":
    train_three_models()
