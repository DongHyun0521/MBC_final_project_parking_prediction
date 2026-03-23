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
    
    cols_to_drop_init = ['datetime', 'date', '최종_주차대수', 'minute']
    X_master = df.drop(columns=[c for c in cols_to_drop_init if c in df.columns])
    X_master = pd.get_dummies(X_master, dtype=int)
    y = df['최종_주차대수']

    # =========================================================
    # 🎯 [수정 2] 중기(mid) 모델을 위한 "일(Day) 단위" 데이터셋 생성
    # =========================================================
    print("📅 중기 예측용 '일(Day) 단위' 데이터 압축 중...")
    # 날짜별로 뭉치기 (예약은 다 더하고, 날씨는 평균/합계 내고, 주차대수는 하루 최고 피크치를 추출!)
    df_daily = df.groupby('date').agg({
        'month': 'first',
        'dayofweek': 'first',
        'is_holiday': 'first',
        'temp': 'mean',          # 하루 평균 기온
        'rainfall_mm': 'sum',    # 하루 총 강수량
        '예약_내과': 'sum',
        '예약_정형외과': 'sum',
        '예약_소아청소년과': 'sum',
        '예약_이비인후과': 'sum',
        '예약_신경외과': 'sum',
        '예약_피부과': 'sum',
        '예약_안과': 'sum',
        '예약_치과': 'sum',
        '예약_정신건강의학과': 'sum',
        '예약_총외래환자': 'sum', # 하루 총 예약 환자!
        '최종_주차대수': 'max'    # 프론트 막대그래프용: 그날의 '최대 주차 수요(피크)'
    }).reset_index()

    X_mid = df_daily.drop(columns=['date', '최종_주차대수'])
    X_mid = pd.get_dummies(X_mid, dtype=int)
    y_mid = df_daily['최종_주차대수']

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

    # 3. 중기 (3일~11일) - 일(Day) 단위
    print("🔴 [중기] 일(Day) 단위 거시적 모델 학습 중...")
    model_mid = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_mid.fit(X_mid, y_mid)
    joblib.dump(model_mid, os.path.join(base_path, "parking_rf_mid.pkl"))
    joblib.dump(X_mid.columns.tolist(), os.path.join(base_path, "columns_mid.pkl"))

    print("\n🎉 대성공! 프론트 UI(30분 선그래프 & 일별 막대그래프)에 완벽 대응하는 뇌 3개가 완성되었습니다!")

if __name__ == "__main__":
    train_three_models()