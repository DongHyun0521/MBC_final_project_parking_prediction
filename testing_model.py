import pandas as pd
import numpy as np 
import joblib
import os
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, max_error,
    median_absolute_error, mean_squared_log_error
)

def evaluate_metrics(y_true, y_pred, X_test):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    max_err = max_error(y_true, y_pred)
    
    n = len(y_true)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    medae = median_absolute_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, np.maximum(y_pred, 0))
    rmsle = np.sqrt(msle)
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
    mbe = np.mean(y_pred - y_true)
    tolerance = 20
    within = np.mean(np.abs(y_true - y_pred) <= tolerance) * 100

    # 적합도
    print(f" ✅ R²: {r2:.4f} | Adj R²: {adj_r2:.4f}")
    # 절대 오차 (단위: 대)
    print(f" ✅ MAE: {mae:.2f}대 | MedAE: {medae:.2f}대 | RMSE: {rmse:.2f}대")
    # 상대 오차 (단위 없음)
    print(f" ✅ RMSLE: {rmsle:.4f} | sMAPE: {smape:.2f}%")
    # 편향 / 극단값 / 실용 정확도
    print(f" ✅ MBE: {mbe:+.2f}대 | MAX Err: {max_err:.1f}대 | ±{tolerance}대 이내 정확도: {within:.1f}%")

def run_testing():
    base_path = "MBC_final_project_parking_prediction/features"
    file_path = os.path.join(base_path, "merged_features.csv")
    
    print("⏳ 테스트용 원본 데이터(Test Set)를 준비 중입니다...")
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek 
    df['minute'] = df['datetime'].dt.minute

    # 1) 초단기/단기용 데이터 분할
    cols_to_drop_init = ['datetime', 'date', '최종_주차대수']
    X_master = df.drop(columns=[c for c in cols_to_drop_init if c in df.columns])
    X_master = pd.get_dummies(X_master, dtype=int)
    y = df['최종_주차대수']
    split_idx = int(len(X_master) * 0.8)

    # 2) 중기용 데이터 분할 - 운영시간 30분 슬롯
    def is_operating(row):
        if row['is_holiday'] == 1 or row['dayofweek'] == 6:  # 일요일/공휴일
            return True
        elif row['dayofweek'] == 5:                           # 토요일: 08:00~13:00
            return 8 <= row['hour'] < 13
        else:                                                  # 평일: 08:00~19:00
            return 8 <= row['hour'] < 19

    df_mid = df[df.apply(is_operating, axis=1)].copy()

    # 중기 API는 일 단위 값만 제공 → 관련 피처를 일 단위로 통일
    df_mid['temp'] = df_mid.groupby('date')['temp'].transform('mean')
    df_mid['rainfall_mm'] = df_mid.groupby('date')['rainfall_mm'].transform('sum')
    if 'snowfall_cm' in df_mid.columns:
        df_mid['snowfall_cm'] = df_mid.groupby('date')['snowfall_cm'].transform('sum')

    # 중기 API가 제공하지 않는 피처 제거
    cols_to_drop_mid = ['datetime', 'date', '최종_주차대수',
                        'wind_speed', 'humidity', 'pm10', 'pm25', 'pm10_grade', 'pm25_grade']
    X_mid = df_mid.drop(columns=[c for c in cols_to_drop_mid if c in df_mid.columns])
    X_mid = pd.get_dummies(X_mid, dtype=int)
    y_mid = df_mid['최종_주차대수']
    split_idx_mid = int(len(X_mid) * 0.8)

    # =========================================================
    # 🎯 1. 모델 채점 및 중요도 평가 (Test Set)
    # =========================================================
    print("\n===========================================================")
    print(" 📊 [PART 1] 3개 모델 정밀 채점 (하위 20% 테스트 셋)")
    print("===========================================================")
    
    models_info = [
        ("🟢 초단기 (30분 단위)", "vshort", X_master.drop(columns=['pm10_grade', 'pm25_grade'], errors='ignore'), y, split_idx),
        ("🟡 단기 (30분 단위)", "short", X_master.drop(columns=['pm10', 'pm25'], errors='ignore'), y, split_idx),
        ("🔴 중기 (운영시간 30분 슬롯)", "mid", X_mid, y_mid, split_idx_mid)
    ]
    loaded_models = {}
    for name, m_type, X_data, y_data, s_idx in models_info:
        print(f"\n▶ {name} 모델 평가")
        model = joblib.load(os.path.join(base_path, f"parking_rf_{m_type}.pkl"))
        columns = joblib.load(os.path.join(base_path, f"columns_{m_type}.pkl"))
        loaded_models[m_type] = {"model": model, "columns": columns}
        
        X_test = X_data.iloc[s_idx:][columns]
        y_test = y_data.iloc[s_idx:]
        
        y_pred = model.predict(X_test)
        evaluate_metrics(y_test, y_pred, X_test)

    # =========================================================
    # 🔮 2. 프론트엔드 UI 맞춤형 가상 시뮬레이션
    # =========================================================
    print("\n===========================================================")
    print(" 🔮 [PART 2] 프론트엔드 UI 연동 시뮬레이션 (가상 데이터)")
    print("===========================================================")

    # ---------------------------------------------------------
    # UI 왼쪽: 초단기/단기 (하루 30분 단위 흐름 실선 그래프)
    # ---------------------------------------------------------
    print("\n📈 [UI 왼쪽] 3월 17일(화) '초단기 모델' 30분 단위 주차 흐름 (비 오는 날)")
    daily_patients = 1200
    
    def get_30min_weight(h, m):
        time_val = h + (m / 60.0)
        if 9.0 <= time_val <= 11.0: return 0.50 / 5
        elif 13.5 <= time_val <= 15.5: return 0.30 / 5
        elif 16.0 <= time_val <= 17.5: return 0.10 / 4
        return 0.02

    vshort_rows = []
    for h in range(8, 19):
        for m in [0, 30]:
            row = {col: 0 for col in loaded_models['vshort']['columns']}
            row['month'], row['dayofweek'], row['hour'], row['minute'] = 3, 1, h, m
            row['is_holiday'] = 0
            row['temp'], row['rainfall_mm'], row['wind_speed'], row['humidity'] = 12.0, 5.0, 3.5, 80
            row['pm10'], row['pm25'] = 30.0, 10.0
            row['예약_총외래환자'] = int(daily_patients * get_30min_weight(h, m))
            vshort_rows.append(row)
            
    df_vshort_sim = pd.DataFrame(vshort_rows)[loaded_models['vshort']['columns']]
    vshort_preds = loaded_models['vshort']['model'].predict(df_vshort_sim)
    
    idx = 0
    for h in range(8, 19):
        for m in [0, 30]:
            cars = int(vshort_preds[idx])
            bar = "█" * int(cars / 5)
            alert = " 🚨만차!" if cars >= 200 else ""
            print(f" - {h:02d}:{m:02d} 예상: {cars:3d}대 | {bar}{alert}")
            idx += 1

    # ---------------------------------------------------------
    # UI 오른쪽: 중기 (향후 5일간 운영시간 30분 슬롯 평균 막대 그래프)
    # ---------------------------------------------------------
    print("\n📊 [UI 오른쪽] '중기 모델' 향후 5일간 운영시간 평균 주차 수요 예측")
    
    mid_scenarios = [
        {"date": "3/25(수)", "day": 2, "hol": 0, "temp": 15.0, "rain": 0.0, "환자": 1300},
        {"date": "3/26(목)", "day": 3, "hol": 0, "temp": 18.0, "rain": 0.0, "환자": 1250},
        {"date": "3/27(금)", "day": 4, "hol": 0, "temp": 12.0, "rain": 20.0, "환자": 1400},
        {"date": "3/28(토)", "day": 5, "hol": 0, "temp": 10.0, "rain": 0.0, "환자": 600},
        {"date": "3/29(일)", "day": 6, "hol": 1, "temp": 14.0, "rain": 0.0, "환자": 0}
    ]
    
    dept_ratio = {
        '예약_내과': 0.25, '예약_정형외과': 0.15, '예약_소아청소년과': 0.15,
        '예약_이비인후과': 0.12, '예약_신경외과': 0.08, '예약_피부과': 0.08,
        '예약_안과': 0.08, '예약_치과': 0.05, '예약_정신건강의학과': 0.04
    }

    def get_operating_slots(dayofweek, is_holiday):
        slots = []
        if is_holiday == 1 or dayofweek == 6:
            for h in range(0, 24):
                for m in [0, 30]: slots.append((h, m))
        elif dayofweek == 5:                     # 토요일: 08:00~13:00
            for h in range(8, 13):
                for m in [0, 30]: slots.append((h, m))
        else:                                    # 평일: 08:00~19:00
            for h in range(8, 19):
                for m in [0, 30]: slots.append((h, m))
        return slots

    for sc in mid_scenarios:
        slots = get_operating_slots(sc['day'], sc['hol'])
        slot_preds = []
        for h, m in slots:
            row = {col: 0 for col in loaded_models['mid']['columns']}
            row['month'], row['dayofweek'], row['is_holiday'] = 3, sc['day'], sc['hol']
            row['temp'], row['rainfall_mm'] = sc['temp'], sc['rain']
            row['hour'], row['minute'] = h, m
            w = get_30min_weight(h, m)
            row['예약_총외래환자'] = int(sc['환자'] * w)
            for dept, ratio in dept_ratio.items():
                if dept in row: row[dept] = int(sc['환자'] * ratio * w)
            df_row = pd.DataFrame([row])[loaded_models['mid']['columns']]
            slot_preds.append(loaded_models['mid']['model'].predict(df_row)[0])
        avg_cars = int(np.mean(slot_preds))
        bar = "🟦" * int(avg_cars / 10)
        alert = " 🚨혼잡주의" if avg_cars >= 150 else ""
        print(f" [{sc['date']}] 운영시간 평균: {avg_cars:3d}대 | {bar}{alert}")
        if sc['rain'] > 0:
            print(f"    └ 🌧️ 날씨: 비({sc['rain']}mm), 예약: {sc['환자']}명")
        else:
            print(f"    └ ☀️ 날씨: 맑음, 예약: {sc['환자']}명")

if __name__ == "__main__":
    run_testing()
