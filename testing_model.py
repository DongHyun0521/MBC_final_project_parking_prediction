import pandas as pd
import numpy as np 
import joblib
import os
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, max_error,
    explained_variance_score, median_absolute_error, mean_squared_log_error
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
    evs = explained_variance_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, np.maximum(y_pred, 0))
    rmsle = np.sqrt(msle)
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100

    print(f" ✅ R²: {r2:.4f} | Adj R²: {adj_r2:.4f} | EVS: {evs:.4f}")
    print(f" ✅ MAE: {mae:.2f}대 | MedAE: {medae:.2f}대 | RMSE: {rmse:.2f}대")
    print(f" ✅ MSE: {mse:.2f} | RMSLE: {rmsle:.4f} | sMAPE: {smape:.2f}% | MAX Err: {max_err:.1f}대")

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

    # 2) 중기용 데이터(일 단위 압축) 분할
    df_daily = df.groupby('date').agg({
        'month': 'first', 'dayofweek': 'first', 'is_holiday': 'first',
        'temp': 'mean', 'rainfall_mm': 'sum',
        '예약_내과': 'sum', '예약_정형외과': 'sum', '예약_소아청소년과': 'sum',
        '예약_이비인후과': 'sum', '예약_신경외과': 'sum', '예약_피부과': 'sum',
        '예약_안과': 'sum', '예약_치과': 'sum', '예약_정신건강의학과': 'sum',
        '예약_총외래환자': 'sum', '최종_주차대수': 'max'
    }).reset_index()
    X_mid = df_daily.drop(columns=['date', '최종_주차대수'])
    X_mid = pd.get_dummies(X_mid, dtype=int)
    y_mid = df_daily['최종_주차대수']
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
        ("🔴 중기 (일 단위 피크)", "mid", X_mid, y_mid, split_idx_mid)
    ]

    loaded_models = {}

    for name, m_type, X_data, y_data, s_idx in models_info:
        print(f"\n▶ {name} 모델 평가")
        model = joblib.load(os.path.join(base_path, f"parking_rf_{m_type}.pkl"))
        columns = joblib.load(os.path.join(base_path, f"columns_{m_type}.pkl"))
        loaded_models[m_type] = {"model": model, "columns": columns}
        
        X_test = X_data.iloc[s_idx:][columns] # 컬럼 순서 맞추기
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
    
    # 30분 단위 가중치 (오전 피크 집중)
    def get_30min_weight(h, m):
        time_val = h + (m / 60.0)
        if 9.0 <= time_val <= 11.0: return 0.50 / 5  # 9시~11시 (5개 슬롯)에 50% 몰림
        elif 13.5 <= time_val <= 15.5: return 0.30 / 5 # 오후 재개
        elif 16.0 <= time_val <= 17.5: return 0.10 / 4
        return 0.02

    vshort_rows = []
    for h in range(8, 19):
        for m in [0, 30]:
            row = {col: 0 for col in loaded_models['vshort']['columns']}
            row['month'], row['dayofweek'], row['hour'], row['minute'] = 3, 1, h, m
            row['is_holiday'] = 0
            # 비 오는 날씨 세팅
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
            # 200대 넘어가면 경고등(만차) 표시!
            alert = " 🚨만차!" if cars >= 200 else ""
            print(f" - {h:02d}:{m:02d} 예상: {cars:3d}대 | {bar}{alert}")
            idx += 1

    # ---------------------------------------------------------
    # UI 오른쪽: 중기 (향후 5일간의 '일일 최대 피크' 막대 그래프)
    # ---------------------------------------------------------
    print("\n📊 [UI 오른쪽] '중기 모델' 향후 5일간 일일 최대 주차 수요(Peak) 예측")
    
    mid_scenarios = [
        {"date": "3/25(수)", "day": 2, "hol": 0, "temp": 15.0, "rain": 0.0, "환자": 1300},
        {"date": "3/26(목)", "day": 3, "hol": 0, "temp": 18.0, "rain": 0.0, "환자": 1250},
        {"date": "3/27(금)", "day": 4, "hol": 0, "temp": 12.0, "rain": 20.0, "환자": 1400}, # 비 오는 금요일 헬게이트
        {"date": "3/28(토)", "day": 5, "hol": 0, "temp": 10.0, "rain": 0.0, "환자": 600},  # 오전 진료만
        {"date": "3/29(일)", "day": 6, "hol": 1, "temp": 14.0, "rain": 0.0, "환자": 0}    # 휴일
    ]
    
    # 각 진료과별 평균 배분 비율
    dept_ratio = {
        '예약_내과': 0.25, '예약_정형외과': 0.15, '예약_소아청소년과': 0.15,
        '예약_이비인후과': 0.12, '예약_신경외과': 0.08, '예약_피부과': 0.08,
        '예약_안과': 0.08, '예약_치과': 0.05, '예약_정신건강의학과': 0.04
    }
    
    mid_rows = []
    for sc in mid_scenarios:
        row = {col: 0 for col in loaded_models['mid']['columns']}
        row['month'], row['dayofweek'], row['is_holiday'] = 3, sc['day'], sc['hol']
        row['temp'], row['rainfall_mm'] = sc['temp'], sc['rain']
        
        # 🔥 [버그 픽스] 총 환자 수에 맞춰 각 진료과별 환자 수 정상 분배!
        row['예약_총외래환자'] = sc['환자']
        for dept, ratio in dept_ratio.items():
            if dept in row: row[dept] = int(sc['환자'] * ratio)
                
        mid_rows.append(row)
        
    df_mid_sim = pd.DataFrame(mid_rows)[loaded_models['mid']['columns']]
    mid_preds = loaded_models['mid']['model'].predict(df_mid_sim)
    
    for i, sc in enumerate(mid_scenarios):
        peak_cars = int(mid_preds[i])
        bar = "🟦" * int(peak_cars / 10)
        alert = " 🚨만차주의" if peak_cars >= 200 else ""
        print(f" [{sc['date']}] 최고 피크: {peak_cars:3d}대 | {bar}{alert}")
        if sc['rain'] > 0:
            print(f"    └ 🌧️ 날씨: 비({sc['rain']}mm), 예약: {sc['환자']}명")
        else:
            print(f"    └ ☀️ 날씨: 맑음, 예약: {sc['환자']}명")

if __name__ == "__main__":
    run_testing()