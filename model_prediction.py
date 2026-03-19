import pandas as pd
import numpy as np # 💡 수학 계산을 위해 추가
import joblib 
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# 💡 검증 지표들을 우주 끝까지 몽땅 불러옵니다!
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, max_error,
    explained_variance_score, median_absolute_error, mean_squared_log_error
)

def train_and_predict():
    # 💡 1. 파일 경로 수정 (결측치 치료가 끝난 최종 V2 데이터)
    base_path = "MBC_final_project_parking_prediction/features"
    file_path = os.path.join(base_path, "merged_features_v2.csv")
    
    print("⏳ 결측치가 완벽히 치료된 최종 마스터 데이터를 불러오는 중...")
    df = pd.read_csv(file_path)

    # 1. 💡 [Feature Engineering] AI가 날짜를 이해할 수 있게 숫자 조각으로 쪼개주기
    print("⚙️ AI가 이해하기 쉽게 날짜 데이터를 월, 시간, 요일로 변환 중...")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek # 0:월, 1:화 ... 6:일
    
    # 2. 문제지(X)와 정답지(y) 분리
    target_col = '주차수요예측' 
    y = df[target_col]
    
    cols_to_drop = ['datetime', 'date', target_col, '예약_총외래환자', '예약_응급실방문']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 💡 [여기 추가!] AI는 글자를 모릅니다! 문자를 숫자(0과 1)로 변환 (원-핫 인코딩)
    print("🪄 '맑음/흐림' 같은 글자 데이터를 AI가 좋아하는 숫자로 번역 중...")
    X = pd.get_dummies(X, dtype=int)

    # 3. 💡 [시간순 분할] AI의 커닝을 완벽 차단!
    print("✂️ 데이터를 시간순으로(과거 80% 학습, 최신 20% 테스트) 엄격하게 분할합니다...")
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 4. 머신러닝 모델 생성 및 학습 (Random Forest)
    print("🧠 AI (랜덤 포레스트)가 과거 10년 치 데이터를 미친 듯이 학습하고 있습니다... (1~2분 소요될 수 있음)")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 💡 4-1. 완성된 뇌(AI)를 파일로 저장!
    model_save_path = os.path.join(base_path, "parking_rf_model.pkl")
    columns_save_path = os.path.join(base_path, "model_columns.pkl")
    joblib.dump(model, model_save_path)
    joblib.dump(X.columns.tolist(), columns_save_path)
    print(f"💾 똑똑해진 AI 모델이 '{model_save_path}'에 영구 보존되었습니다!")

    # =========================================================
    # 💡 5. 풀옵션 10대 채점표 확인 (테스트 세트로 예측)
    # =========================================================
    print("\n===========================================")
    print("📊 [AI 모델 다각도 채점 결과 (초정밀 10대 검증)]")
    y_pred = model.predict(X_test)
    
    # 1. 기본 회귀 지표
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    max_err = max_error(y_test, y_pred)
    
    # 2. 고급 회귀 지표 추가!
    # 2-1. Adjusted R² (수정된 결정계수: 쓸데없는 피처에 페널티 부여)
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # 2-2. Explained Variance Score (설명 분산 점수)
    evs = explained_variance_score(y_test, y_pred)
    
    # 2-3. MedAE (중앙값 절대 오차: 이상치에 덜 민감함)
    medae = median_absolute_error(y_test, y_pred)
    
    # 2-4. RMSLE (루트 평균 제곱 로그 오차: 마이너스 값이 없도록 np.maximum 처리)
    msle = mean_squared_log_error(y_test, np.maximum(y_pred, 0))
    rmsle = np.sqrt(msle)
    
    # 2-5. sMAPE (대칭 평균 절대 비율 오차: 아까 그 4경 퍼센트 에러를 막는 완벽한 비율 공식)
    smape = np.mean(2.0 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred) + 1e-10)) * 100

    print(f"✅ 1. R²       (결정계수) : {r2:.4f} (1에 가까울수록 완벽)")
    print(f"✅ 2. Adj R²   (수정된 R²): {adj_r2:.4f} (변수 개수까지 고려한 진짜 설명력)")
    print(f"✅ 3. EVS      (설명분산) : {evs:.4f} (R²와 비슷하나, 오차의 평균 이동을 잡아냄)")
    print(f"✅ 4. MAE      (평균오차) : {mae:.2f} 대 (가장 직관적인 평균 오차)")
    print(f"✅ 5. MedAE    (중앙오차) : {medae:.2f} 대 (극단적으로 틀린 값을 제외한 현실적 오차)")
    print(f"✅ 6. MSE      (제곱오차) : {mse:.2f} (큰 오차에 페널티를 준 분산)")
    print(f"✅ 7. RMSE     (루트오차) : {rmse:.2f} 대 (실제 체감되는 변동성 오차)")
    print(f"✅ 8. RMSLE    (로그오차) : {rmsle:.4f} (값의 '크기'보다 '비율'을 중시하는 오차)")
    print(f"✅ 9. sMAPE    (대칭비율) : {smape:.2f} % (0대 예측 시 폭주하는 MAPE를 0~200% 사이로 안정화)")
    print(f"✅ 10. MAX     (최대오차) : {max_err:.1f} 대 (가장 운 나쁘게 크게 틀렸을 때)")
    print("===========================================\n")

    # =========================================================
    # 🔍 6. AI의 속마음 엿보기: 어떤 데이터가 가장 중요했을까? (숫자로 출력!)
    # =========================================================
    print("📊 [AI가 생각하는 가장 중요한 요인 TOP 15 (가중치 %)]")
    
    importances = model.feature_importances_
    
    feature_imp_df = pd.DataFrame({
        '데이터 종류 (Feature)': X.columns,
        '가중치 (%)': importances * 100 # 퍼센트로 보기 좋게 변환
    }).sort_values(by='가중치 (%)', ascending=False)

    feature_imp_df = feature_imp_df.reset_index(drop=True)
    pd.options.display.float_format = '{:.2f}%'.format
    
    print(feature_imp_df.head(15).to_string())
    print("===========================================\n")

    # =========================================================
    # 🔮 7. 대망의 2026년 3월 17일(화) '하루 전체' 가상 시나리오 예측!
    # =========================================================
    print("🔮 [2026년 3월 17일(화) 일일 시간대별 & 총 주차 수요 예측]")
    
    daily_patients = {
        '예약_내과': 60, '예약_정형외과': 40, '예약_소아청소년과': 30,
        '예약_이비인후과': 25, '예약_신경외과': 20, '예약_피부과': 15,
        '예약_안과': 15, '예약_치과': 15, '예약_정신건강의학과': 10
    }

    def get_hour_weight(h):
        if 9 <= h <= 10: return 0.25 / 2       
        elif 11 <= h <= 12: return 0.15 / 2      
        elif h == 13: return 0.05                
        elif 14 <= h <= 15: return 0.25 / 2      
        elif 16 <= h <= 17: return 0.20 / 2      
        elif h == 18: return 0.10                
        else: return 0.0                         

    future_rows = []
    for h in range(24):
        row = {col: 0 for col in X.columns}
        
        if 'month' in row: row['month'] = 3
        if 'dayofweek' in row: row['dayofweek'] = 1 
        if 'hour' in row: row['hour'] = h
        if 'temp' in row: row['temp'] = 15.0
        if 'rainfall_mm' in row: row['rainfall_mm'] = 0.0
        if 'PM10' in row: row['PM10'] = 45.0
        
        weather_cols = [c for c in X.columns if '맑음' in c]
        if weather_cols: row[weather_cols[0]] = 1
        
        hw = get_hour_weight(h)
        for dept, total_patients in daily_patients.items():
            if dept in row:
                row[dept] = int(total_patients * hw)
                
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    hourly_preds = model.predict(future_df)

    total_daily_cars = int(sum(hourly_preds)) 
    peak_hour = int(hourly_preds.argmax())    
    peak_cars = int(hourly_preds.max())       

    print(f"\n🚙 [2026년 3월 17일(화) 주차장 운영 리포트]")
    print(f"▶ 하루 총 방문 예상 차량: 약 [ {total_daily_cars} 대 ]")
    print(f"▶ 만차 주의(Peak Time) : [ {peak_hour:02d}시 ~ {peak_hour+1:02d}시 ] (동시 주차 약 {peak_cars}대 예상)")
    print("-" * 45)
    print("⏰ [주요 시간대별 주차 수요 흐름]")
    for h in range(8, 19): 
        bar = "█" * int(hourly_preds[h] / 2) 
        print(f" - {h:02d}시: 예상 {int(hourly_preds[h]):2d}대 | {bar}")
    print("===========================================\n")

if __name__ == "__main__":
    train_and_predict()