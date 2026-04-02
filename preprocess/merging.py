import pandas as pd
import numpy as np
import os

def generate_final_parking_data():
    print("🚗 [STEP 1] 4개 원본 데이터 로드 및 병합 시작...")
    base_path = "MBC_final_project_parking_prediction/features"
    
    # 1. 파일 불러오기
    df_res = pd.read_csv(os.path.join(base_path, 'reservation/reservation_2015_2025.csv'))
    # ================================
    res_cols = [c for c in df_res.columns if '예약' in c]
    df_res[res_cols] = df_res[res_cols].rolling(window=2, min_periods=1).mean().astype(int)
    # ================================

    df_weather = pd.read_csv(os.path.join(base_path, 'weather/weather_2015_2025.csv'))
    df_air = pd.read_csv(os.path.join(base_path, 'air/air_2015_2025.csv'))
    df_hol = pd.read_csv(os.path.join(base_path, 'holiday/holiday_2015_2025.csv'))
    
    # 2. 날짜/시간 병합
    df_res['datetime'] = pd.to_datetime(df_res['datetime'])
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_air['datetime'] = pd.to_datetime(df_air['datetime'])
    
    df = pd.merge(df_res, df_weather, on='datetime', how='left')
    df = pd.merge(df, df_air, on='datetime', how='left')
    
    df_hol['date'] = pd.to_datetime(df_hol['date']).dt.strftime('%Y-%m-%d')
    df = pd.merge(df, df_hol[['date', 'holiday_name']], on='date', how='left')
    df['holiday_name'] = df['holiday_name'].fillna('')
    
    # 3. 결측치 방어 및 시간/요일 추출
    df['rainfall_mm'] = df['rainfall_mm'].fillna(0.0)
    df['snowfall_cm'] = df['snowfall_cm'].fillna(0.0)
    df['wind_speed'] = df['wind_speed'].fillna(0.0)
    df['humidity'] = df['humidity'].fillna(50.0)
    df['pm10'] = df['pm10'].fillna(30.0)
    df['pm25'] = df['pm25'].fillna(15.0)
    
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['weekday'] = df['datetime'].dt.dayofweek 
    df['time_val'] = df['hour'] + (df['minute'] / 60.0)
    
    df.loc[df['weekday'] == 6, 'is_holiday'] = 1
    
    np.random.seed(42)

    # ==============================================================================
    # 🎭 [STEP 2] 내부 계산용: 날씨/대기질 악화로 인한 '노쇼' 및 '현장 방문객' 증감
    # ==============================================================================
    print("🚶‍♂️ [STEP 2] 노쇼 및 현장 접수 시뮬레이션 중...")
    no_show_rate = np.random.uniform(0.05, 0.10, len(df))
    
    # 날씨 최악일 때 노쇼 페널티 대폭 강화! (AI가 날씨를 보게 만듦)
    bad_weather_noshow = (df['rainfall_mm'] > 5.0) | (df['snowfall_cm'] > 2.0) | (df['wind_speed'] > 10.0) | (df['pm25'] > 75)
    no_show_rate[bad_weather_noshow] += np.random.uniform(0.15, 0.25, size=bad_weather_noshow.sum())
    실제_방문_예약환자 = (df['예약_총외래환자'] * (1 - no_show_rate)).astype(int)

    # 🔥 [수정] 현장접수를 예약환자 비율이 아닌 '독립 난수'로 변경
    현장접수_환자 = np.random.randint(5, 25, len(df))
    # 날씨나 대기질이 나쁘면 현장 환자 70% 증발
    bad_weather_walkin = (df['rainfall_mm'] > 2.0) | (df['pm10'] > 80) | (df['temp'] < 0)
    현장접수_환자[bad_weather_walkin] = (현장접수_환자[bad_weather_walkin] * 0.3).astype(int)
    # 진료 없는 시간(밤/새벽) 현장접수 0 처리
    현장접수_환자[(df['time_val'] < 9.0) | (df['time_val'] > 18.0) | (df['is_holiday'] == 1)] = 0
    
    총_실제방문환자 = 실제_방문_예약환자 + 현장접수_환자

    # ==============================================================================
    # 🅿️ [STEP 3] 자차 이용률 수학적 산출 (모든 기상/대기질 피처 반영)
    # ==============================================================================
    print("🅿️ [STEP 3] 기상 및 대기질을 반영한 자차 이용률 산출 중...")
    # 🔥 [수정] 맑은 날 기본 자차 이용률을 낮추고, 악천후 시 확 올려서 갭을 키움
    df['자차_이용률'] = 0.25 
    
    df.loc[df['rainfall_mm'] > 0.0, '자차_이용률'] = 0.45
    df.loc[df['rainfall_mm'] > 5.0, '자차_이용률'] = 0.65 # 비 많이 오면 엄청 탐
    df.loc[df['snowfall_cm'] > 1.0, '자차_이용률'] += 0.10
    
    extreme_temp = (df['temp'] < -5) | (df['temp'] > 32)
    df.loc[extreme_temp, '자차_이용률'] += 0.15 # 너무 춥거나 더우면 무조건 차 탐

    df.loc[df['wind_speed'] > 10.0, '자차_이용률'] += 0.05
    unpleasant_weather = (df['temp'] >= 25) & (df['humidity'] >= 80)
    df.loc[unpleasant_weather, '자차_이용률'] += 0.05

    bad_air = (df['pm10'] > 80) | (df['pm25'] > 35)
    df.loc[bad_air, '자차_이용률'] += 0.05
    
    very_bad_air = (df['pm10'] > 150) | (df['pm25'] > 75)
    df.loc[very_bad_air, '자차_이용률'] += 0.05 

    df['자차_이용률'] = df['자차_이용률'].clip(upper=0.85)

    # ==============================================================================
    # 🚨 [STEP 3.5] 미세먼지 비상저감조치 (배출가스 5등급 차량 서울 진입 통제)
    # ==============================================================================
    print("🚨 [STEP 3.5] 비상저감조치(5등급 차량 운행제한) 시뮬레이션 적용 중...")
    df['도착예정_차량'] = (총_실제방문환자 * df['자차_이용률']).astype(int)
    df.loc[very_bad_air, '도착예정_차량'] = (df.loc[very_bad_air, '도착예정_차량'] * 0.90).astype(int) # 페널티 강화

    # ==============================================================================
    # 🚑 [STEP 3.8] 응급실 및 입원환자 면회객 주차 수요 (예약과 무관한 독립 수요!)
    # ==============================================================================
    print("🚑 [STEP 3.8] 응급실 및 면회객 주차 수요 계산 중...")
    def get_er_base(row):
        h, w, hol = row['hour'], row['weekday'], row['is_holiday']
        hol_name = str(row['holiday_name'])
        
        night_multiplier = 1.5 if (h >= 20 or h <= 8) else 1.0
        weekend_multiplier = 1.8 if (w in [5, 6] or hol == 1) else 1.0
        if '설날' in hol_name or '추석' in hol_name: weekend_multiplier = 0.7
        return 2.5 * night_multiplier * weekend_multiplier

    df['er_base'] = df.apply(get_er_base, axis=1)
    df['응급실_차량'] = np.maximum(0, np.random.normal(df['er_base'], 1)).astype(int)

    # 🔥 [추가] 입원환자 면회객 차량 (예약과 완전 무관)
    df['면회객_차량'] = 0
    # 평일 저녁 면회 (18:00 ~ 20:00)
    idx_wk_visit = (df['weekday'] < 5) & (df['is_holiday'] == 0) & (df['time_val'] >= 18.0) & (df['time_val'] <= 20.0)
    df.loc[idx_wk_visit, '면회객_차량'] = np.random.randint(15, 35, sum(idx_wk_visit))
    # 주말/휴일 낮, 저녁 면회
    idx_we_visit = ((df['weekday'] >= 5) | (df['is_holiday'] == 1)) & ((df['time_val'] >= 10.0) & (df['time_val'] <= 14.0) | (df['time_val'] >= 18.0) & (df['time_val'] <= 20.0))
    df.loc[idx_we_visit, '면회객_차량'] = np.random.randint(25, 60, sum(idx_we_visit))
    
    # 궂은 날씨엔 면회객 급감 (환경 변수 가중치 상승 요인)
    df.loc[bad_weather_walkin, '면회객_차량'] = (df['면회객_차량'] * 0.4).astype(int)

    # ==============================================================================
    # 🚗 [STEP 4] 실시간 주차 수요(Target) 산출
    # ==============================================================================
    print("🚗 [STEP 4] 주차장 회전율 및 상시 주차 대수 계산 중...")
    df['신규_입차_차량'] = df['도착예정_차량'].shift(-1).fillna(0)

    # 응급실 및 면회객 차량 합산
    df['환자_주차대수'] = (
        df['신규_입차_차량'] * 1.0 +
        df['신규_입차_차량'].shift(1).fillna(0) * 0.9 +
        df['신규_입차_차량'].shift(2).fillna(0) * 0.6 +
        df['신규_입차_차량'].shift(3).fillna(0) * 0.2
    ).astype(int) + df['응급실_차량'] + df['면회객_차량']

    # 상시 주차 (임직원)
    df['상시_주차대수'] = 10 + np.random.randint(-2, 3, size=len(df))
    is_workday = (df['weekday'] < 5) & (df['is_holiday'] == 0)
    is_saturday = (df['weekday'] == 5) & (df['is_holiday'] == 0)
    is_working_day = is_workday | is_saturday

    df.loc[is_working_day & (df['time_val'] == 8.0), '상시_주차대수'] = 30 + np.random.randint(-3, 4, sum(is_working_day & (df['time_val'] == 8.0)))
    df.loc[is_working_day & (df['time_val'] == 8.5), '상시_주차대수'] = 45 + np.random.randint(-3, 4, sum(is_working_day & (df['time_val'] == 8.5)))
    df.loc[is_workday & (df['time_val'] >= 9.0) & (df['time_val'] <= 19.5), '상시_주차대수'] = 50 + np.random.randint(-4, 5, sum(is_workday & (df['time_val'] >= 9.0) & (df['time_val'] <= 19.5)))
    df.loc[is_workday & (df['time_val'] == 20.0), '상시_주차대수'] = 25 + np.random.randint(-3, 4, sum(is_workday & (df['time_val'] == 20.0)))
    df.loc[is_saturday & (df['time_val'] >= 9.0) & (df['time_val'] <= 13.5), '상시_주차대수'] = 50 + np.random.randint(-4, 5, sum(is_saturday & (df['time_val'] >= 9.0) & (df['time_val'] <= 13.5)))
    df.loc[is_saturday & (df['time_val'] == 14.0), '상시_주차대수'] = 20 + np.random.randint(-3, 4, sum(is_saturday & (df['time_val'] == 14.0)))

    df['최종_주차대수'] = df['환자_주차대수'] + df['상시_주차대수']

    def apply_soft_cap(val):
        if val <= 180:
            return val
        else:
            return int(180 + (val - 180) * 0.5)

    df['최종_주차대수'] = df['최종_주차대수'].apply(apply_soft_cap)

    # ==============================================================================
    # 🧹 [STEP 5] 데이터 정리 및 저장
    # ==============================================================================
    print("🧹 [STEP 5] 모델이 학습할 순수 피처(Feature)만 추려내는 중...")
    final_cols = [
        'datetime', 'date', 'hour', 'minute', 
        'is_holiday', 'holiday_name',
        '예약_내과', '예약_정형외과', '예약_소아청소년과', '예약_이비인후과',
        '예약_신경외과', '예약_피부과', '예약_안과', '예약_치과', '예약_정신건강의학과',
        '예약_총외래환자',
        'temp', 'rainfall_mm', 'wind_speed', 'humidity', 'snowfall_cm',
        'pm10', 'pm25', 'pm10_grade', 'pm25_grade',
        '최종_주차대수'
    ]
    
    existing_cols = [c for c in final_cols if c in df.columns]
    df_final = df[existing_cols].copy()
    
    out_filename = os.path.join(base_path, 'merged_features.csv')
    df_final.to_csv(out_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 [최종 완성] 데이터셋이 '{out_filename}'에 저장되었습니다!")
    return df_final

if __name__ == "__main__":
    generate_final_parking_data()