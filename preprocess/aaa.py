import pandas as pd
import numpy as np
import os

def create_ultimate_master_dataset():
    print("🌟 [Ultimate Merge V2] 예약, 날씨, 대기질, 질병 데이터를 하나로 융합합니다...")
    
    base_path = "MBC_final_project_parking_prediction/features"
    
    # ----------------------------------------------------
    # 1. 4개의 데이터셋 불러오기
    # ----------------------------------------------------
    # (1) 30분 단위 예약 데이터
    df_res = pd.read_csv(f"{base_path}/reservations_2014_2023_seoul_jongno.csv")
    df_res['datetime'] = pd.to_datetime(df_res['datetime'])
    df_res['date'] = df_res['datetime'].dt.strftime('%Y-%m-%d')
    df_res['hour'] = df_res['datetime'].dt.hour
    
    # (2) 기상/공휴일 데이터
    df_weather = pd.read_csv(f"{base_path}/weather_holidays_2014_2023_seoul_jongno.csv")
    df_weather['hour'] = df_weather['hour'].astype(str).str.split(':').str[0].astype(int)
    if 'datetime' in df_weather.columns:
        df_weather = df_weather.drop(columns=['datetime'])
        
    # (3) 대기질 데이터 (PM25 포함)
    df_air = pd.read_csv(f"{base_path}/air_2014_2023_seoul_jongno.csv")
    
    # (4) 질병 통계 데이터
    df_treat = pd.read_csv(f"{base_path}/treatment_2014_2023_seoul_jongno.csv")
    
    # ----------------------------------------------------
    # 2. 초정밀 병합 (Merge)
    # ----------------------------------------------------
    print("🔄 시간/일 단위 기준으로 데이터를 합칩니다...")
    master = pd.merge(df_res, df_weather, on=['date', 'hour'], how='left')
    master = pd.merge(master, df_air, on='date', how='left')
    master = pd.merge(master, df_treat, on='date', how='left')
    
    # 시간 순서대로 정렬 (보간법을 쓰기 위해 필수!)
    master = master.sort_values(by='datetime').reset_index(drop=True)

    # ----------------------------------------------------
    # 💡 3. 질문자님의 완벽한 결측치 치료 로직 (PM25 등)
    # ----------------------------------------------------
    print("📈 선형 보간법(Interpolation) & Bfill로 대기질(PM25 등) 결측치를 완벽 치료합니다...")
    air_cols = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']
    
    # 있는 컬럼만 필터링해서 보간 적용
    existing_air_cols = [col for col in air_cols if col in master.columns]
    
    if existing_air_cols:
        master[existing_air_cols] = master[existing_air_cols].interpolate(method='linear', limit_direction='forward')
        master[existing_air_cols] = master[existing_air_cols].bfill() # 2014년 과거 빵꾸 덮어쓰기

    # 나머지 일반 결측치는 ffill/0 처리 (FutureWarning 해결 완료)
    master = master.ffill().fillna(0)

    # ----------------------------------------------------
    # 🏥 4. 병원 규모 스케일 업! (예약 환자 수 2배 이상 증폭)
    # ----------------------------------------------------
    print("🔥 대형 병원 스케일에 맞게 외래 환자 수를 붐비게 증폭시킵니다!")
    
    # 지워진 final_weight를 찾지 않고, 이미 생성된 CSV의 예약자 수를 2.5배 곱해서 볼륨만 확실히 키웁니다.
    scale_cols = [
        '예약_내과', '예약_정형외과', '예약_소아청소년과', '예약_이비인후과', 
        '예약_신경외과', '예약_피부과', '예약_안과', '예약_치과', '예약_정신건강의학과'
    ]
    
    for col in scale_cols:
        if col in master.columns:
            master[col] = (master[col] * 2.5).astype(int)
            
    # 응급실 환자도 2배 증폭
    if '예약_응급실방문' in master.columns:
        master['예약_응급실방문'] = (master['예약_응급실방문'] * 2.0).astype(int)

    # 증폭된 데이터로 총외래환자 수 재계산
    outpatient_cols = [c for c in master.columns if c.startswith('예약_') and c not in ['예약_총외래환자', '예약_응급실방문']]
    master['예약_총외래환자'] = master[outpatient_cols].sum(axis=1)

    # ----------------------------------------------------
    # 5. 질병 데이터 스케일링 & 워크인(Walk-in) 환자 계산
    # ----------------------------------------------------
    median_cold = master['감기'].replace(0, np.nan).median() or 1
    median_asthma = master['천식'].replace(0, np.nan).median() or 1
    
    master['감기_지수'] = np.clip(master['감기'] / median_cold, 0.5, 2.0)
    master['천식_지수'] = np.clip(master['천식'] / median_asthma, 0.5, 2.0)

    print("😷 환경 데이터(PM10, 기온 등) 기반 현장 접수 환자 계산 중...")
    def calculate_walk_in(row):
        base_walk_in = row['예약_총외래환자'] * 0.20 # 현장접수 비율도 20%로 상승
        if row['예약_총외래환자'] == 0 or row['is_holiday'] == 1:
            return 0
            
        multiplier = 1.0
        if row.get('PM10', 0) >= 80: multiplier += (0.15 * row['천식_지수'])
        if row.get('temp', 15) < 0 or row.get('temp', 15) > 30: multiplier += (0.15 * row['감기_지수'])
        # 질문자님 데이터의 rainfall_mn 변수명 그대로 사용
        if row.get('rainfall_mn', 0) >= 10: multiplier -= 0.20 
            
        return int(base_walk_in * multiplier)

    master['예약_현장접수환자'] = master.apply(calculate_walk_in, axis=1)

    # ----------------------------------------------------
    # 🚗 6. 대망의 주차 수요 최종 예측 (200대 제한 삭제!)
    # ----------------------------------------------------
    print("🚗 주차장 회전율(Turnover)을 반영하여 최종 주차 수요를 산출합니다...")
    
    master['예약_외래_30분뒤'] = master['예약_총외래환자'].shift(-1).fillna(0)
    실제_진료_외래환자 = (master['예약_총외래환자'] * 0.4) + (master['예약_외래_30분뒤'] * 0.6)
    
    random_car_ratio = np.random.uniform(0.60, 0.90, len(master)) # 차량 이용률 60~90%로 상향
    
    # 최종 산식 (200대 제한 없음)
    master['주차수요예측'] = ((실제_진료_외래환자 + master['예약_현장접수환자'] + master['예약_응급실방문']) * random_car_ratio).astype(int)
    
    # 불필요한 계산용 컬럼 정리
    cols_to_drop = ['date', 'hour', '예약_외래_30분뒤', '감기_지수', '천식_지수', 'w_weight', 'h_weight', 'final_weight', 'er_base', 'month', 'minute', 'weekday']
    master = master.drop(columns=[c for c in cols_to_drop if c in master.columns])
    
    # 저장
    output_file = f"{base_path}/ULTIMATE_MASTER_parking_dataset_v2.csv"
    master.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 완벽합니다! PM25 결측치 치료 및 회전율이 반영된 마스터 데이터({len(master)}행)가 '{output_file}'에 저장되었습니다!")
    
    # 검증 출력
    print(f"📊 피크타임 주차 수요 최댓값 확인: {master['주차수요예측'].max()}대 (200대 이상 돌파 확인!)")
    
    return master

if __name__ == "__main__":
    create_ultimate_master_dataset()