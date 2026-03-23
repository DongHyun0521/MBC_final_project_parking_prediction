import pandas as pd
import numpy as np
import os

def generate_half_hourly_hospital_data():
    print("🏥 [V9] 최신 병원 운영시간(평일 19:30, 토요일 13:30 마감) 반영 데이터 생성 중...")
    
    base_path = "MBC_final_project_parking_prediction/features"
    os.makedirs(base_path, exist_ok=True)
    
    # ==============================================================================
    # 1. 30분 단위 뼈대 생성 (우리의 완벽한 타겟 기간)
    # ==============================================================================
    dates_part1 = pd.date_range(start='2015-01-01 00:00:00', end='2019-12-31 23:30:00', freq='30min')
    dates_part2 = pd.date_range(start='2023-06-01 00:00:00', end='2025-10-31 23:30:00', freq='30min')
    df = pd.DataFrame({'datetime': dates_part1.union(dates_part2)})
    
    df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['weekday'] = df['datetime'].dt.dayofweek 
    df['month'] = df['datetime'].dt.month
    
    np.random.seed(42)
    
    # ==============================================================================
    # 2. 공휴일 연동 및 이름(holiday_name) 가져오기
    # ==============================================================================
    holiday_file = os.path.join(base_path, 'holiday/holiday_2015_2025.csv')
    if os.path.exists(holiday_file):
        df_hol = pd.read_csv(holiday_file)
        hol_name_map = dict(zip(df_hol['date'], df_hol['holiday_name']))
        df['holiday_name'] = df['date'].map(hol_name_map).fillna('')
        df['is_holiday'] = (df['holiday_name'] != '').astype(int)
    else:
        df['holiday_name'] = ''
        df['is_holiday'] = 0

    # ==============================================================================
    # 🌟 3. 연속 휴일 수(Streak) 비례 폭주 계산 로직
    # ==============================================================================
    daily_df = df[['date', 'is_holiday', 'weekday', 'holiday_name']].drop_duplicates().sort_values('date')
    daily_df['is_off'] = ((daily_df['is_holiday'] == 1) | (daily_df['weekday'] == 6)).astype(int)
    daily_df['off_streak'] = daily_df['is_off'].groupby((daily_df['is_off'] == 0).cumsum()).cumsum()
    daily_df['yesterday_off_streak'] = daily_df['off_streak'].shift(1).fillna(0)
    
    def get_post_holiday_multiplier(row):
        if row['is_off'] == 1: return 0.0
        streak = row['yesterday_off_streak']
        if streak == 1: return 1.2
        elif streak == 2: return 1.5
        elif streak == 3: return 2.0
        elif streak >= 4: return 3.0
        return 1.0
        
    daily_df['streak_multiplier'] = daily_df.apply(get_post_holiday_multiplier, axis=1)
    df['streak_multiplier'] = df['date'].map(dict(zip(daily_df['date'], daily_df['streak_multiplier'])))

    # ==============================================================================
    # ⏱️ 4. [핵심 업데이트] 최신 병원 운영시간 및 교대 진료 완벽 반영
    # ==============================================================================
    # 🔽🔽🔽 여기서부터 교체된 부분입니다 🔽🔽🔽
    def get_time_weight(row):
        time_val = row['hour'] + (row['minute'] / 60.0)
        w = row['weekday']
        
        # 1. 일요일/공휴일 휴진
        if row['is_holiday'] == 1 or w == 6: return 0.0 
        
        base_weight = 0.0
        
        # 2. 토요일 (09:00 ~ 13:30 운영 / 12:30 예약 마감)
        if w == 5:
            if time_val < 9.0 or time_val >= 13.0: return 0.0
            if 9.0 <= time_val <= 10.5: base_weight = 0.5
            elif 11.0 <= time_val <= 12.5: base_weight = 0.3
            
        # 3. 평일 (09:00 ~ 19:30 운영 / 18:30 예약 마감)
        else:
            if time_val < 9.0 or time_val >= 19.0: return 0.0
            if 9.0 <= time_val <= 11.0: base_weight = 0.60      # 오전 피크
            elif 11.0 <= time_val <= 12.0: base_weight = 0.25   # 오전 끝물
            elif 12.5 <= time_val <= 13.0: base_weight = 0.10   # 점심 연속진료
            elif 13.5 <= time_val <= 15.0: base_weight = 0.35   # 오후 재개
            elif 15.5 <= time_val <= 17.0: base_weight = 0.20   # 오후 일반
            elif 17.5 <= time_val <= 18.5: base_weight = 0.10   # 마감 직전
            
        # 🔥 요일별 가중치 추가 (월요일 버프, 금요일 너프)
        if w == 0: # 월요일
            base_weight *= 1.15
        elif w == 4 and time_val >= 13.5: # 금요일 오후
            base_weight *= 0.85
            
        return base_weight
    # 🔼🔼🔼 여기까지 교체된 부분입니다 🔼🔼🔼
        
    df['time_weight'] = df.apply(get_time_weight, axis=1)
    df['final_weight'] = df['time_weight'] * df['streak_multiplier']

    # ==============================================================================
    # 🏥 5. 외래 예약 환자 투입 (가중치 기반)
    # ==============================================================================
    print("🏥 부서별 환자를 투입 중입니다...")
    df['예약_내과'] = np.maximum(0, (np.random.normal(80, 20, len(df)) * df['final_weight'])).astype(int)
    df['예약_정형외과'] = np.maximum(0, (np.random.normal(60, 15, len(df)) * df['final_weight'])).astype(int)
    
    season_multi = np.where(df['month'].isin([3, 4, 5, 9, 10, 11]), 1.3, 1.0)
    df['예약_소아청소년과'] = np.maximum(0, (np.random.normal(50, 12, len(df)) * season_multi * df['final_weight'])).astype(int)
    df['예약_이비인후과'] = np.maximum(0, (np.random.normal(45, 10, len(df)) * season_multi * df['final_weight'])).astype(int)
    
    df['예약_신경외과'] = np.maximum(0, (np.random.normal(35, 8, len(df)) * df['final_weight'])).astype(int)
    df['예약_피부과'] = np.maximum(0, (np.random.normal(30, 8, len(df)) * df['final_weight'])).astype(int)
    df['예약_안과'] = np.maximum(0, (np.random.normal(35, 8, len(df)) * df['final_weight'])).astype(int)
    df['예약_치과'] = np.maximum(0, (np.random.normal(30, 8, len(df)) * df['final_weight'])).astype(int)
    df['예약_정신건강의학과'] = np.maximum(0, (np.random.normal(20, 5, len(df)) * df['final_weight'])).astype(int)
    
    outpatient_cols = [c for c in df.columns if c.startswith('예약_')]
    df['예약_총외래환자'] = df[outpatient_cols].sum(axis=1)

    # ==============================================================================
    # 🚑 6. 응급실 명절 도심 공동화 현상 반영
    # ==============================================================================
    '''def get_er_base(row):
        h, w, hol = row['hour'], row['weekday'], row['is_holiday']
        hol_name = row['holiday_name']
        
        night_multiplier = 1.5 if (h >= 20 or h <= 8) else 1.0
        weekend_multiplier = 1.8 if (w in [5, 6] or hol == 1) else 1.0
        
        if '설날' in hol_name or '추석' in hol_name:
            weekend_multiplier = 0.7  # 명절 서울 도심 공동화
            
        return 2.5 * night_multiplier * weekend_multiplier

    df['er_base'] = df.apply(get_er_base, axis=1)
    df['응급실방문'] = np.maximum(0, np.random.normal(df['er_base'], 1)).astype(int)'''

    # ==============================================================================
    # 🧹 7. 정리 및 저장
    # ==============================================================================
    drop_cols = ['time_weight', 'streak_multiplier', 'hour', 'minute', 'weekday', 'month', 'holiday_name']
    df = df.drop(columns=drop_cols)
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    filename = os.path.join(base_path, 'reservation/reservation_2015_2025.csv')
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 대성공! 새로운 운영시간(평일 18:30 예약마감)이 완벽히 반영된 데이터가 '{filename}'에 저장되었습니다!")
    return df

if __name__ == "__main__":
    generate_half_hourly_hospital_data()