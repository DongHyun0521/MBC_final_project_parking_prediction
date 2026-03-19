import pandas as pd
import numpy as np
import os

def generate_hourly_hospital_data():
    print("🏥 [V3] 시간별(Hourly) 초정밀 병원 진료/주차 더미 데이터 생성을 시작합니다...")
    print("✂️ 코로나19 기간 제외 & 🚗 주차장 200면 & ⏰ 24시간 운영 스케일 적용 중...")
    
    # 1. 일별(Daily)이 아닌 시간별(Hourly) 뼈대 생성!
    dates_part1 = pd.date_range(start='2014-01-01 00:00:00', end='2019-12-31 23:00:00', freq='H')
    dates_part2 = pd.date_range(start='2023-06-01 00:00:00', end='2023-12-31 23:00:00', freq='H')
    dates = dates_part1.union(dates_part2)
    
    df = pd.DataFrame({'datetime': dates})
    
    # 시간 관련 정보 추출
    df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.dayofweek 
    df['month'] = df['datetime'].dt.month
    
    np.random.seed(42)
    
    # ----------------------------------------------------
    # 💡 2. 요일별 & 시간대별 복합 가중치 설계 (핵심!)
    # ----------------------------------------------------
    # (1) 요일 가중치 (월: 제일 바쁨, 일: 휴진)
    weekday_weight = {0: 1.2, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.5, 6: 0.0}
    df['w_weight'] = df['weekday'].map(weekday_weight)
    
    # (2) 시간대 가중치 (평일 기준 하루 환자를 시간대별로 쪼개는 비율)
    def get_hourly_weight(row):
        h = row['hour']
        w = row['weekday']
        
        # 일요일(6)은 무조건 외래 0
        if w == 6: return 0.0
        
        # 토요일(5)은 09:00~12:30 (오전 피크에 다 몰림)
        if w == 5:
            if 9 <= h <= 10: return 0.4  # 오전 9~10시에 40% 몰림
            elif 11 <= h <= 12: return 0.6 # 11~12시에 60% 몰림
            else: return 0.0
            
        # 평일(0~4) 스케줄
        if 9 <= h <= 10: return 0.25      # 오전 피크
        elif 11 <= h <= 12: return 0.15     # 점심 전
        elif h == 13: return 0.05           # 점심시간 (12:30~13:30) -> 13시에는 거의 없음
        elif 14 <= h <= 15: return 0.25     # 오후 피크
        elif 16 <= h <= 17: return 0.20     # 마감 전
        elif h == 18: return 0.10           # 18:30 진료 마감 잔여 인원
        else: return 0.0                    # 야간 휴진
        
    # 데이터프레임에 시간 가중치 적용 (시간이 좀 걸릴 수 있습니다)
    print("⏳ 시간대별로 환자를 분배하고 있습니다...")
    df['h_weight'] = df.apply(get_hourly_weight, axis=1)
    
    # 최종 가중치 = 요일 * 시간
    df['final_weight'] = df['w_weight'] * df['h_weight']

    # ----------------------------------------------------
    # 3. 외래 예약 환자 (시간별 분배!)
    # ----------------------------------------------------
    # 💡 이전처럼 일일 평균을 넣되, 마지막에 final_weight를 곱해서 시간별로 쪼갭니다.
    print("🏥 부서별 환자를 시간표에 맞게 투입 중...")
    
    df['예약_내과'] = np.maximum(0, (np.random.normal(50, 10, len(df)) * df['final_weight'])).astype(int)
    df['예약_정형외과'] = np.maximum(0, (np.random.normal(30, 8, len(df)) * df['final_weight'])).astype(int)
    
    # 계절성 타는 과
    season_multi = np.where(df['month'].isin([3, 4, 5, 9, 10, 11]), 1.3, 1.0)
    df['예약_소아청소년과'] = np.maximum(0, (np.random.normal(25, 5, len(df)) * season_multi * df['final_weight'])).astype(int)
    df['예약_이비인후과'] = np.maximum(0, (np.random.normal(20, 5, len(df)) * season_multi * df['final_weight'])).astype(int)
    
    # 일반 과
    df['예약_신경외과'] = np.maximum(0, (np.random.normal(15, 3, len(df)) * df['final_weight'])).astype(int)
    df['예약_피부과'] = np.maximum(0, (np.random.normal(10, 3, len(df)) * df['final_weight'])).astype(int)
    df['예약_안과'] = np.maximum(0, (np.random.normal(12, 3, len(df)) * df['final_weight'])).astype(int)
    df['예약_치과'] = np.maximum(0, (np.random.normal(10, 3, len(df)) * df['final_weight'])).astype(int)
    df['예약_정신건강의학과'] = np.maximum(0, (np.random.normal(8, 2, len(df)) * df['final_weight'])).astype(int)
    
    outpatient_cols = [c for c in df.columns if c.startswith('예약_')]
    df['예약_총외래환자'] = df[outpatient_cols].sum(axis=1)

    # ----------------------------------------------------
    # 4. 응급의학과 워크인 환자 (24시간 풀가동!)
    # ----------------------------------------------------
    # 평일/주말, 주간/야간에 따라 응급실 방문 빈도가 다름
    def get_er_base(row):
        h = row['hour']
        w = row['weekday']
        
        # 밤~새벽(20시~08시)이 응급실은 더 붐빔 (동네 병원이 닫아서)
        night_multiplier = 1.5 if (h >= 20 or h <= 8) else 1.0
        # 주말(토, 일)은 동네 병원이 닫아서 더 붐빔
        weekend_multiplier = 2.0 if w in [5, 6] else 1.0
        
        # 시간당 기본 2~4명 수준으로 세팅 (하루 합치면 50~100명 선)
        return 2 * night_multiplier * weekend_multiplier

    df['er_base'] = df.apply(get_er_base, axis=1)
    df['예약_응급실방문'] = np.maximum(0, np.random.normal(df['er_base'], 1)).astype(int)

    # ----------------------------------------------------
    # 5. 주차 수요 예측 계산 (최대 200대)
    # ----------------------------------------------------
    car_ratio = 0.7 
    # 시간대별 예측이므로, 해당 시간에 머무르는 차들의 수!
    df['주차수요예측'] = ((df['예약_총외래환자'] + df['예약_응급실방문']) * car_ratio).astype(int)
    df['주차수요예측'] = np.minimum(200, df['주차수요예측'])

    # 쓸모없는 컬럼 정리
    df = df.drop(columns=['w_weight', 'h_weight', 'final_weight', 'er_base', 'hour', 'weekday', 'month'])
    
    # 💡 마스터 병합을 위해 날짜 포맷팅
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 파일 저장
    save_dir = "MBC_final_project_parking_prediction/features"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/reservations_2014_2023_seoul_jongno.csv"
    
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 대성공! 5만 7천 시간 분량의 초정밀 시간별 데이터가 '{filename}'에 저장되었습니다!")
    
    # 검증 출력 (월요일 아침 시간대 확인)
    print("\n[검증] 월요일(평일) 아침 시간대 환자 변화량 확인 (9시~13시)")
    sample_monday = df[df['datetime'].str.contains('2014-01-06 09|2014-01-06 10|2014-01-06 11|2014-01-06 12|2014-01-06 13')]
    print(sample_monday[['datetime', '예약_총외래환자', '예약_응급실방문', '주차수요예측']].to_string(index=False))
    
    return df

if __name__ == "__main__":
    generate_hourly_hospital_data()