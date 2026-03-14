# 진료 예약 더미 데이터 만들기

import pandas as pd
import numpy as np

def generate_custom_hospital_data(start_date='2023-01-01', end_date='2023-12-31'):
    print("🏥 [질문자님 맞춤형] 병원 진료 및 응급실 방문 데이터 생성을 시작합니다...")
    
    dates = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame({'date': dates})
    df['weekday'] = df['date'].dt.dayofweek # 0:월 ~ 6:일
    df['month'] = df['date'].dt.month
    
    # 외래 가중치: 평일(1.0), 월요일(1.2-주말밀림), 토요일(0.5-오전진료), 일요일(0-휴진)
    outpatient_weight = {0: 1.2, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.5, 6: 0.0}
    df['out_weight'] = df['weekday'].map(outpatient_weight)
    
    np.random.seed(42) # 재현성을 위한 시드 고정
    
    # ----------------------------------------------------
    # 1. 외래 예약 환자 (9개 진료과)
    # ----------------------------------------------------
    # (1) 규모가 큰 메인 과 (내과, 정형외과)
    df['resv_Internal'] = np.maximum(0, (np.random.normal(400, 30, len(df))) * df['out_weight']).astype(int)
    df['resv_Ortho'] = np.maximum(0, (np.random.normal(250, 20, len(df))) * df['out_weight']).astype(int)
    
    # (2) 계절성(환절기) 타는 과 (소아청소년과, 이비인후과) -> 3,4,5, 9,10,11월에 1.3배 증가
    season_multi = np.where(df['month'].isin([3, 4, 5, 9, 10, 11]), 1.3, 1.0)
    df['resv_Pediatrics'] = np.maximum(0, (np.random.normal(200, 25, len(df)) * season_multi) * df['out_weight']).astype(int)
    df['resv_ENT'] = np.maximum(0, (np.random.normal(150, 15, len(df)) * season_multi) * df['out_weight']).astype(int)
    
    # (3) 꾸준한 일반 과 (신경외과, 피부과, 안과, 치과, 정신건강의학과)
    df['resv_Neuro'] = np.maximum(0, (np.random.normal(100, 10, len(df))) * df['out_weight']).astype(int)
    df['resv_Derm'] = np.maximum(0, (np.random.normal(80, 10, len(df))) * df['out_weight']).astype(int)
    df['resv_Ophtha'] = np.maximum(0, (np.random.normal(90, 10, len(df))) * df['out_weight']).astype(int)
    df['resv_Dental'] = np.maximum(0, (np.random.normal(70, 8, len(df))) * df['out_weight']).astype(int)
    df['resv_Psych'] = np.maximum(0, (np.random.normal(50, 5, len(df))) * df['out_weight']).astype(int)
    
    # 총 외래 예약수 합산
    outpatient_cols = [c for c in df.columns if c.startswith('resv_')]
    df['total_outpatient'] = df[outpatient_cols].sum(axis=1)

    # ----------------------------------------------------
    # 2. 응급의학과 워크인 환자 (365일 무휴, 휴일 폭증 패턴)
    # ----------------------------------------------------
    # 평일 평균 80명, 일요일/토요일엔 동네병원 닫아서 평균 130명으로 증가
    er_base = np.where(df['weekday'].isin([5, 6]), 130, 80)
    df['walkin_ER'] = np.maximum(0, np.random.normal(er_base, 15)).astype(int)

    # ----------------------------------------------------
    # 3. 주차장 방문 총 예상 차량 대수 (차량 동반 비율 적용)
    # ----------------------------------------------------
    # 환자 1명이 꼭 차 1대를 가져오지 않음 (대중교통 이용 고려해 약 70%가 차를 가져온다고 가정)
    car_ratio = 0.7 
    df['est_parking_demand'] = ((df['total_outpatient'] + df['walkin_ER']) * car_ratio).astype(int)

    # 쓸모없는 컬럼 정리
    df = df.drop(columns=['out_weight'])
    
    # 저장
    filename = '2023_hospital_visitors_v2.csv'
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"🎉 성공! 질문자님의 통찰이 담긴 '{filename}' 생성 완료!")
    print("\n[주말/평일 비교 샘플 (응급실 차이 확인!)]")
    # 금, 토, 일 샘플만 예쁘게 보여주기
    print(df[['date', 'weekday', 'total_outpatient', 'walkin_ER', 'est_parking_demand']].head(7))
    return df

# 실행
hospital_df = generate_custom_hospital_data()