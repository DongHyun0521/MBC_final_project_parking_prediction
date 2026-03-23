import pandas as pd
import numpy as np
import os

def clean_and_grade_air_data():
    print("😷 [V2] 대기질 데이터 결측치 보간 및 환경부 기준 등급화 수술을 시작합니다...")
    
    base_path = "MBC_final_project_parking_prediction/features/air"
    raw_file = os.path.join(base_path, "air_2015_2025_raw.csv")
    
    if not os.path.exists(raw_file):
        print(f"🚨 오류: '{raw_file}' 파일이 없습니다. 이전 병합 코드를 먼저 실행해주세요.")
        return

    # 1. 원본 데이터 불러오기
    df = pd.read_csv(raw_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # 🔥 [수정 1] 결측치 치료 전에 에러코드(-999 등)를 진짜 빈칸(NaN)으로 청소!
    df = df.replace(-999, np.nan)
    
    # ==============================================================================
    # 2. 결측치 치료 (Interpolation)
    # ==============================================================================
    print("🌡️ 1. 센서 점검으로 비어있는 시간(결측치)을 앞뒤 농도로 스무스하게 채웁니다...")
    df['pm10'] = df['pm10'].interpolate(method='time').ffill().bfill()
    df['pm25'] = df['pm25'].interpolate(method='time').ffill().bfill()
    
    # ==============================================================================
    # 3. 30분 단위 보간 (Upsampling)
    # ==============================================================================
    print("⏱️ 2. 날씨 데이터와 싱크를 맞추기 위해 30분 단위로 정밀하게 쪼갭니다...")
    
    # 🔥 [수정 2] 코로나 기간(2020~2023.05)이 억지로 생성되지 않게 기간을 분리해서 resample!
    mask1 = df.index <= '2019-12-31 23:59:59'
    mask2 = df.index >= '2023-06-01 00:00:00'
    
    df_period1 = df[mask1].resample('30min').interpolate(method='time')
    df_period2 = df[mask2].resample('30min').interpolate(method='time')
    
    # 분리해서 뻥튀기한 데이터를 다시 하나로 합치기
    df_30min = pd.concat([df_period1, df_period2])
    
    # ==============================================================================
    # 4. 환경부 공식 기준 등급화 (단기 예보 API 매칭용)
    # ==============================================================================
    print("📊 3. 내일 예보 API와 똑같이 맞추기 위해 0~3 등급(좋음~매우나쁨) 파생 변수를 생성합니다...")
    
    # PM10 등급 (0:좋음, 1:보통, 2:나쁨, 3:매우나쁨)
    def get_pm10_grade(val):
        if val <= 30: return 0
        elif val <= 80: return 1
        elif val <= 150: return 2
        else: return 3
        
    # PM2.5 등급
    def get_pm25_grade(val):
        if val <= 15: return 0
        elif val <= 35: return 1
        elif val <= 75: return 2
        else: return 3
        
    df_30min['pm10_grade'] = df_30min['pm10'].apply(get_pm10_grade)
    df_30min['pm25_grade'] = df_30min['pm25'].apply(get_pm25_grade)
    
    # 마무리 정리
    df_30min['pm10'] = df_30min['pm10'].round(1)
    df_30min['pm25'] = df_30min['pm25'].round(1)
    df_30min = df_30min.reset_index()
    
    # 저장
    save_filename = os.path.join(base_path, "air_2015_2025.csv")
    df_30min.to_csv(save_filename, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("🎉 대성공! 모델에 바로 넣을 수 있는 '30분 단위 + 등급화' 데이터가 완성되었습니다!")
    print("="*50)
    print("\n👀 뇌에 쏙쏙 박히는 데이터 미리보기:")
    print(df_30min.head(10))
    print("\n✅ 최종 결측치 검사 (전부 0이어야 함):")
    print(df_30min.isnull().sum())

if __name__ == "__main__":
    clean_and_grade_air_data()