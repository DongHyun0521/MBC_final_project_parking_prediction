import pandas as pd
import os

def preprocess_and_upsample_weather():
    print("🛠️ [V2] 날씨 데이터 결측치 처리 및 30분 단위 보간(Upsampling)을 시작합니다...")
    
    base_path = "MBC_final_project_parking_prediction/features/weather"
    raw_file = os.path.join(base_path, "weather_2015_2025_raw.csv")
    
    if not os.path.exists(raw_file):
        print(f"🚨 오류: '{raw_file}' 파일이 없습니다. 이전 병합 코드를 먼저 실행해주세요.")
        return

    # 1. 원본 데이터 불러오기
    df = pd.read_csv(raw_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # ==============================================================================
    # 2. 결측치(NaN) 완벽 처리
    # ==============================================================================
    print("💧 1. 비/눈이 안 온 날의 결측치를 0.0으로 채웁니다...")
    df['rainfall_mm'] = df['rainfall_mm'].fillna(0.0)
    df['snowfall_cm'] = df['snowfall_cm'].fillna(0.0)
    
    print("🌡️ 2. 기온/습도/풍속의 센서 오류(결측치)를 앞뒤 시간의 평균으로 메꿉니다...")
    # interpolate('time') : 시간에 비례해서 빈칸을 자연스럽게 채움 (예: 1시 10도, 3시 14도면 -> 2시는 12도로 채움)
    df['temp'] = df['temp'].interpolate(method='time').ffill().bfill()
    df['humidity'] = df['humidity'].interpolate(method='time').ffill().bfill()
    df['wind_speed'] = df['wind_speed'].interpolate(method='time').ffill().bfill()
    
    # ==============================================================================
    # 3. 30분 단위로 뻥튀기 (Upsampling) - 코로나 기간 제외 유지
    # ==============================================================================
    print("⏱️ 3. 1시간 단위 데이터를 30분 단위로 정밀하게 쪼개고 보간합니다...")
    
    # 기간을 두 개로 분리
    mask1 = df.index <= '2019-12-31 23:59:59'
    mask2 = df.index >= '2023-06-01 00:00:00'
    
    # 각각 독립적으로 30분 단위 resample 및 보간 진행
    df_period1 = df[mask1].resample('30T').interpolate(method='time')
    df_period2 = df[mask2].resample('30T').interpolate(method='time')
    
    # 다시 하나로 병합 (코로나 기간은 쏙 빠진 채로 병합됨)
    df_30min = pd.concat([df_period1, df_period2])
    
    # ==============================================================================
    # 4. 마무리 및 저장
    # ==============================================================================
    df_30min = df_30min.reset_index()
    
    # 실수형(float) 소수점 정리 (기온/풍속 1자리, 강수/적설 1자리, 습도는 정수로)
    df_30min['temp'] = df_30min['temp'].round(1)
    df_30min['rainfall_mm'] = df_30min['rainfall_mm'].round(1)
    df_30min['wind_speed'] = df_30min['wind_speed'].round(1)
    df_30min['snowfall_cm'] = df_30min['snowfall_cm'].round(1)
    df_30min['humidity'] = df_30min['humidity'].round(0).astype(int)
    
    save_filename = os.path.join(base_path, "weather_2015_2025.csv")
    df_30min.to_csv(save_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 대성공! 머신러닝에 당장 넣어도 완벽한 30분 단위 날씨 데이터가 '{save_filename}'에 저장되었습니다!")
    print("\n👀 30분 단위로 예쁘게 채워진 데이터 미리보기:")
    print(df_30min.head(10))
    print("\n✅ 최종 결측치 검사 (전부 0이 나와야 정상입니다!):")
    print(df_30min.isnull().sum())

if __name__ == "__main__":
    preprocess_and_upsample_weather()