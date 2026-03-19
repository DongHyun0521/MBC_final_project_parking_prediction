import os
import pandas as pd

def create_hourly_master_table_v3():
    base_path = "MBC_final_project_parking_prediction/features"

    # 1. 4개의 황금 조각 파일 경로
    file_weather = os.path.join(base_path, "weather_holidays_2014_2023_seoul_jongno.csv")
    file_air = os.path.join(base_path, "air_2014_2023_seoul_jongno.csv")
    file_treat = os.path.join(base_path, "treatment_2014_2023_seoul_jongno.csv")
    file_reserv = os.path.join(base_path, "reservations_2014_2023_seoul_jongno.csv")

    print("⏳ 4개의 전처리 완료 데이터를 로딩합니다...")
    
    try:
        df_weather = pd.read_csv(file_weather)
        df_air = pd.read_csv(file_air)
        df_treat = pd.read_csv(file_treat)
        df_reserv = pd.read_csv(file_reserv)
    except FileNotFoundError as e:
        print(f"🚨 파일을 찾을 수 없습니다. 에러: {e}")
        return

    # 2. 'datetime'이 있는 날씨 데이터를 마스터 뼈대로 지정!
    print("🔄 시간별 날씨 데이터를 뼈대로 삼아 살을 붙입니다...")
    master_df = df_weather.copy()

    # 💡 [핵심 해결책 1] 열쇠가 안 맞아서 튕기지 않게, 양쪽 다 완벽한 datetime 타입으로 강제 통일!
    print("🔑 열쇠(datetime)가 딱 들어맞도록 모양을 다듬고 있습니다...")
    master_df['datetime'] = pd.to_datetime(master_df['datetime'])
    df_reserv['datetime'] = pd.to_datetime(df_reserv['datetime'])

    # 💡 [핵심 해결책 2] 예약 데이터에서 쓸데없는 'date' 컬럼을 미리 지워서 _x, _y 분신술 방지!
    if 'date' in df_reserv.columns:
        df_reserv = df_reserv.drop(columns=['date'])

    # 3. 병합 (Merge) 시작!
    # (1) 대기질, 진료 데이터는 아직 '일별'이므로 예전처럼 'date'를 열쇠로 합칩니다.
    master_df = pd.merge(master_df, df_air, on='date', how='left')
    master_df = pd.merge(master_df, df_treat, on='date', how='left')
    
    # (2) 예약 데이터는 이제 '시간별'이므로 'datetime'을 열쇠로 1:1 완벽 결합!
    master_df = pd.merge(master_df, df_reserv, on='datetime', how='left')

    # 4. 정렬
    print("📅 시간(datetime) 순으로 데이터를 예쁘게 정렬하는 중...")
    master_df = master_df.sort_values('datetime').reset_index(drop=True)

    print(f"\n✅ 조립 완료! 총 {len(master_df):,}시간 치의 마스터 데이터가 탄생했습니다.")

    # 5. 결측치(NaN) 상태 진단 출력
    print("\n===========================================")
    print("🚨 [현재 V3 마스터 테이블의 결측치(NaN) 발생 현황]")
    print("===========================================")
    missing_info = master_df.isnull().sum()
    missing_info = missing_info[missing_info > 0]
    
    if missing_info.empty:
        print("결측치가 하나도 없습니다! 완벽합니다!")
    else:
        print(missing_info)
    print("===========================================\n")

    # 6. 최종 저장
    output_file = os.path.join(base_path, "merged_features_v1.csv")
    master_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"🎉 [V3] 시간별 마스터 파일이 '{output_file}'에 안전하게 저장되었습니다!")
    print("\n[마스터 테이블 엿보기]")
    # 주요 컬럼만 살짝 보여주기
    print(master_df[['datetime', 'temp', '예약_총외래환자', '주차수요예측']].head(5))

if __name__ == "__main__":
    create_hourly_master_table_v3()