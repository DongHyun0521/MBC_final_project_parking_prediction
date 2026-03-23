import pandas as pd
import glob
import os

def merge_and_inspect_weather():
    print("🌤️ [V2] 7.5년치 기상청 날씨 데이터 병합 및 데이터 검증(EDA)을 시작합니다...\n")
    
    # ==============================================================================
    # 1. 8개의 CSV 파일 한 번에 불러오기 및 병합
    # ==============================================================================
    # 다운받으신 파일 패턴에 맞게 경로를 설정합니다. (features 폴더 내의 weather_ 로 시작하는 csv 모두 읽기)
    base_path = "MBC_final_project_parking_prediction/features/weather"
    file_pattern = os.path.join(base_path, "weather_*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        print("🚨 오류: 지정된 경로에 날씨 CSV 파일이 없습니다! 파일명이나 경로를 확인해주세요.")
        return
        
    print(f"📥 총 {len(files)}개의 날씨 파일을 발견했습니다. 병합을 진행합니다...")
    
    # 기상청 데이터는 보통 cp949 인코딩을 사용합니다.
    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f, encoding='cp949'))
        except UnicodeDecodeError:
            df_list.append(pd.read_csv(f, encoding='utf-8')) # 혹시 몰라 utf-8도 대비
            
    df = pd.concat(df_list, ignore_index=True)
    
    # ==============================================================================
    # 2. 컬럼명 영문으로 직관적으로 변경
    # ==============================================================================
    col_map = {
        '일시': 'datetime',
        '기온(°C)': 'temp',
        '강수량(mm)': 'rainfall_mm',
        '풍속(m/s)': 'wind_speed',
        '습도(%)': 'humidity',
        '적설(cm)': 'snowfall_cm'
    }
    
    # 원본 데이터에 존재하는 컬럼만 매핑하여 이름 변경
    rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # 필수 컬럼만 남기기 (지점 번호 이런 건 날려버림)
    essential_cols = ['datetime', 'temp', 'rainfall_mm', 'wind_speed', 'humidity', 'snowfall_cm']
    df = df[[c for c in essential_cols if c in df.columns]]
    
    # ==============================================================================
    # 3. 타겟 기간 필터링 (골든 타임 픽스!)
    # ==============================================================================
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    mask = (
        ((df['datetime'] >= '2015-01-01') & (df['datetime'] <= '2019-12-31 23:59:59')) |
        ((df['datetime'] >= '2023-06-01') & (df['datetime'] <= '2025-10-31 23:59:59'))
    )
    df_target = df.loc[mask].copy()
    
    # 시간순으로 예쁘게 정렬
    df_target = df_target.sort_values('datetime').reset_index(drop=True)
    
    # ==============================================================================
    # 🕵️‍♂️ 4. 결측값(NaN) 및 이상값(Outlier) 진단 레포트
    # ==============================================================================
    print("\n" + "="*50)
    print("📊 [데이터 진단 레포트]")
    print("="*50)
    
    print(f"✅ 총 데이터 행(Row) 수: {len(df_target):,}건 (1시간 단위 기준)\n")
    
    print("🔍 1. 결측치(빈칸) 개수 확인:")
    null_counts = df_target.isnull().sum()
    print(null_counts)
    print("\n💡 Tip: 강수량과 적설량의 결측치는 '비/눈이 안 온 날'이므로 0으로 채우면 됩니다!")
    print("💡 Tip: 기온/습도/풍속의 결측치는 센서 오류일 가능성이 높으므로 앞뒤 시간으로 보간(Interpolate)해야 합니다.\n")
    
    print("🔍 2. 이상치(Outlier) 확인을 위한 기초 통계량:")
    # 보기 편하게 소수점 1자리까지만 출력
    print(df_target.describe().round(1))
    
    print("\n💡 이상치 체크리스트:")
    print(" - temp(기온)가 45도 이상이거나 -30도 이하인 미친 값이 있는가?")
    print(" - humidity(습도)가 0% 미만이거나 100%를 초과하는가?")
    print(" - wind_speed(풍속)가 비정상적으로 높지는 않은가?")
    print("="*50 + "\n")
    
    # ==============================================================================
    # 5. 합쳐진 원본 파일 저장 (결측치 채우기 전의 순수 데이터)
    # ==============================================================================
    save_filename = os.path.join(base_path, 'weather_2015_2025_raw.csv')
    df_target.to_csv(save_filename, index=False, encoding='utf-8-sig')
    
    print(f"🎉 성공! 병합된 파일이 '{save_filename}'에 저장되었습니다.")
    print("⚠️ (주의) 현재 파일은 합치기만 한 상태이며, 결측치 처리는 아직 하지 않았습니다.")

if __name__ == "__main__":
    merge_and_inspect_weather()