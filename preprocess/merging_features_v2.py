import pandas as pd
import os

def handle_missing_v2():
    base_path = "MBC_final_project_parking_prediction/features"
    input_file = os.path.join(base_path, "merged_features_v3.csv") # 방금 만든 V3 파일
    
    print("⏳ V3 마스터 데이터를 불러옵니다...")
    df = pd.read_csv(input_file)

    # (코로나 자르는 코드 삭제됨! 이미 완벽한 데이터니까요!)

    # 1. 자잘한 결측치 치료 (대기질 선형 보간법)
    print("📈 선형 보간법(Interpolation)으로 대기질의 흐름을 부드럽게 이어줍니다...")
    air_cols = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']
    df[air_cols] = df[air_cols].interpolate(method='linear', limit_direction='forward')

    # 2. PM2.5의 거대한 2014년 빵꾸 치료 (과거로 끌어당기기)
    print("⏪ 2014년 초미세먼지 빈칸을 2015년 데이터로 덮어씁니다 (Backward Fill)...")
    df[air_cols] = df[air_cols].bfill()

    # 3. 치료 후 결측치 재확인
    missing_info = df.isnull().sum()
    print("\n🚨 [치료 후 결측치 잔여 현황]")
    print(missing_info[missing_info > 0] if not missing_info[missing_info > 0].empty else "✨ 결측치 0개! 완벽 치료 완료! ✨")

    # 4. 최종 AI 학습용 버전으로 저장!
    output_file = os.path.join(base_path, "merged_features_final.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n🎉 결측치 치료가 완료된 찐 최종 파일이 '{output_file}'에 저장되었습니다!")

if __name__ == "__main__":
    handle_missing_v2()