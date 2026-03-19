import os
import glob
import pandas as pd

def preprocess_medical_data():
    base_folder = "MBC_final_project_parking_prediction/features"
    search_pattern = os.path.join(base_folder, "*", "*진료정보*.csv")
    all_files = glob.glob(search_pattern)
    
    # 1. 서울 종로구의 공공데이터 지역코드
    JONGRO_SIGUNGU_CODE = 11110
    SEOUL_SIDO_CODE = 11
    
    print("⏳ 파일별로 질병을 분류하며 서울/종로구 데이터만 정밀 타격 중...")
    
    df_list = []
    
    for file in all_files:
        file_name = os.path.basename(file)
        
        disease_type = "알수없음"
        for d in ["감기", "눈병", "천식", "피부염"]:
            if d in file_name:
                disease_type = d
                break
                
        try:
            try:
                df = pd.read_csv(file, encoding='cp949')
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding='utf-8')
                
            if '시군구지역코드' in df.columns:
                df = df[df['시군구지역코드'] == JONGRO_SIGUNGU_CODE]
            elif '시도지역코드' in df.columns:
                df = df[df['시도지역코드'] == SEOUL_SIDO_CODE]
            
            if not df.empty:
                df['질병명'] = disease_type
                df_list.append(df)
                
        except Exception as e:
            continue

    print("🔄 정제된 알짜배기 데이터를 병합하는 중...")
    master_df = pd.concat(df_list, ignore_index=True)
    
    # 5. 💡 [오류 해결] 섞여 있는 날짜 포맷을 'YYYY-MM-DD'로 완벽하게 통일!
    print("📅 날짜 포맷을 'YYYY-MM-DD'로 예쁘게 통일하는 중...")
    # 1단계: 일단 짝대기(-)가 있으면 전부 지워서 '20140101' 형태로 강제 통일
    master_df['날짜'] = master_df['날짜'].astype(str).str.replace('-', '', regex=False)
    # 2단계: 그 상태에서 날짜로 인식시킨 후 'YYYY-MM-DD'로 예쁘게 다시 포맷팅!
    master_df['날짜'] = pd.to_datetime(master_df['날짜'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    
    # ---------------------------------------------------------
    # 💡 [핵심 추가] 코로나 기간(2020.01 ~ 2023.05) 및 불필요한 기간 싹둑 자르기!
    print("✂️ 코로나 기간 및 불필요한 날짜 데이터를 도려냅니다...")
    
    # 조건 1: 2014년 1월 1일 ~ 2019년 12월 31일
    mask1 = (master_df['날짜'] >= '2014-01-01') & (master_df['날짜'] <= '2019-12-31')
    # 조건 2: 2023년 6월 1일 ~ 2023년 12월 31일
    mask2 = (master_df['날짜'] >= '2023-06-01') & (master_df['날짜'] <= '2023-12-31')
    
    # 두 조건 중 하나라도 만족하는(코로나 기간이 아닌) 진짜 데이터만 남김
    master_df = master_df[mask1 | mask2]
    # ---------------------------------------------------------

    # 6. 같은 날짜, 같은 질병끼리 묶어서 발생건수를 더함
    df_grouped = master_df.groupby(['날짜', '질병명'])['발생건수(건)'].sum().reset_index()
    
    # 7. 피벗(Pivot): 세로로 긴 데이터를 '날짜' 기준으로 가로(감기, 눈병...)로 넓게 펼치기!
    print("📉 날짜별/질병별로 표를 예쁘게 펼치는 중...")
    df_final = df_grouped.pivot(index='날짜', columns='질병명', values='발생건수(건)').reset_index()
    
    # 결측치(NaN)는 0건으로 처리
    df_final = df_final.fillna(0)
    
    # 컬럼 이름 예쁘게 변경 (날짜 -> date)
    df_final.rename(columns={'날짜': 'date'}, inplace=True)
    
    # 8. 최종 저장 (이름도 통일성 있게 변경)
    output_file = "MBC_final_project_parking_prediction/features/treatment_2014_2023_seoul_jongno.csv"
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n🎉 대성공! 진료 건수 데이터 전처리 완료!")
    print(f"💾 결과가 '{output_file}'에 저장되었습니다. (총 {len(df_final):,}일치 데이터)")
    print("\n[전처리 완료된 데이터 샘플]")
    print(df_final.head())

if __name__ == "__main__":
    preprocess_medical_data()