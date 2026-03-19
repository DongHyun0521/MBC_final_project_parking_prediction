import os
import glob
import pandas as pd

def merge_air_quality_data():
    # 1. 파일이 있는 폴더 경로 지정
    folder_path = "MBC_final_project_parking_prediction/features/air"
    
    # 2. 💡 [수정됨] .csv 대신 .xlsx 파일들을 싹 다 찾습니다!
    all_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    if not all_files:
        print("🚨 폴더에 Excel 파일이 없습니다. 경로를 다시 확인해 주세요!")
        return
        
    print(f"📂 총 {len(all_files)}개의 대기질 엑셀 파일을 발견했습니다. 병합을 시작합니다...")
    
    df_list = []
    
    # 3. 파일들을 하나씩 읽어서 리스트에 담기
    for file in all_files:
        file_name = os.path.basename(file)
        
        try:
            # 1. 엑셀을 읽는다
            df = pd.read_excel(file, engine='openpyxl')
            
            # 💡 [질문자님 아이디어 적용!] 읽자마자 바로 종로구만 싹둑 잘라낸다!
            df_jongro_only = df[(df['지역'] == '서울 종로구') & (df['측정소명'] == '종로구')]
            
            # 2. 가벼워진 종로구 데이터만 리스트에 담는다
            df_list.append(df_jongro_only)
            
            print(f" - ✅ {file_name} 처리 완료 (종로구 데이터만: {len(df_jongro_only):,}줄 추출)")
            
        except Exception as e:
            print(f"🚨 {file_name} 읽기 실패: {e}")
            continue

    # 4. 이제 합쳐봐야 겨우 수만 줄밖에 안 되는 가벼운 표가 됩니다!
    master_df = pd.concat(df_list, ignore_index=True)
    
    print("\n🎉 대성공! 모든 파일 병합 완료!")
    print(f"📊 총 데이터 갯수: {len(master_df):,} 줄")
    
    # 5. 합쳐진 데이터의 컬럼명 출력
    print("\n===========================================")
    print("📋 [현재 합쳐진 데이터의 컬럼(열) 이름들입니다]")
    print("===========================================")
    print(master_df.columns.tolist())
    print("===========================================\n")
    
    # 6. 💡 [핵심] 엑셀로 읽었지만, 나중에 우리가 다루기 편하도록 저장할 땐 CSV로 변환해서 저장합니다!
    save_path = "MBC_final_project_parking_prediction/features/merged_air.csv"
    master_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"💾 엑셀 파일들을 합쳐서 '{save_path}' (CSV 형태)로 안전하게 변환/저장했습니다.")

if __name__ == "__main__":
    merge_air_quality_data()