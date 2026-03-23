# 실행 전 pip install openpyxl

import pandas as pd
import numpy as np
import glob
import os

def extract_and_inspect_air_data():
    print("😷 [V2] 에어코리아 데이터 '종로구' 추출 및 진단(EDA)을 시작합니다...")
    print("☕ 엑셀 파일(.xlsx)이 많아서 시간이 조금 걸릴 수 있습니다. (진짜 쌩노가다 중...)\n")
    
    base_path = "MBC_final_project_parking_prediction/features/air"
    # 하위 폴더의 모든 xlsx 파일을 찾습니다.
    files_xlsx = glob.glob(os.path.join(base_path, "**", "*.xlsx"), recursive=True)
    files_xls = glob.glob(os.path.join(base_path, "**", "*.xls"), recursive=True) # 구형 엑셀 추가!
    files = files_xlsx + files_xls # 두 개 합치기
    
    # 엑셀이 열려있을 때 생기는 가짜 유령 파일(~$...) 완벽 차단!
    files = [f for f in files if not os.path.basename(f).startswith('~$')]
    
    print(f"📥 총 {len(files)}개의 엑셀 파일을 샅샅이 찾아냈습니다!")
    
    if not files:
        print("🚨 오류: 지정된 경로에 엑셀 파일이 없습니다. 폴더 위치를 확인해주세요!")
        return

    dfs = []
    
    for i, f in enumerate(files, 1):
        print(f"[{i:02d}/{len(files):02d}] 읽는 중: {os.path.basename(f)} ...", end=" ")
        
        try:
            df = pd.read_excel(f)
            
            # 연도별로 지멋대로인 컬럼명 통일시키기
            col_mapping = {}
            for col in df.columns:
                c_str = str(col).replace(" ", "").upper()
                if '측정일시' in c_str or 'DATE' in c_str: col_mapping[col] = 'datetime_raw'
                elif '측정소명' in c_str or 'STATION' in c_str: col_mapping[col] = 'station'
                elif 'PM10' in c_str: col_mapping[col] = 'pm10'
                elif 'PM2.5' in c_str or 'PM25' in c_str: col_mapping[col] = 'pm25'
                
            df = df.rename(columns=col_mapping)
            
            if 'station' not in df.columns or 'datetime_raw' not in df.columns:
                print("❌ 규격 다름 (스킵)")
                continue
                
            # 🔥 '종로구' 데이터만 필터링!
            df_jongno = df[df['station'].astype(str).str.contains('종로구', na=False)].copy()
            
            keep_cols = ['datetime_raw']
            if 'pm10' in df_jongno.columns: keep_cols.append('pm10')
            if 'pm25' in df_jongno.columns: keep_cols.append('pm25')
            
            df_jongno = df_jongno[keep_cols]
            dfs.append(df_jongno)
            print(f"✅ 종로구 {len(df_jongno)}건 추출 완료")
            
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            
    if not dfs:
        print("🚨 추출된 종로구 데이터가 없습니다.")
        return
        
    print("\n🧩 추출된 모든 종로구 데이터를 하나로 병합합니다...")
    final_df = pd.concat(dfs, ignore_index=True)
    
    # ==============================================================================
    # ⏰ 시간 변환 (24시 함정만 먼저 처리하여 시간순 정렬 가능하게 만듦)
    # ==============================================================================
    date_str = final_df['datetime_raw'].astype(str).str.replace(".0", "", regex=False)
    dates = pd.to_datetime(date_str.str[:-2], format='%Y%m%d', errors='coerce')
    hours = pd.to_timedelta(date_str.str[-2:].astype(int, errors='ignore'), unit='h', errors='coerce')
    final_df['datetime'] = dates + hours
    
    # 쓸모없는 raw 컬럼 버리고, datetime 기준으로 정렬
    cols = ['datetime']
    if 'pm10' in final_df.columns: cols.append('pm10')
    if 'pm25' in final_df.columns: cols.append('pm25')
    final_df = final_df[cols].dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    
    # ==============================================================================
    # 🕵️‍♂️ 결측치 및 이상치 진단 레포트
    # ==============================================================================
    # 1. 에러코드(-999 등) 개수 파악
    error_codes = [-999, -999.0, -99, -99.0, -1, -1.0]
    pm10_errors = final_df['pm10'].isin(error_codes).sum() if 'pm10' in final_df.columns else 0
    pm25_errors = final_df['pm25'].isin(error_codes).sum() if 'pm25' in final_df.columns else 0
    
    # 2. 통계를 정확히 내기 위해 에러코드를 NaN으로 임시 변환
    # (안 그러면 -999 때문에 평균값이 영하로 찍힙니다!)
    stat_df = final_df.replace(error_codes, np.nan)
    
    print("\n" + "="*50)
    print("📊 [에어코리아 종로구 데이터 진단 레포트]")
    print("="*50)
    print(f"✅ 총 데이터 행(Row) 수: {len(stat_df):,}건 (1시간 단위)")
    
    print("\n🔍 1. 통신장애/점검 에러코드(-999 등) 개수:")
    print(f" - PM10 에러코드: {pm10_errors:,}건")
    print(f" - PM2.5 에러코드: {pm25_errors:,}건")
    
    print("\n🔍 2. 실제 결측치(빈칸 + 에러코드) 총합:")
    print(stat_df.isnull().sum())
    
    print("\n🔍 3. 이상치(Outlier) 확인을 위한 기초 통계량 (에러코드 제외):")
    print(stat_df.describe().round(1))
    
    print("\n💡 이상치 체크리스트:")
    print(" - max 값을 보세요. 상식 밖의 미친 숫자(예: 3000 이상)가 있나요?")
    print(" - (참고: 역대급 황사가 올 때 PM10은 1000 근처까지 갈 수 있습니다.)")
    print("="*50 + "\n")
    
    # ==============================================================================
    # 원본 파일 저장 (결측치 메꾸기 전)
    # ==============================================================================
    save_filename = os.path.join(base_path, "air_2015_2025_raw.csv")
    
    # 저장할 때는 에러코드(-999)가 들어있는 원래 데이터를 저장합니다!
    final_df.to_csv(save_filename, index=False, encoding='utf-8-sig')
    
    print(f"🎉 성공! 병합된 순수 원본 파일이 '{save_filename}'에 저장되었습니다.")
    print("⚠️ (주의) 현재 파일은 결측치 처리를 하지 않은 1시간 단위 원본 상태입니다.")

if __name__ == "__main__":
    extract_and_inspect_air_data()