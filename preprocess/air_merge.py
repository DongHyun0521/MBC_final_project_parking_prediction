# 실행 전 pip install openpyxl

import pandas as pd
import numpy as np
import glob
import os

def extract_and_inspect_air_data():
    print("=> 에어코리아: 대기질 (종로구) - 엑셀 파일 병합")
    
    # ========== 엑셀 파일 위치 파악 ==========

    base_path = "MBC_final_project_parking_prediction/features/air"
    
    files_xlsx = glob.glob(os.path.join(base_path, "**", "*.xlsx"), recursive=True)
    files_xls = glob.glob(os.path.join(base_path, "**", "*.xls"), recursive=True)

    files = files_xlsx + files_xls
    files = [f for f in files if not os.path.basename(f).startswith('~$')]
    print(f"=> 엑셀 파일 : {len(files)}개")
    
    if not files:
        print("=> 잘못된 경로")
        return

    dfs = []
    
    for i, f in enumerate(files, 1):
        print(f"[{i:02d}/{len(files):02d}] 읽는 중: {os.path.basename(f)} ...", end=" ")
        
        try:
            df = pd.read_excel(f)
            
            # ========== 측정일시 컬럼 형식 통일하기 ==========

            col_mapping = {}
            for col in df.columns:
                c_str = str(col).replace(" ", "").upper()
                if '측정일시' in c_str or 'DATE' in c_str: col_mapping[col] = 'datetime_raw'
                elif '측정소명' in c_str or 'STATION' in c_str: col_mapping[col] = 'station'
                elif 'PM10' in c_str: col_mapping[col] = 'pm10'
                elif 'PM2.5' in c_str or 'PM25' in c_str: col_mapping[col] = 'pm25'
                
            df = df.rename(columns=col_mapping)
            
            if 'station' not in df.columns or 'datetime_raw' not in df.columns:
                print("=> 규격 다름 (스킵)")
                continue
                
            # ========== 종로구 필터링 ==========

            df_jongno = df[df['station'].astype(str).str.contains('종로구', na=False)].copy()
            
            keep_cols = ['datetime_raw']
            if 'pm10' in df_jongno.columns: keep_cols.append('pm10')
            if 'pm25' in df_jongno.columns: keep_cols.append('pm25')
            
            df_jongno = df_jongno[keep_cols]
            dfs.append(df_jongno)
            print(f"=> 종로구 데이터 : {len(df_jongno)}건")
            
        except Exception as e:
            print(f"=> 에러 발생: {e}")
            
    if not dfs:
        print("=> 종로구 데이터 없음")
        return
        
    print("=> 종로구 데이터 하나로 병합")
    final_df = pd.concat(dfs, ignore_index=True)
    
    # ========== 24시 -> 다음날 00시로 변환 ==========

    date_str = final_df['datetime_raw'].astype(str).str.replace(".0", "", regex=False)
    dates = pd.to_datetime(date_str.str[:-2], format='%Y%m%d', errors='coerce')
    hours = pd.to_timedelta(date_str.str[-2:].astype(int, errors='ignore'), unit='h', errors='coerce')
    final_df['datetime'] = dates + hours
    
    # ========== datetime, pm10, pm25만 추출 후, datetime 기준 정렬 ==========

    cols = ['datetime']
    if 'pm10' in final_df.columns: cols.append('pm10')
    if 'pm25' in final_df.columns: cols.append('pm25')
    final_df = final_df[cols].dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

    save_filename = os.path.join(base_path, "air_2015_2025_raw.csv")
    final_df.to_csv(save_filename, index=False, encoding='utf-8-sig')
    print(f"🎉 성공! 병합된 순수 원본 파일이 '{save_filename}'에 저장되었습니다.")
    
    # ========== 결측/이상치 파악 ==========

    error_codes = [-999, -999.0, -99, -99.0, -1, -1.0]
    pm10_errors = final_df['pm10'].isin(error_codes).sum() if 'pm10' in final_df.columns else 0
    pm25_errors = final_df['pm25'].isin(error_codes).sum() if 'pm25' in final_df.columns else 0
    
    # ========== 에러코드 -> NaN 변환 ==========
    stat_df = final_df.replace(error_codes, np.nan)
    
    print("========== 에어코리아: 대기질 (종로구) 결측/이상치 파악 ==========")
    print(f"=> 전체 데이터 : {len(stat_df):,}건")
    
    print(f"=> PM10 에러코드 : {pm10_errors:,}건")
    print(f"=> PM2.5 에러코드 : {pm25_errors:,}건")
    
    print("=> 실제 결측치(빈칸 + 에러코드) 총합:")
    print(stat_df.isnull().sum())
    
    print("🔍 3. 이상치(Outlier) 확인을 위한 기초 통계량 (에러코드 제외):")
    print(stat_df.describe().round(1))

if __name__ == "__main__":
    extract_and_inspect_air_data()