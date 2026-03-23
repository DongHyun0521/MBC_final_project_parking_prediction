import requests
import pandas as pd
import os
import time
import urllib3

# SSL 경고창 숨기기 (공공데이터 API 찌를 때 가끔 나옴)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_holidays():
    print("📅 [V2] 한국천문연구원 특일정보(공휴일) API 수집을 시작합니다...")
    
    api_key = "a913790eaa3034943cc340da8ef5b185195b7ea78dce8dc3183461115d62e68e"
    url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
    
    # 1. 타겟 기간 월(Month) 리스트 정교하게 생성
    target_months = []
    for year in range(2015, 2026):
        for month in range(1, 13):
            if 2015 <= year <= 2019:
                target_months.append((year, month))
            elif year in [2020, 2021, 2022]:
                continue # 코로나 기간 스킵
            elif year == 2023 and month >= 6:
                target_months.append((year, month))
            elif year == 2024:
                target_months.append((year, month))
            elif year == 2025 and month <= 10:
                target_months.append((year, month))

    holidays = []
    
    print(f"🎯 총 {len(target_months)}개월 치의 달력을 뒤집니다! (서버 과부하 방지를 위해 약간의 시간이 소요됩니다.)\n")
    
    # 2. API 찌르기 반복문
    for year, month in target_months:
        params = {
            "serviceKey": api_key,
            "solYear": str(year),
            "solMonth": f"{month:02d}", # 1 -> "01" 포맷
            "_type": "json",
            "numOfRows": "100"
        }
        
        try:
            # API 요청 (verify=False는 SSL 인증서 오류 무시)
            response = requests.get(url, params=params, verify=False, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # 해당 월에 공휴일이 0개인 경우 (예: 11월)
                if data['response']['body']['totalCount'] == 0:
                    continue
                    
                items = data['response']['body']['items']['item']
                
                # 🚨 공공데이터 API 고질병 방어: 데이터가 1개면 딕셔너리, 여러 개면 리스트로 옴
                if isinstance(items, dict):
                    items = [items]
                    
                for item in items:
                    if item.get('isHoliday') == 'Y':
                        holidays.append({
                            'date': pd.to_datetime(str(item['locdate']), format='%Y%m%d'),
                            'holiday_name': item['dateName'],
                            'is_holiday': 1 # 머신러닝이 먹기 좋게 1로 세팅
                        })
            else:
                print(f"❌ API 응답 에러 [{year}-{month:02d}]: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ 통신 에러 [{year}-{month:02d}]: {e}")
        
        # 공공기관 서버를 배려하는 0.1초 휴식 (IP 차단 방지)
        time.sleep(0.1)

    # 3. 데이터프레임 변환 및 전처리
    df_holidays = pd.DataFrame(holidays)
    
    # 혹시 모를 중복 데이터 제거 및 날짜순 정렬
    df_holidays = df_holidays.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    
    # 4. 저장
    save_path = "MBC_final_project_parking_prediction/features/holiday"
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "holiday_2015_2025.csv")
    
    df_holidays.to_csv(save_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"🎉 대성공! 총 {len(df_holidays)}일의 공휴일이 '{save_file}'에 안전하게 저장되었습니다.")
    print("="*50)
    print("👀 첫 5개 공휴일:")
    print(df_holidays.head())
    print("\n👀 마지막 5개 공휴일:")
    print(df_holidays.tail())

if __name__ == "__main__":
    fetch_holidays()