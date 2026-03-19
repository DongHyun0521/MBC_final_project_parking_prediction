import os
import requests
import pandas as pd
import time
import calendar

# 질문자님의 공공데이터 API 인증키
API_KEY = "a913790eaa3034943cc340da8ef5b185195b7ea78dce8dc3183461115d62e68e"

def is_covid_period(year, month):
    """
    코로나19의 영향을 강력하게 받은 기간을 필터링합니다.
    (2020년 1월 ~ 2023년 5월)
    """
    if 2020 <= year <= 2022:
        return True
    if year == 2023 and month <= 5:
        return True
    return False

def get_hourly_weather_10years():
    print("🌤️ 기상청 API에서 10년 치(코로나 제외) 시간별 날씨 데이터를 가져옵니다...")
    url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    weather_list = []
    
    # 2014년부터 2023년까지 반복
    for year in range(2014, 2024):
        for month in range(1, 13):
            # 💡 [핵심 최적화] 코로나 기간이면 아예 API 요청을 안 하고 건너뜀!
            if is_covid_period(year, month):
                continue
                
            last_day = calendar.monthrange(year, month)[1] 
            start_dt = f"{year}{month:02d}01"
            end_dt = f"{year}{month:02d}{last_day:02d}"
            
            params = {
                "serviceKey": API_KEY,
                "pageNo": "1",
                "numOfRows": "999", 
                "dataType": "JSON",
                "dataCd": "ASOS",
                "dateCd": "HR",
                "startDt": start_dt,
                "startHh": "00",
                "endDt": end_dt,
                "endHh": "23",
                "stnIds": "108" # 서울
            }
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{url}?{query_string}"
            
            try:
                response = requests.get(full_url)
                data = response.json()
                
                # API 서버 에러 방어 로직
                if data['response']['header']['resultCode'] != '00':
                    print(f"🚨 {year}년 {month}월 에러: {data['response']['header']['resultMsg']}")
                    continue
                    
                items = data['response']['body']['items']['item']
                
                for item in items:
                    tm = item.get('tm', '')
                    if not tm: continue
                    date_part = tm.split(' ')[0]
                    hour_part = tm.split(' ')[1]
                    
                    temp = float(item.get('ta', 0) or 0)
                    rn = float(item.get('rn', 0) or 0)
                    ws = float(item.get('ws', 0) or 0)
                    
                    is_extreme_weather = 1 if rn >= 5.0 or ws >= 10.0 else 0
                    weather_status = "비/눈" if rn > 0 else "맑음/흐림"
                        
                    weather_list.append({
                        "datetime": tm,
                        "date": date_part,
                        "hour": hour_part,
                        "temp": temp,
                        "rainfall_mm": rn,
                        "weather_status": weather_status,
                        "is_extreme_weather": is_extreme_weather
                    })
                    
                print(f" - {year}년 {month}월 날씨 수집 완료")
                
            except Exception as e:
                print(f"🚨 {year}년 {month}월 파싱 에러:", e)
                
            time.sleep(0.5) # API 서버 과부하 방지
            
    return pd.DataFrame(weather_list)


def get_holidays_10years():
    print("\n📅 한국천문연구원 API에서 공휴일 데이터를 가져오는 중...")
    url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
    holiday_list = []
    
    for year in range(2014, 2024):
        for month in range(1, 13):
            # 코로나 기간 건너뛰기
            if is_covid_period(year, month):
                continue
                
            month_str = str(month).zfill(2)
            full_url = f"{url}?serviceKey={API_KEY}&solYear={year}&solMonth={month_str}&_type=json"
            
            try:
                response = requests.get(full_url)
                data = response.json()
                body = data.get('response', {}).get('body', {})
                if body.get('totalCount', 0) > 0:
                    items = body.get('items', {}).get('item', [])
                    if isinstance(items, dict): items = [items]
                    for item in items:
                        raw_date = str(item['locdate'])
                        formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                        holiday_list.append({
                            "date": formatted_date,
                            "holiday_name": item['dateName'],
                            "is_holiday": 1
                        })
            except Exception as e:
                pass
            time.sleep(0.1)
            
    return pd.DataFrame(holiday_list)


if __name__ == "__main__":
    df_weather_hourly = get_hourly_weather_10years()
    df_holidays = get_holidays_10years()
    
    # 시간별 날씨(datetime 기준)에 일별 공휴일(date 기준) 병합
    df_final = pd.merge(df_weather_hourly, df_holidays, on='date', how='left')
    
    # 결측치(평일) 처리
    df_final['is_holiday'] = df_final['is_holiday'].fillna(0).astype(int)
    df_final['holiday_name'] = df_final['holiday_name'].fillna('평일')
    
    # 1. 저장할 폴더 경로 설정 (슬래시 사용)
    save_dir = "MBC_final_project_parking_prediction/features"
    
    # 2. 폴더가 없으면 자동으로 생성해 주는 마법의 코드!
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. 폴더 경로와 파일 이름을 합쳐서 최종 저장
    filename = f"{save_dir}/clean_api_features_hourly_2014_2023.csv"
    df_final.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 대성공! 총 {len(df_final):,}시간에 대한 깨끗한 데이터가 '{filename}'에 저장되었습니다!")