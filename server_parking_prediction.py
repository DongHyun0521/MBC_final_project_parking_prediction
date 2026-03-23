from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import uvicorn
import os
import traceback

app = FastAPI(title="🅿️ 주차 수요 예측 AI 앙상블 서버 (초단기/단기/중기)")

print("======================================")
print("🧠 AI 뇌(Random Forest 3종 세트) 로딩 중...")

base_path = "MBC_final_project_parking_prediction/features"

# 1. 3개의 맞춤형 모델과 컬럼 정보 미리 모두 로딩
models = {}
try:
    for m_type in ["vshort", "short", "mid"]:
        model_path = os.path.join(base_path, f"parking_rf_{m_type}.pkl")
        columns_path = os.path.join(base_path, f"columns_{m_type}.pkl")
        
        models[m_type] = {
            "model": joblib.load(model_path),
            "columns": joblib.load(columns_path)
        }
    print("✅ 초단기/단기/중기 3종 모델 로딩 완벽 성공! (포트 8002 대기 중)")
except Exception as e:
    print(f"🚨 모델 로딩 실패! 파일 경로를 확인하세요: {e}")
print("======================================")


# 💡 2. Java ↔ Python 완벽 매핑 통신 규약
class ParkingRequest(BaseModel):
    # 🔥 [핵심 추가] Java에서 어떤 모델을 쓸지 파이썬에게 지시하는 값
    # "vshort" (초단기), "short" (단기), "mid" (중기) 중 하나를 보냄
    forecast_type: str 

    # [1] 시간 데이터
    month: int
    dayofweek: int       
    hour: int
    is_holiday: int      

    # [2] 기상 데이터 (Optional 처리: 중기예보에서 안 보내줘도 에러 안 나게 방어)
    temp: float          
    rainfall_mm: float   
    wind_speed: float = 2.0       # 기본값 세팅
    humidity: int = 50            # 기본값 세팅
    snowfall_cm: float = 0.0      # 기본값 세팅

    # [3] 대기오염 데이터 (Optional 처리)
    pm10: float = 40.0            # 기본값 세팅
    pm25: float = 15.0            # 기본값 세팅
    pm10_grade: int = 1           
    pm25_grade: int = 1

    # [4] 예약 환자 데이터 
    예약_내과: int = 0
    예약_정형외과: int = 0
    예약_소아청소년과: int = 0
    예약_이비인후과: int = 0
    예약_신경외과: int = 0
    예약_피부과: int = 0
    예약_안과: int = 0
    예약_치과: int = 0
    예약_정신건강의학과: int = 0


# 3. 예측 API 엔드포인트
@app.post("/parking_prediction")
async def predict_parking(data: ParkingRequest):
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    weekday_str = weekdays[data.dayofweek]

    print(f"\n📥 [요청 수신] {data.month}월 {weekday_str}요일 {data.hour}시 주차 예측")
    print(f"🎯 [타겟 모델] {data.forecast_type.upper()} (Java 지정)")
    
    # 올바른 forecast_type이 들어왔는지 검증
    if data.forecast_type not in models:
        raise HTTPException(status_code=400, detail="forecast_type은 'vshort', 'short', 'mid' 중 하나여야 합니다.")

    try:
        # 1) Java가 지시한 타겟 모델과 컬럼 정보 꺼내오기
        target_model = models[data.forecast_type]["model"]
        target_columns = models[data.forecast_type]["columns"]

        # 2) 해당 모델이 필요로 하는 빈칸(0) 만들기
        input_data = {col: 0 for col in target_columns}
        
        # 3) Java에서 보내준 데이터 덮어쓰기 (공통 변수)
        if 'month' in input_data: input_data['month'] = data.month
        if 'dayofweek' in input_data: input_data['dayofweek'] = data.dayofweek
        if 'hour' in input_data: input_data['hour'] = data.hour
        if 'is_holiday' in input_data: input_data['is_holiday'] = data.is_holiday
        if 'temp' in input_data: input_data['temp'] = data.temp
        if 'rainfall_mm' in input_data: input_data['rainfall_mm'] = data.rainfall_mm
        
        # 모델별로 필요한 변수만 쏙쏙 골라서 덮어씀 (에러 완벽 차단)
        if 'wind_speed' in input_data: input_data['wind_speed'] = data.wind_speed
        if 'humidity' in input_data: input_data['humidity'] = data.humidity
        if 'snowfall_cm' in input_data: input_data['snowfall_cm'] = data.snowfall_cm
        if 'pm10' in input_data: input_data['pm10'] = data.pm10
        if 'pm25' in input_data: input_data['pm25'] = data.pm25
        if 'pm10_grade' in input_data: input_data['pm10_grade'] = data.pm10_grade
        if 'pm25_grade' in input_data: input_data['pm25_grade'] = data.pm25_grade
        
        if '예약_내과' in input_data: input_data['예약_내과'] = data.예약_내과
        if '예약_정형외과' in input_data: input_data['예약_정형외과'] = data.예약_정형외과
        if '예약_소아청소년과' in input_data: input_data['예약_소아청소년과'] = data.예약_소아청소년과
        if '예약_이비인후과' in input_data: input_data['예약_이비인후과'] = data.예약_이비인후과
        if '예약_신경외과' in input_data: input_data['예약_신경외과'] = data.예약_신경외과
        if '예약_피부과' in input_data: input_data['예약_피부과'] = data.예약_피부과
        if '예약_안과' in input_data: input_data['예약_안과'] = data.예약_안과
        if '예약_치과' in input_data: input_data['예약_치과'] = data.예약_치과
        if '예약_정신건강의학과' in input_data: input_data['예약_정신건강의학과'] = data.예약_정신건강의학과
        
        # 🔥 가장 중요한 핵심 피처 자동 계산
        if '예약_총외래환자' in input_data:
            input_data['예약_총외래환자'] = (
                data.예약_내과 + data.예약_정형외과 + data.예약_소아청소년과 + 
                data.예약_이비인후과 + data.예약_신경외과 + data.예약_피부과 + 
                data.예약_안과 + data.예약_치과 + data.예약_정신건강의학과
            )
            
        # 4) AI 예측 실행
        df_input = pd.DataFrame([input_data])
        df_input = df_input[target_columns] # 컬럼 순서 완벽 정렬
        
        prediction = target_model.predict(df_input)[0]
        result_cars = int(prediction)
        
        print(f"✨ [예측 성공] 결과: {result_cars}대 (사용 모델: {data.forecast_type})")
        
        return {
            "status": "success",
            "forecast_type": data.forecast_type,
            "predicted_cars": result_cars,
            "message": f"성공 ({result_cars}대)"
        }
        
    except Exception as e:
        print(f"🚨 [에러]: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("server_parking_prediction:app", host="0.0.0.0", port=8002, reload=True)