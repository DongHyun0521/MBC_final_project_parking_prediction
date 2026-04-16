"""
주차 수요 예측 AI 서버 (port 8002)

Random Forest 모델 3종(초단기/단기/중기)으로 시간대별 입차 대수 예측.
Spring Boot 스케줄러가 기상/예약 데이터와 함께 호출.
"""
import os
import traceback

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ═══════════════════════════════════════════════════════════════════
# 모델 로드 (서버 기동 시 1회)
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")

models = {}
for m_type in ["vshort", "short", "mid"]:
    models[m_type] = {
        "model":   joblib.load(os.path.join(BASE_DIR, f"parking_rf_{m_type}.pkl")),
        "columns": joblib.load(os.path.join(BASE_DIR, f"columns_{m_type}.pkl")),
    }
    print(f"  Random Forest ({m_type}): 로드 완료")

# ═══════════════════════════════════════════════════════════════════
# 요청 스키마 (Spring Boot → Python)
# ═══════════════════════════════════════════════════════════════════
class ParkingRequest(BaseModel):
    forecast_type: str              # "vshort" | "short" | "mid"
    target_datetime: str = ""       # "2026-03-29 14:30"

    # 시간 피처
    month: int
    dayofweek: int
    hour: int
    minute: int
    is_holiday: int

    # 기상 피처
    temp: float
    rainfall_mm: float
    wind_speed: float = 2.0
    humidity: int = 50
    snowfall_cm: float = 0.0

    # 대기질 피처
    pm10: float = 40.0
    pm25: float = 15.0
    pm10_grade: int = 1
    pm25_grade: int = 1

    # 9개 진료과 예약 피처
    예약_내과: int = 0
    예약_정형외과: int = 0
    예약_소아청소년과: int = 0
    예약_이비인후과: int = 0
    예약_신경외과: int = 0
    예약_피부과: int = 0
    예약_안과: int = 0
    예약_치과: int = 0
    예약_정신건강의학과: int = 0

# 요청 필드 → 모델 컬럼 매핑에 사용할 필드 목록
_REQUEST_FIELDS = [
    'month', 'dayofweek', 'hour', 'minute', 'is_holiday',
    'temp', 'rainfall_mm', 'wind_speed', 'humidity', 'snowfall_cm',
    'pm10', 'pm25', 'pm10_grade', 'pm25_grade',
    '예약_내과', '예약_정형외과', '예약_소아청소년과', '예약_이비인후과',
    '예약_신경외과', '예약_피부과', '예약_안과', '예약_치과', '예약_정신건강의학과',
]

_DEPT_FIELDS = [
    '예약_내과', '예약_정형외과', '예약_소아청소년과', '예약_이비인후과',
    '예약_신경외과', '예약_피부과', '예약_안과', '예약_치과', '예약_정신건강의학과',
]

# ═══════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════
app = FastAPI()


@app.post("/parking_prediction")
async def predict_parking(data: ParkingRequest):
    if data.forecast_type not in models:
        raise HTTPException(status_code=400, detail="forecast_type: 'vshort', 'short', 'mid' 중 하나")

    try:
        target_model   = models[data.forecast_type]["model"]
        target_columns = models[data.forecast_type]["columns"]

        # 모델이 필요로 하는 컬럼만 0으로 초기화 후, 요청 데이터로 덮어쓰기
        input_data = {col: 0 for col in target_columns}
        for field in _REQUEST_FIELDS:
            if field in input_data:
                input_data[field] = getattr(data, field)

        # 총외래환자 = 9개 진료과 합산 (파생 피처)
        if '예약_총외래환자' in input_data:
            input_data['예약_총외래환자'] = sum(getattr(data, f) for f in _DEPT_FIELDS)

        # 예측 실행
        df = pd.DataFrame([input_data])[target_columns]
        predicted_cars = int(target_model.predict(df)[0])

        weekdays = ["월", "화", "수", "목", "금", "토", "일"]
        print(f"  [{data.forecast_type}] {data.target_datetime} ({weekdays[data.dayofweek]}) → {predicted_cars}대")

        return {
            "status": "success",
            "forecast_type": data.forecast_type,
            "predicted_cars": predicted_cars,
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
