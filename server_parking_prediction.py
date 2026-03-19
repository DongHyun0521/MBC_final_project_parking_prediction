from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import os

app = FastAPI(title="🅿️ 주차 수요 예측 AI 서버 (Random Forest)")

print("======================================")
print("🧠 AI 뇌(Random Forest) 로딩 중...")

# 💡 1. 아까 저장해둔 AI 모델과 문제지(컬럼) 뼈대를 불러옵니다!
base_path = "MBC_final_project_parking_prediction/features" # (경로가 안 맞으면 절대경로로 수정하세요!)
model_path = os.path.join(base_path, "parking_rf_model.pkl")
columns_path = os.path.join(base_path, "model_columns.pkl")

try:
    rf_model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    print("✅ AI 예측 모델 로딩 완료! (포트 8002 대기 중)")
except Exception as e:
    print(f"🚨 모델 로딩 실패! 경로를 확인하세요: {e}")
print("======================================")

# 💡 2. Java 백엔드(또는 프론트)에서 파이썬으로 보내줄 데이터의 양식(규칙)을 정합니다.
class ParkingRequest(BaseModel):
    month: int
    dayofweek: int       # 0:월, 1:화 ... 6:일
    hour: int
    temp: float
    rainfall_mm: float
    PM10: float
    is_sunny: int        # 맑음이면 1, 아니면 0
    # 중요한 예약 환자수 데이터
    예약_내과: int
    예약_정형외과: int
    예약_소아청소년과: int
    예약_이비인후과: int
    예약_신경외과: int
    예약_피부과: int
    예약_안과: int
    예약_치과: int
    예약_정신건강의학과: int

# 💡 3. API 엔드포인트 오픈! (누군가 데이터를 던지면 예측해서 돌려줌)
@app.post("/parking_prediction")
async def predict_parking(data: ParkingRequest):
    print(f"📥 [요청 수신] {data.month}월 {data.hour}시 주차 예측 요청이 들어왔습니다.")
    
    try:
        # 1) AI가 학습했던 문제지(0으로 채워진 빈칸)를 똑같이 만듭니다.
        input_data = {col: 0 for col in model_columns}
        
        # 2) Java에서 보내준 데이터로 빈칸을 채워 넣습니다.
        input_data['month'] = data.month
        input_data['dayofweek'] = data.dayofweek
        input_data['hour'] = data.hour
        input_data['temp'] = data.temp
        input_data['rainfall_mm'] = data.rainfall_mm
        input_data['PM10'] = data.PM10
        
        input_data['예약_내과'] = data.예약_내과
        input_data['예약_정형외과'] = data.예약_정형외과
        input_data['예약_소아청소년과'] = data.예약_소아청소년과
        input_data['예약_이비인후과'] = data.예약_이비인후과
        input_data['예약_신경외과'] = data.예약_신경외과
        input_data['예약_피부과'] = data.예약_피부과
        input_data['예약_안과'] = data.예약_안과
        input_data['예약_치과'] = data.예약_치과
        input_data['예약_정신건강의학과'] = data.예약_정신건강의학과
        
        # 날씨 원-핫 인코딩 처리
        weather_cols = [c for c in model_columns if '맑음' in c]
        if weather_cols and data.is_sunny == 1:
            input_data[weather_cols[0]] = 1
            
        # 3) 판다스 데이터프레임으로 변환
        df_input = pd.DataFrame([input_data])
        
        # 4) AI 예측 실행!
        prediction = rf_model.predict(df_input)[0]
        result_cars = int(prediction)
        
        print(f"✨ [예측 완료] 예상 주차 대수: {result_cars}대")
        
        # 5) 결과를 다시 Java(또는 프론트)로 예쁘게 JSON 포장해서 리턴!
        return {
            "status": "success",
            "predicted_cars": result_cars,
            "message": f"성공적으로 예측했습니다. ({result_cars}대)"
        }
        
    except Exception as e:
        print(f"🚨 [파이썬 서버 에러]: {str(e)}")
        return {"status": "error", "message": str(e)}

# 번호판 인식(8001)과 충돌하지 않게 8002번 포트 사용!
if __name__ == "__main__":
    uvicorn.run("parking_prediction_server:app", host="0.0.0.0", port=8002, reload=True)