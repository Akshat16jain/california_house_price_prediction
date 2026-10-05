# API for predicting California house prices
# Run with: uvicorn main:app --reload

import io

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

# load the files saved by train.py
model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")

# rough average error of the model in dollars (from train.py)
ERROR = 39000


# what the user sends to /predict
class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income of the area")
    HouseAge: float = Field(gt=0, description="Average age of houses")
    AvgRoom: float = Field(gt=0, description="Average number of rooms")
    AvgBedroom: float = Field(gt=0, description="Average number of bedrooms")
    Population: float = Field(gt=0, description="Population of the area")
    AvgOccupation: float = Field(gt=0, description="Average number of people per house")
    Latitude: float = Field(gt=32, le=42, description="Latitude")
    Longitude: float = Field(gt=-125, le=-114, description="Longitude")


@app.get("/")
def home():
    return {
        "message": "California house price API",
        "status": "running",
        "how_to_use": "send a POST request to /predict",
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": f"${ERROR:,}",
    }


@app.post("/predict")
def predict(house: HouseFeatures):
    # column names must match the ones used in training
    row = pd.DataFrame([{
        "MedInc": house.MedInc,
        "HouseAge": house.HouseAge,
        "AveRooms": house.AvgRoom,
        "AveBedrms": house.AvgBedroom,
        "Population": house.Population,
        "AveOccup": house.AvgOccupation,
        "Latitude": house.Latitude,
        "Longitude": house.Longitude,
    }])

    try:
        result = model.predict(row)[0]   # price in $100,000
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    price = result * 100000
    return {
        "predicted_price": f"${price:,.0f}",
        "predicted_price_short": f"{result:.2f} (x $100,000)",
        "likely_range": f"${price - ERROR:,.0f} to ${price + ERROR:,.0f}",
    }


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="The file is empty")

    # check that all needed columns are there
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    try:
        prediction = model.predict(df[features])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    df["predicted_price_usd"] = prediction * 100000

    output = df.to_csv(index=False)
    return StreamingResponse(
        io.StringIO(output),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
