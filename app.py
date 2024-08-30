import os
import joblib
from pydantic import BaseModel
from main import main_pipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from src.make_predictions import IrisInput, make_prediction


# Initialize FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Iris Model Prediction API"}


@app.get("/train")
def read_root():
    main_pipeline()
    return {"message": "Pipeline Executed successfully"}


@app.post("/predict")
def predict(iris: IrisInput):
    try:
        response = make_prediction(iris)
        return JSONResponse(content=response, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")


# # Run the application
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=80, reload=True)
