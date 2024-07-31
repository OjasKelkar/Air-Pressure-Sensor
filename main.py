from sensor2.exception import SensorException
from sensor2.logger import logging
import os
import sys
#from sensor2.utills import dump_csv_file_to_mongo_db
from sensor2.configuration.mongo_db import MongoDBClient
from sensor2.pipeline.training_pipeline import TrainPipeline
from sensor2.constant.application import APP_HOST, APP_PORT
from fastapi import FastAPI
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from sensor2.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor2.utills.main_utills import load_object
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, File, UploadFile, Response
from sensor2.constant.training_pipeline import SAVED_MODEL_DIR
import pandas as pd
# def test_exception():
#     try:
#         logging.info("ki yaha p bhaiaa ek error ayegi diveision by zero wali error ")
#         a=1/0
#     except Exception as e:
#        raise SensorException(e,sys) 

app = FastAPI()

origins = ["*"]
#Cross-Origin Resource Sharing (CORS) 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def train():
    try:

        training_pipeline = TrainPipeline()

        if training_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        
        training_pipeline.run_pipeline()
        return Response("Training successfully completed!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/predict")
async def predict():
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(r"C:\Users\OJAS\Documents\trainingset\ok.csv")  # Fixed path string

      

        # Resolve the best model from the saved models
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available", status_code=404)
        
        # Load the best model
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        
        # Make predictions
        y_pred = model.predict(df)
        df['predicted_column'] = y_pred
        
        # Reverse mapping for predictions
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(), inplace=True)

        # Convert the DataFrame to JSON and return as a response
        result = df.to_json(orient="records")
        return Response(content=result, media_type="application/json") 

    except Exception as e:
        logging.exception("An error occurred during prediction.")
        raise SensorException(e, sys)




def main():
    try:
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
        
    except Exception as e:
        print(e)
        logging.exception(e)
        
    


if __name__ == "__main__":

    # file_path="/Users/myhome/Downloads/sensorlive/aps_failure_training_set1.csv"
    # database_name="ineuron"
    # collection_name ="sensor"
    # dump_csv_file_to_mongodb_collection(file_path,database_name,collection_name)
    app_run(app ,host=APP_HOST,port=APP_PORT)
