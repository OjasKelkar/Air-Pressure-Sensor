import pandas as pd
import numpy as np
import logging
import json
from sensor2.config import mongo_clent



def dump_csv_file_to_mongo_db(file_path:str,database_name:str,collection_name:str)->None:
    try:
        df = pd.read_csv(file_path)
        df.reset_index(True,inplace=True)
        df.T.to_json()
        json_records = list(json.loads(df.T.to_json()).values())
        
        
        mongo_clent[database_name][collection_name].insert_many(json_records)
        
        
        
    except Exception as e:
        print(e)