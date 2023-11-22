# uvicorn main:app --reload
# uvicorn main:app --reload --port 8001
from fastapi import FastAPI, Request
from anomaly_lib import *
from pycaret_clf import *
from pycaret.anomaly import AnomalyExperiment
from pycaret.classification import ClassificationExperiment
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
import ray

app = FastAPI()

@app.get('/')
def main():
    return 'test'

@app.post('/clf/')
async def clf_test(request: Request): # dict
    data = await request.json() # 데이터를 비동기 방식으로 load
    # df = request['json_data']
    df = pd.read_json(data['json_data'])
    target = data['target']

    # ray를 활용한 머신러닝 분산 학습
    if not ray.is_initialized():
        ray.init()
    # model_compare_clf = train_caret_clf(df, df[target])
    model_compare_task = train_caret_clf.remote(df, df[target]) # 분류 모델 객체 선언
    model_compare_clf = ray.get(model_compare_task) # 분류 모델 분산 학습
    ray.shutdown()  # 머신러닝 모델 분산 학습 종료
    df_json = model_compare_clf.to_json() # 학습결과를 json으로 변환
    result_data = json.loads(df_json) # json을 파이썬 객체로 변한
    return JSONResponse(content={'result': result_data})

@app.post('/anomaly/')
async def anomaly_test(request: Request):
    data = await request.json() # 데이터를 비동기 방식으로 load
    result = train_caret_anomaly(data) # 이상탐지 모델 학습
    
    zero_df = pd.DataFrame(columns=['model_name', 'novelty', 'novelty_counts', 'novelty_mean_score'])
    one_df = pd.DataFrame(columns=['model_name', 'anomaly', 'anomaly_counts', 'anomaly_mean_score'])
    score_df = pd.DataFrame()
    dataframe = []
    for index, model_name in enumerate(result.keys()):
        df_to_json = result[model_name]['Anomaly_Score'].to_json()
        dataframe.append(json.loads(df_to_json))

        score_series = result[model_name]['Anomaly_Score']
        score_df = pd.concat([score_df, pd.DataFrame({model_name: score_series})], axis=1)
        

        try:
            key_zero = result[model_name]['Anomaly'].value_counts().keys()[0]
            value_zero = result[model_name]['Anomaly'].value_counts().values[0] 
            novelty_mean_score = np.mean(result[model_name].query('Anomaly == 0')['Anomaly_Score']) # 정상 데이터 평균
            novelty_std_score = np.std(result[model_name].query('Anomaly == 0')['Anomaly_Score']) # 정상 데이터 표준편차
            zero_df = pd.concat([zero_df, pd.DataFrame({'model_name':model_name, 'novelty':key_zero, 'novelty_counts':value_zero, 
                                                        'novelty_mean_score': novelty_mean_score, 'novelty_std_score': novelty_std_score, }, index=[index])])
        except IndexError:
            continue
        try:
            key_one = result[model_name]['Anomaly'].value_counts().keys()[1]
            value_one = result[model_name]['Anomaly'].value_counts().values[1]
            anomaly_mean_score = np.mean(result[model_name].query('Anomaly == 1')['Anomaly_Score']) # 이상 데이터 평균
            anomaly_std_score = np.std(result[model_name].query('Anomaly == 1')['Anomaly_Score']) # 이상 데이터 표준편차
            one_df = pd.concat([one_df, pd.DataFrame({'model_name':model_name, 'anomaly':key_one, 'anomaly_counts':value_one, 
                                                        'anomaly_mean_score': anomaly_mean_score, 'anomaly_std_score': anomaly_std_score}, index=[index])])
        except IndexError:
            continue
    temp_score = score_df.to_json()
    temp_zero = zero_df.to_json()
    temp_one = one_df.to_json()
    json_score = json.loads(temp_score) # 이상치 score
    json_zero = json.loads(temp_zero) # 정상 데이터
    json_one = json.loads(temp_one) # 이상 데이터
    return JSONResponse(content={'json_zero':json_zero, 'json_one': json_one, 
                                 'dataframe': dataframe, 'json_score': json_score})

# /home/work/AI/test/recomendation.py