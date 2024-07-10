# fastapi 서버 실행 명령어: uvicorn main:app --reload --port 8009 (port 번호 변경 시, gene.py에 있는 port 번호도 함께 수정)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd
import json
from gene_lib.sampling_lib import *
from io import StringIO
app = FastAPI()

@app.get('/')
def main():
    return 'main'

@app.post('/aug/')
async def clf_test(request: Request): # dict
    data = await request.json()

    selected_feature_df = pd.read_json(StringIO(data['json_data']))
    target_feature = data['target']
    
    grid_result = make_grid(selected_feature_df, target_feature)    # Grid search를 통한 feature selection. Multiclass에서 증강할 필드값 선택 시 적용
    sampling_strategy = make_ratio(grid_result)                     # Grid search로 도출한 데이터 증강 비율을 sampling_strategy에 적합한 형태로 입력 
    result_dump = json.dumps(sampling_strategy)
    result_data = json.loads(result_dump)                           # json을 파이썬 객체로 변한
    
    return JSONResponse(content={'result': result_data})

@app.post('/compare')
async def compare_aug_data(request: Request):
    data = await request.json()
    before_data = data['before_data']
    after_data = data['after_data']
    df_before = pd.read_json(StringIO(before_data))
    df_after = pd.read_json(StringIO(after_data))
    column = df_before.columns.intersection(df_after.columns)

    compare_result = {}
    for col in column:
        df_data = pd.DataFrame({col: pd.Series(df_before[col]),  # before_series
                                'sampled_' + col: pd.Series(df_after[col])}) # after_series
        compare_result[col] = df_data.to_json()

    df_dump = json.dumps(compare_result)
    df_data = json.loads(df_dump)
    return JSONResponse(content={'compare_result': df_data})
