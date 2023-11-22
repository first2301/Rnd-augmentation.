# https://docs.streamlit.io/knowledge-base/deploy/increase-file-uploader-limit-streamlit-cloud
# https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options
# 실행방법1: streamlit run recomendation.py
# 실행방법2: streamlit run recomendation.py --server.maxUploadSize 500 --server.maxMessageSize 500 (업로드 파일 용량 증대할 경우)
# import time
import ray
import json
import requests
import pandas as pd
import streamlit as st
# 머신러닝 모델
# from model_lib.pycaret_anomaly import *
# from model_lib.pycaret_clf import *

st.set_page_config(layout="wide")
st.sidebar.title("Details")

# 분류, 이상 탐지 등 추천받을 머신러닝 모델 선택
option = st.sidebar.selectbox(
    'Select Machine Task',
    ('분류', '이상탐지'))

uploaded_file = st.sidebar.file_uploader("csv file upload", type="csv") # 파일 업로드

# @st.cache_data # resource
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

with st.spinner('Wait for it...'):
    if uploaded_file is None: 
        st.write(
        '''
        ### 머신러닝 실행 방법
        * 분류
        1. Upload csv file 
        2. Select Target column 
        3. Drop cloumns
        4. 제거할 Target 데이터 선택

        * 이상 탐지
        1. Upload csv file
        2. 머신러닝 테스트 실행
        ''')

    updated_df = None
    # Uploaded data Dashboard
    target_feture = "" # 예측할 Label

    if uploaded_file is not None:
        st.subheader('데이터 분석')
        df = load_data(uploaded_file)
        col_list = df.columns.tolist() # 데이터 전처리 옵션 설정 리스트

        target_feture = ""
        if option == '분류':
            target_feture = st.sidebar.multiselect('Select Target Column', options=col_list)

        data_to_drop = st.sidebar.multiselect('Drop Cloumns', options=col_list)

        tab_eda_df, tab_eda_info, tab_Label_counts = st.tabs(['Original data', 'Null information', 'Target Data Counts']) # tab_Label_counts Labels counts

        with tab_eda_df:
            st.write('Original data')
            st.dataframe(df)
        with tab_eda_info:
            st.write('Null information')
            info_df = pd.DataFrame({'Column names': df.columns,
                                    'Non-Null Count': df.count(),
                                    'Null Count': df.isnull().sum(),
                                    'Dtype': df.dtypes,
                                    })
            info_df.reset_index(inplace=True)
            st.write(info_df.iloc[:, 1:].astype(str))

        label_to_drop = ""
        with tab_Label_counts: # Target Data 정보 출력 및 시각화
            val_counts_df = None
            if target_feture:            
                test = df[target_feture].value_counts().reset_index()
                val_counts_df = pd.DataFrame({'Labels': test.iloc[:, 0],
                                            'Counts': test.iloc[:, 1]})
                st.write(val_counts_df)
                bar_data = val_counts_df
                bar_data.index = val_counts_df['Labels']
                st.bar_chart(bar_data['Counts'])

                # Target Data 설정해야 제거할 Label 선택 가능
                label_to_drop = st.sidebar.multiselect('제거할 Target 데이터 선택', options=val_counts_df.iloc[:, 0])
            else:
                sample_df = pd.DataFrame({'Label': ['Select Label Column'], # Sample Data
                                        'Counts': ['Select Label Column']})
                st.write(sample_df)

        # 선택한 Column 제거
        
        if data_to_drop:
            for data in data_to_drop:
                updated_df = df.drop(data_to_drop, axis=1)

        # 선택한 Target Data 제거

        try:
            if label_to_drop:
                target_feture = target_feture[0]
                label_to_drop = label_to_drop[0]
                updated_df = df[df[target_feture] != label_to_drop]
        except ValueError:
            st.write('1개 이상 데이터가 남아있어야 합니다.')

        # 데이터 전처리된 데이터 출력
        if updated_df is not None: 
            st.subheader('데이터 전처리')
            st.dataframe(updated_df)
        
        if st.sidebar.button("초기화"):
            st.cache_resource.clear()

        button_for_training = st.sidebar.button("머신러닝 테스트 실행", key="button1") 
        if button_for_training: # 분류, 이상탐지 옵션에 따라 머신러닝 학습 진행
            # start_time = time.time() # 학습 시간 체크 시 설정

            if option == '분류':
                st.subheader('머신러닝 학습 결과')
                with st.spinner('Wait for it...'):
                    if updated_df is None:
                        updated_df = df   
                    json_data = updated_df.to_json() # pandas DataFrame를 json 형태로 변환
                    data_dump = json.dumps({'json_data':json_data, 'target': target_feture}) # 학습 데이터, Target Data 객체를 문자열로 직렬화(serialize)
                    data = json.loads(data_dump) # json을 파이썬 객체로 변환

                    response = requests.post('http://127.0.0.1:8001/clf', json=data) # NIPA 서버로 머신러닝 학습데이터 request
                    if response.status_code == 200: 
                        json_data = response.json() # NIPA 서버에서 학습한 데이터를 json으로 response 
                        model_compare_clf = json_data['result'] 

                        tab_line, tab_bar = st.tabs(['Line Chart', 'Bar Chart']) # 분류모델 학습 결과 시각화
                        with tab_line:
                            st.subheader('Line Chart')
                            st.line_chart(model_compare_clf)
                        with tab_bar:
                            st.subheader('Bar Chart')
                            st.bar_chart(model_compare_clf)

                        st.subheader('Score')
                        st.dataframe(model_compare_clf)
                        ray.shutdown() # 머신러닝 모델 분산 학습 종료
           
            if option == '이상탐지':
                st.subheader('머신러닝 학습 결과')
                with st.spinner('Wait for it...'):
                    if updated_df is None:
                        updated_df = df
                    json_data = updated_df.to_json() # pandas DataFrame를 json 형태로 변환
                    data = json.loads(json_data) # json을 파이썬 객체로 변환
                    
                    response = requests.post('http://127.0.0.1:8001/anomaly', json=data)  # NIPA 서버로 머신러닝 학습데이터 request

                    if response.status_code == 200:
                        # st.write('정상 데이터 평균 점수가 낮은 순서로 추천')
                        updated_json = response.json() # NIPA 서버에서 학습한 데이터를 json으로 response 
                        # st.write(updated_json)
                        zero_df = pd.DataFrame(updated_json['json_zero'])
                        zero_df = zero_df.sort_values('novelty_mean_score', ascending=True)
                        one_df = pd.DataFrame(updated_json['json_one'])
                        one_df = one_df.sort_values('anomaly_mean_score', ascending=True)
                        
                        col1, col2 = st.columns(2) # 이상탐지 모델 결과 시각화
                        with col1:
                            st.write('정상 데이터', zero_df)
                        with col2:
                            st.write('이상 데이터', one_df)
                        score_df = pd.DataFrame(updated_json['json_score'])
                        st.write('평균 점수가 낮은 순서로 시각화')
                        for idx, model_name in enumerate(zero_df['model_name']): 
                            st.write(model_name)
                            st.line_chart(score_df[model_name])

                    # end_time = time.time()
                    # execution_time = end_time - start_time
                    # minutes, seconds = divmod(execution_time, 60)
                    # print(f"코드 실행 시간: {int(minutes)}분 {seconds:.2f}초")
                    ray.shutdown() # 머신러닝 모델 분산 학습 종료