# https://docs.streamlit.io/knowledge-base/deploy/increase-file-uploader-limit-streamlit-cloud
# https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options
# 실행방법1: streamlit run gene.py
# 실행방법2: streamlit run gene.py --server.maxUploadSize 500 --server.maxMessageSize 500 (업로드 파일 용량 증대할 경우)
# streamlit 1.24.0 이상 버전에서 파일 업로드할 경우 AxiosError: Request failed with status code 403 발생할 수 있음
# AxiosError 403 에러 발생 시 streamlit==1.24.0 버전으로 변경
# pip install streamlit==1.24.0 or 1.26.0

import json
import requests
import pandas as pd
import streamlit as st
from tabs.tab_vis import *
from gene_lib.sampling_lib import *
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 
from io import StringIO
import time

st.set_page_config( # 레이아웃 설정
        page_title="Data generator",
        layout="wide"
    )

with st.spinner('Wait for it...'): # 로딩이 완료되지 않으면 "Wair for it..."이  UI에 등장
    
    st.sidebar.title("Details") # sidebar
    uploaded_file = st.sidebar.file_uploader("csv file upload", type="csv") # 파일 업로드
    if uploaded_file is None: 
        st.write(
        '''
        ### 데이터증강 실행 방법
        1. Upload csv file
        2. Select Target Column 
        3. Drop cloumns
        4. Target data 정수 인코딩
        5. 제거할 Target 데이터 선택
        ''')
                 
    # @st.cache_data
    def load_data(uploaded_file):
        return pd.read_csv(uploaded_file)
    
    # 변수 선언 및 변수 초기화
    updated_df = None
    feature_selection = None
    target_feature = "" # 예측할 Label
    # target_feature = [0] # target_feature가 tuple로 감싸져 있어서 인덱싱
    le = LabelEncoder() # LabelEncoder 객체 선언 
    sampling_threshold = 0 # Binary class 임계값 초기화 
#################### sidebar
    if uploaded_file is not None: # 파일을 업로드해야 데이터전처리 옵션설정 가능
        st.subheader('데이터 분석')
        df = load_data(uploaded_file)
        col_list = df.columns.tolist() # multiselect list

        # 데이터 전처리 옵션 설정
        target_feature = st.sidebar.multiselect('Select Target Column', options=col_list) # 타겟 데이터 선택(필수)
        drop_columns = st.sidebar.multiselect('Drop Cloumns', options=col_list) # 불필요한 컬럼 제거
        data_for_labelencoding = st.sidebar.multiselect('Target data 정수 인코딩', options=col_list) # 타겟 데이터가 str인 경우, 선택
        feature_selection = st.sidebar.multiselect('Target data 유형 선택', options=['Multiclass', 'Binary class']) # 타겟 데이터의 필드값이 여러개인 경우 Multiclass, 2개인 경우 Binary class
        
        initial_value = 50
        sampling_threshold = st.sidebar.slider('Binary Class 임계값 설정', 0, 100, initial_value)

        # Binary class 선택할 때만 slider 등장하게 할 경우 각주 해제
        # if feature_selection: 
        #     feature_selection[0] == 'Binary class'
        #     initial_value = 50
        #     sampling_threshold = st.sidebar.slider('Binary Class 임계값 설정', 0, 100, initial_value)

#################### Original Data tabs
        tab_raw_df, tab_null_info, tab_label_counts = st.tabs(['Original data', 'Null information', 'Target counts'])
        with tab_raw_df: # Original data tab
            st.subheader("Original data")
            st.dataframe(df, use_container_width=True)

        with tab_null_info: # null information tab
            eda_null_info(df) # tabs > tab_vis (디렉토리에 있는 경로 확인)

        label_to_drop = ""
        val_counts_df = None
        with tab_label_counts: # Target data counts tab
            st.write("Label counts")
            val_counts_df = None
            if target_feature:            
                val_counts = df[target_feature].value_counts().reset_index()
                val_counts_df = pd.DataFrame({'Labels': val_counts.iloc[:, 0],
                                            'Counts': val_counts.iloc[:, 1]})
                
                st.dataframe(val_counts_df, use_container_width=True)
                # Target Data 설정해야 제거할 Label 선택 가능
                label_to_drop = st.sidebar.multiselect('제거할 Target 데이터 선택', options=val_counts_df.iloc[:, 0])
                bar_data = val_counts_df
                bar_data.index = val_counts_df['Labels']
                st.bar_chart(bar_data['Counts'])
            else:
                sample_df = pd.DataFrame({'Label': ['Select Target Column'],
                                        'Counts': ['Select Target Column']})
                st.write(sample_df) # Target데이터 선택
 
#################### Target Label 삭제      
    
        try:
            if label_to_drop:                                                                    # 제거할 Target label data
                if updated_df is None:
                    target_feature = target_feature[0]
                    label_to_drop = label_to_drop[0]
                    updated_df = df[df[target_feature] != label_to_drop]                         # 제거할 데이터만 제외하고 데이터프레임 업데이트
                if updated_df is not None: 
                    for drop_label in label_to_drop:
                        updated_df = updated_df[updated_df[target_feature] != drop_label]
        except ValueError:
            st.write('1개 이상 Label이 남아있어야 합니다.')

#################### LabelEncoding     
    
        label_col_name = ""                                                                     # Label Data
        if data_for_labelencoding:                                                              # 데이터프레임에 str 타입이 있는 경우, int 타입으로 정수 인코딩
            label_col_name = data_for_labelencoding[0]   
            if updated_df is None:   
                if label_col_name == target_feature:
                    df[target_feature] = le.fit_transform(df[target_feature])                   # fit_transform 사용할 경우, 학습과 인코딩 동시에 가능
                df[label_col_name] = le.fit_transform(df[label_col_name])                       # target 데이터가 아닌 str 타입의 데이터 정수 인코딩
                updated_df = df

            if updated_df is not None:
                if label_col_name == target_feature:
                    updated_df[target_feature] = le.fit_transform(updated_df[target_feature])
                updated_df[label_col_name] = le.fit_transform(updated_df[label_col_name])

#################### 제거할 column 데이터

        try:
            if drop_columns:
                if updated_df is None:
                    updated_df = df.drop(drop_columns, axis=1)
                else:
                    updated_df = updated_df.drop(drop_columns, axis=1)
        except ValueError:
            st.write('1개 이상 데이터가 남아있어야 합니다.')
                
#################### generator_button & Clear

        generator_button = st.sidebar.button('데이터 증강')

        if st.sidebar.button("초기화"):
            st.cache_resource.clear()

############### preprocessing
        sampling_df = None
        if updated_df is not None: 
            st.subheader('데이터 전처리')
            st.dataframe(updated_df)


#################### sampling strategy
        try:
            if generator_button:
                start_time = time.time()
                if updated_df is None:
                    updated_df = df
                
                st.subheader('Generated Data')
                with st.spinner('Wait for it...'):
                    target = target_feature

                    selected_feature_df = None
                    sampling_strategy = None
                    thresh_ratio = (sampling_threshold / 100)
                    thresh = len(df) * thresh_ratio

                    if feature_selection:
                        if feature_selection[0] == 'Multiclass':
                            selected_feature_df = label_feature_selection(updated_df[target_feature], updated_df)
                            # sampling_strategy = thresh  
                        if feature_selection[0] == 'Binary class':
                            sampling_strategy = thresh  

                    if selected_feature_df is not None: # Multiclass인 경우
                        json_data = selected_feature_df.to_json() 
                        data_dump = json.dumps({'json_data':json_data, 'target': target_feature})
                        data = json.loads(data_dump) 
                        response = requests.post('http://127.0.0.1:8009/aug', json=data)
                        if response.status_code == 200: 
                            json_data = response.json() 
                            json_result = json_data['result'] 
                            sampling_strategy = {int(key): int(value) for key, value in json_result.items()}

                        # grid_result = make_grid(selected_feature_df, target_feature) 
                        # sampling_strategy = make_ratio(grid_result) 
                    if selected_feature_df is None: # Binary class인 경우
                        label_df = val_counts_df[val_counts_df['Counts'] == min(val_counts_df['Counts'])]           # Binary class: 가장 작은 값 선택
                        sampling_strategy = {label_df['Labels'].iloc[0]: round(thresh)}                             # 기본 설정값 2:1 비율, threshold 설정 값에 따라 조절 가능
                        
    #################### Oversampling Start
                    # 데이터 증강은 일반 지도학습과 같은 절차로 진행되기 때문에, 속성값을 X인 입력변수로, target을 y인 목표 변수로 설정
                    
                    X, y = updated_df.drop(target, axis=1), updated_df[target] 
                    sampling_df = None # sampling_df 초기화
                    X_over_resampled, y_over_resampled = SMOTE(sampling_strategy=sampling_strategy).fit_resample(X, y)          # sampling_strategy 옵션: 'auto', sampling_strategy
                    sampling_df = X_over_resampled
                    sampling_df[target] = y_over_resampled

    #################### 증강된 데이터 전, 후 비교 출력
                    # st.write(target)
                    df_before = updated_df
                    df_before = df_before.drop(target, axis=1)
                    df_after = sampling_df
                    df_after = df_after.drop(target, axis=1)
                    
                    compare_tab_data = {'before_data': df_before.to_json(), 'after_data': df_after.to_json()}
                    compare_dump_data = json.dumps(compare_tab_data)
                    compare_json_data = json.loads(compare_dump_data) 
                    compare_response = requests.post('http://127.0.0.1:8009/compare', json=compare_json_data)
                    if compare_response.status_code == 200:
                        compare_response_data = compare_response.json()
                        compare_result_data = compare_response_data['compare_result'] 

                        tab_names = list(compare_result_data.keys())
                        tabs = st.tabs(tab_names)

                        for idx, tab_name in enumerate(tab_names):
                            with tabs[idx]:
                                df_data = pd.read_json(StringIO(compare_result_data[tab_name]))
                                st.area_chart(df_data, color=['#7cfc00','#00bfff'] ) # , color=['#7cfc00','#00bfff'] 

        #################### 증강한 데이터 출력
                    original_data, oversampling_data = st.columns(2) 
                    with original_data:
                        original_tab(df, target_feature)  # 시각화 변경 시 tab_vis.py 코드 참고

                    with oversampling_data:
                        sampling_tab(sampling_df, target) # 시각화 변경 시 tab_vis.py 코드 참고
                
                end_time = time.time()
                execution_time = end_time - start_time  # 실행 시간 계산
                print(f"코드 실행 시간: {execution_time} 초")
        # 데이터 전처리가 잘못 되었을 경우, 아래 설명 출력
        except ValueError as e:
            st.write(e)
            st.write('최소 2개 이상 Label이 있어합니다.')
            st.write('Target Label이 1개인 경우, 제거해야합니다.')
            st.write('데이터 전처리가 완료되어야 합니다.')
            st.write('Binary class인 경우, Multiclass가 적용되지 않습니다.')            
            # st.write(e)
        except AttributeError as e:
            st.write(e)
            st.write('전처리가 완료되어야 합니다.')
            # st.write(e)

        # 증강한 데이터 csv로 다운로드
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        csv = None
        if sampling_df is not None:
            csv = convert_df(sampling_df)
        
        if csv is not None:
            st.sidebar.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sampling_data.csv',
                mime='text/csv',
            )
    
