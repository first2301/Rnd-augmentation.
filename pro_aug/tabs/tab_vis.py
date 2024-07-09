import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

def original_tab(df, target): # 원본 데이터 분포 확인 및 시각화
    original_target_series = df[target]
    original_value_counts = original_target_series.value_counts()

    n_df = pd.DataFrame(original_value_counts.reset_index()) # original_value counts 
    bar_fig = px.bar(n_df, x=n_df.iloc[:, 0], y=n_df.iloc[:, 1], title='Bar Chart Original Data') # bar chart data
    fig = px.pie(n_df, values=n_df.iloc[:, 1], names=n_df.iloc[:, 0], title='Pie Chart Sampling Data') # pie chart data
    
    st.subheader("Original Data")
    st.plotly_chart(bar_fig) # bar chart 시각화
    st.write("Original Data Total Counts: ", len(df))
    st.dataframe(original_value_counts, use_container_width=True)
    st.plotly_chart(fig) # pie chart 시각화
    st.write("Original DataFrame", df) # dataframe 출력

def sampling_tab(sampling_df, target): # 증강된 데이터 분포 확인 및 시각화
    updated_target_series = sampling_df[target]
    updated_value_counts = updated_target_series.value_counts()

    n_df = pd.DataFrame(updated_value_counts.reset_index()) # sampling data value counts 
    bar_fig = px.bar(n_df, title='Bar Chart Sampling Data') # bar chart data
    sampling_fig = px.pie(n_df, values=n_df.iloc[:, 1], names=n_df.iloc[:, 0], title='Pie Chart Sampling Data')  # pie chart data
    
    st.subheader('OverSampling Data')
    st.plotly_chart(bar_fig) # bar chart 시각화
    st.write("OverSampling Data Total Counts: ", len(sampling_df))
    st.dataframe(sampling_df[target].value_counts(), use_container_width=True)
    st.plotly_chart(sampling_fig) # pie chart 시각화
    
    st.write("OverSampling DataFrame", sampling_df) # dataframe 출력



def eda_null_info(df): # Null 값 확인
    # st.write("Null information")
    info_df = pd.DataFrame({'Column names': df.columns,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Dtype': df.dtypes,
                    })
    info_df.reset_index(inplace=True)
    # st.write(info_df.iloc[:, 1:].astype(str))
    st.dataframe(info_df.iloc[:, 1:].astype(str), use_container_width=True)

def eda_label_couts(target_label_counts): # 타겟 데이터 분포 확인
    val_counts_df = pd.DataFrame({'Labels': target_label_counts.iloc[:, 0],
                                'Counts': target_label_counts.iloc[:, 1]})
    val_counts_df.index = val_counts_df['Labels']

    st.dataframe(val_counts_df, use_container_width=True)
    st.bar_chart(val_counts_df['Counts'])

def prepro_test(updated_df, target_feature, label_to_drop): # drop한 데이터 제외한 데이터 출력 테스트
    updated_df = updated_df[updated_df[target_feature] != label_to_drop]
    return updated_df 