import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

def original_tab(df, target):
    original_target_series = df[target]
    original_value_counts = original_target_series.value_counts()

    st.write("Original Data Total Counts: ", len(df))
    st.write("Original Data: ", original_value_counts) # original_value_counts
    st.write('Bar Chart Original Data')

    n_df = pd.DataFrame(original_value_counts.reset_index())
    bar_fig = px.bar(n_df, x=n_df.iloc[:, 0], y=n_df.iloc[:, 1])
    st.plotly_chart(bar_fig)
    fig = px.pie(n_df, values=n_df.iloc[:, 1], names=n_df.iloc[:, 0], title='Pie Chart Sampling Data')
    st.plotly_chart(fig)
    st.write("Original DataFrame: ", df) 

    # if len(df.columns) <= 2:
    #    n_df = pd.DataFrame(original_value_counts.reset_index())
    #    bar_fig = px.bar(n_df, x=n_df.iloc[:, 0], y=n_df.iloc[:, 1])
    #    st.plotly_chart(bar_fig)
    #    fig = px.pie(n_df, values=n_df.iloc[:, 1], names=n_df.iloc[:, 0], title='Pie Chart Sampling Data')
    #    st.plotly_chart(fig)
    #    st.write("Original DataFrame: ", df) 
    # else:
    #     n_df = pd.DataFrame(original_value_counts.reset_index())
    #     bar_fig = px.bar(n_df, x=n_df.iloc[:, 0], y=n_df.iloc[:, 1])
    #     st.plotly_chart(bar_fig)
    #     fig = px.pie(n_df, values=n_df.iloc[:, 1], names=n_df.iloc[:, 0], title='Pie Chart Sampling Data')
    #     st.plotly_chart(fig)
    #     st.write("Original DataFrame: ", df) 

def sampling_tab(sampling_df, target):
    updated_target_series = sampling_df[target]
    updated_value_counts = updated_target_series.value_counts()

    st.write("OverSampling Data Total Counts: ", len(sampling_df))
    st.write("OverSampling Data: ", sampling_df[target].value_counts())
    st.write('Bar Chart Sampling Data')
    n_df = pd.DataFrame(updated_value_counts.reset_index())
    bar_fig = px.bar(n_df)
    st.plotly_chart(bar_fig)
    sampling_fig = px.pie(n_df, values=n_df.iloc[:, 1], names=n_df.iloc[:, 0], title='Pie Chart Sampling Data')
    st.plotly_chart(sampling_fig)
    st.write("OverSampling DataFrame: ", sampling_df)

    # if len(sampling_df.columns) >= 2:
    #     n_df = pd.DataFrame(updated_value_counts.reset_index())
    #     bar_fig = px.bar(n_df)
    #     st.plotly_chart(bar_fig)
    #     sampling_fig = px.pie(n_df, values=n_df.iloc[:, 1], names=n_df.iloc[:, 0], title='Pie Chart Sampling Data')
    #     st.plotly_chart(sampling_fig)
    #     st.write("OverSampling DataFrame: ", sampling_df)
    # else:
    #     st.bar_chart(updated_value_counts)
    #     fig = px.pie(updated_value_counts, values=updated_value_counts.values, names=updated_value_counts.index, title='Pie Chart Original Data')
    #     st.plotly_chart(fig)
    #     st.write("OverSampling DataFrame: ", sampling_df)

def eda_null_info(df):
    # st.write("Null information")
    info_df = pd.DataFrame({'Column names': df.columns,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Dtype': df.dtypes,
                    })
    info_df.reset_index(inplace=True)
    st.write(info_df.iloc[:, 1:].astype(str))

def eda_label_couts(target_label_counts):
    val_counts_df = pd.DataFrame({'Labels': target_label_counts.iloc[:, 0],
                                'Counts': target_label_counts.iloc[:, 1]})
    st.write(val_counts_df)
    val_counts_df.index = val_counts_df['Labels']
    st.bar_chart(val_counts_df['Counts'])

def prepro_test(updated_df, target_feature, label_to_drop):
    updated_df = updated_df[updated_df[target_feature] != label_to_drop]
    return updated_df 