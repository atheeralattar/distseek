import streamlit as st
import numpy as np
import pandas as pd

DISTRIBUTIONS = ['Normal', 'Geometric', 'Uniform', 'Binomial', 'Weibull', 'Gamma']
st.header('DistSeek: A library to guess the input data distribution')

st.sidebar.title('Choose a starting point')

method = st.sidebar.radio('', ['Input data', 'Generate data'])
if method == 'Input data':
    st.sidebar.title('Choose data input method')
    input_options = st.sidebar.radio('', ['Upload values (*.csv)', 'Paste values'])
    if input_options == 'Paste values':
        raw = st.sidebar.text_area('Paste your data').split('\n')
        data = pd.DataFrame(raw, columns=['data'])
    else:
        header = st.sidebar.selectbox('Data start at row:', [0, 1])
        if header == 0:
            header = None
        else:
            header = 0
            names = ['data']
        raw = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=False)
        if raw is not None:
            data = pd.read_csv(raw, header=header, names=['data'])
            if data.shape[1] > 1:
                st.warning(f"Only first column will be used, uploaded data shape is {data.shape}")
                data = data.iloc[:, 0]
else:
    selected_dist = st.sidebar.selectbox('Select a distribution (default values provided)', DISTRIBUTIONS)
    n = st.sidebar.text_input('No. of Samples')
    if selected_dist == 'Normal':
        mu = st.sidebar.text_input('Mean', value=np.random.randint(10, 100))
        sigma = st.sidebar.text_input('Standard Deviation', value=np.random.randint(1, 10))
    elif selected_dist == 'Geometric':
        p = st.sidebar.text_input('P', value=np.round(np.random.rand(), 2))
    elif selected_dist == 'Uniform':
        a = st.sidebar.text_input('a', value=np.random.randint(0, 10))
        b = st.sidebar.text_input('b', value=np.random.randint(11, 20))
    elif selected_dist == 'Binomial':
        p = st.sidebar.text_input('P', value=np.round(np.random.rand(), 2))
    elif selected_dist == 'Weibull':
        lambda_hat = st.sidebar.text_input('Lambda', value=.161)
        r_hat = st.sidebar.text_input('R', value=.525)
    elif selected_dist == 'Gamma':
        lambda_hat = st.sidebar.text_input('Lambda', value=.161)
        r_hat = st.sidebar.text_input('R', value=.525)

st.write(data)
st.write(data.shape)
