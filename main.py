import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import generators as gen
import plotting
import re

DISTRIBUTIONS = ['Normal', 'Geometric', 'Uniform', 'Bernoulli', 'Weibull', 'Gamma']
warning = ''

st.header('DistSeek: A library to guess the input data distribution')

st.sidebar.title('Choose a starting point')

method = st.sidebar.radio('', ['Input data', 'Generate data'])
if method == 'Input data':
    selected_dist = 'UNKNOWN'
    st.sidebar.title('Choose data input method')
    input_options = st.sidebar.radio('', ['Upload values (*.csv)', 'Paste values'])
    if input_options == 'Paste values':
        raw = st.sidebar.text_area('Paste your data').split('\n')
        data = [float(i) for i in raw if re.match("[+-]?(?:\d+(?:\.\d+)?|\.\d+)$", i)]
        data = pd.DataFrame(data, columns=['data'])

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
            data2 = data * 3
            data2.rename({'data': 'data1'})

            if data.shape[1] > 1:
                st.warning(f"Only first column will be used, uploaded data shape is {data.shape}")
                data = data.iloc[:, 0]
else:
    input_options = "User generated data"
    selected_dist = st.sidebar.selectbox('Select a distribution (default values provided)', DISTRIBUTIONS)
    n = (st.sidebar.text_input('No. of Samples', value=100))

    if selected_dist == 'Normal':
        mu = st.sidebar.text_input('Mean', value=np.float(10))
        sigma = st.sidebar.text_input('Standard Deviation', value=np.float(2))
        data = gen.Generator(np.int(n)).normal(np.float(mu), np.float(sigma))

    elif selected_dist == 'Geometric':
        p = np.float(st.sidebar.text_input('P', value=.7))
        if p > 1:
            warning = 'P value should be <= 1'
        else:
            data = gen.Generator(np.int(n)).geometric(np.float(p))

    elif selected_dist == 'Uniform':
        a = st.sidebar.text_input('a', value=np.int(0))
        b = st.sidebar.text_input('b', value=np.int(1))
        data = gen.Generator(np.int(n)).uniform(np.float(a), np.float(b))

    elif selected_dist == 'Bernoulli':
        p = np.float(st.sidebar.text_input('P', value=.6))
        if p > 1:
            warning = 'P value should be <= 1'
        else:
            data = gen.Generator(np.int(n)).bern(np.float(p))

    elif selected_dist == 'Weibull':
        lambda_hat = st.sidebar.text_input('Lambda', value=.161)
        r_hat = st.sidebar.text_input('R', value=.525)
        data = gen.Generator(np.int(n)).weibull(np.float(r_hat), np.float(lambda_hat))

    elif selected_dist == 'Gamma':
        lambda_hat = st.sidebar.text_input('Lambda', value=.161)
        r_hat = st.sidebar.text_input('R', value=.525)
        data = gen.Generator(np.int(n)).gamma(np.float(r_hat), np.float(lambda_hat))
#

# is data ready
try:
    data_ready = len(data) > 2

except:
    data_ready = False

if data_ready:
    data = pd.DataFrame(data, columns=['data'])
    data2 = pd.DataFrame(data * 3)
    data2.rename({'data': 'data1'})
    sns.set(style="darkgrid")
    fig = sns.kdeplot(data['data'], shade=True, color='r',
                      label='Method: ' + method+"_"+selected_dist)
    fig = sns.kdeplot(data2['data'], shade=True, color='b', label='Estimated')
    mean = np.mean(data)
    plt.axvline(x=data.mean()[0], color='red')
    plt.axvline(x=data2.mean()[0], color='blue')
    plt.legend()
    st.pyplot(fig.figure)
else:
    st.warning('Data is not ready yet')
x = {'Parameter'
     : ['Method'],
     'Value': [method + ': ' + input_options]}
# x = {'Method':method+': '+input_options, 'Method1':mean*2}
st.table(pd.DataFrame(x))
