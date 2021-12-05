import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
sns.set_theme()
def plotter(data):
    fig = sns.displot(data, kde=True, height=5, aspect=1.5)
    plt.axvline(x=data.mean(), color='red')
    st.pyplot(fig)