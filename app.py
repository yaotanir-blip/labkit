import labkit
import numpy as np
import matplotlib.pyplot as plt

from labkit.data import fileread 
from labkit.fitting import tls_fit 
from labkit.results import plot_fit
from labkit.data import getheader

import streamlit as st 

def main():
    st.set_page_config(page_title="labkit",layout="wide")
    st.title("labkit")
    st.caption("Total least squares fitting for lab data ")

    data,header = choose_data()
    x = st.sidebar.selectbox("x", header, index=0)
    y = st.sidebar.selectbox("y", header, index=2)
    dx = st.sidebar.selectbox("dx", header, index=1)
    dy = st.sidebar.selectbox("dy", header, index=3)

    model,p0 = choose_model()

    with st.spinner("Fitting model"):
        fit = tls_fit(model,data,p0=p0,x=x,y=y,sigma_x=dx,sigma_y=dy)
    st.success("Fit complete")

    plot = plot_fit(model=model,data=data,x=x,y=y,dx=dx,dy = dy, params=fit.params, covariance=fit.covariance, x_fitted=fit.x_fitted, chi2=fit.chi2,uncertainties = fit.uncertainties)
    st.pyplot(plot)




def choose_data():
    st.sidebar.header("Data")
    source = st.sidebar.radio("Data source", ["Example data", "Upload CSV"])

    if source == "Example data":
        data = fileread(generate_data=True)
    else:
        uploaded = st.sidebar.file_uploader("Upload csv file", type="csv")
        if uploaded is None:
            st.info("Upload a CSV file to begin.")
            st.stop()        
        data = fileread(uploaded)

    return data, list(data.keys())



def choose_model():
    choice = st.selectbox("Model",["Linear","Exponential","Polynominal"])
    if choice == "Linear":
        model,p0 = linear, [0,1]
    elif choice == "Exponential":
        model, p0 = exponential, [1,0.01]
    else:
        degree = st.number_input("Polynominal degree", min_value= 1,max_value = 10,value =2,step =1)
        model,p0 = polynominal,[0.0]* (int(degree)+1)

    return model,p0

    

def linear(x,a,b):
    return a + b*x
def exponential(x,A,k):
    return A * np.exp(k*x)
def polynominal(x,*c):
    return sum(cj * x**j for j,cj in enumerate(c))


main()
