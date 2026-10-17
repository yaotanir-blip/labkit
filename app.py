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

    model,p0,equation,symbols = choose_model()

    with st.spinner("Fitting model"):
        fit = tls_fit(model,data,p0=p0,x=x,y=y,sigma_x=dx,sigma_y=dy)
    st.success("Fit complete")

    plot = plot_fit(model=model,data=data,x=x,y=y,dx=dx,dy = dy, params=fit.params, covariance=fit.covariance, x_fitted=fit.x_fitted, chi2=fit.chi2,uncertainties = fit.uncertainties)
    left,right = st.columns([2,1])
    with left:
        st.pyplot(plot)
    with right:
        st.subheader("Fit equation")
        st.latex(equation(*symbols))
        st.caption("With the fitted values:")
        st.latex(equation(*[latex_number(p) for p in fit.params]))
        st.caption(f"x = {x}, y = {y}")




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
    choice = st.selectbox("Model",["Linear","Exponential","Polynominal","custom"])
    if choice == "Linear":
        model,p0 = linear, [0,1]
        equation = lambda a, b: signs(rf"y = {a} + {b}\,x")
        symbols = ["a", "b"]
    elif choice == "Exponential":
        model, p0 = exponential, [1,0.01]
        equation = lambda A, k: rf"y = {A}\,e^{{{k}\,x}}"
        symbols = ["A", "k"]
    elif choice == "custom":
        model,p0 = custom, [150,1.6e5]
        equation = lambda E0, w0: (rf"V = \frac{{{E0}}}{{C\sqrt{{\omega^2 R^2 + L^2(\omega^2 - ({w0})^2)^2}}}}"
                                   r",\quad \omega = 2\pi f")
        symbols = ["E_0", r"\omega_0"]
    else:
        degree = st.number_input("Polynominal degree", min_value= 1,max_value = 10,value =2,step =1)
        model,p0 = polynominal,[0.0]* (int(degree)+1)
        equation = lambda *c: signs("y = " + " + ".join(
            [c[0], rf"{c[1]}\,x"] + [rf"{cj}\,x^{{{j}}}" for j, cj in enumerate(c) if j >= 2]))
        symbols = [f"c_{{{j}}}" for j in range(int(degree) + 1)]

    return model,p0,equation,symbols


def latex_number(value):
    """A fitted value as LaTeX, to 4 significant figures: 1.583 \\times 10^{5}."""
    mantissa, _, exponent = f"{value:.4g}".partition("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}" if exponent else mantissa


def signs(latex):
    """Tidy '+ -0.3' into '- 0.3' once negative fitted values are substituted in."""
    return latex.replace("+ -", "- ")



def linear(x,a,b):
    return a + b*x
def exponential(x,A,k):
    return A * np.exp(k*x)
def polynominal(x,*c):
    return sum(cj * x**j for j,cj in enumerate(c))
def custom(x, E0, omega0):
    w = 2*np.pi*x*1e3      
    C = 1.8e-9
    R = 500
    L = 22e-3
    return E0 / (C * np.sqrt(w**2 * R**2 + L**2 * (w**2 - omega0**2)**2))


main()
