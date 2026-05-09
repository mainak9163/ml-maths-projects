import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, expon, poisson

st.set_page_config(page_title="CLT & LLN Simulator", layout="wide")
st.title("CLT & Law of Large Numbers Simulator")
st.markdown("**CSEL 863: Introduction to Data Science** — Sections 2–4")

# Sidebar controls
st.sidebar.header("Distribution Settings")
dist = st.sidebar.selectbox(
    "Choose Distribution",
    ["Uniform (Continuous)", "Exponential (Continuous)", "Poisson (Discrete)"],
)

n_sim = st.sidebar.slider("Number of Simulations (sample means)", 100, 5000, 1000, 100)
n = st.sidebar.select_slider("Sample Size n per simulation", options=[5, 30, 100, 1000])

if dist == "Uniform (Continuous)":
    a, b = st.sidebar.slider("Uniform(a, b)", 0.0, 10.0, (0.0, 1.0))
    mean_theory = (a + b) / 2
    var_theory = (b - a) ** 2 / 12
    pdf_func = lambda x: uniform.pdf(x, loc=a, scale=b - a)
    sample_func = lambda: np.random.uniform(a, b, n)
    st.sidebar.latex(r"f(x) = \frac{1}{b-a} \quad (a \leq x \leq b)")
    st.sidebar.latex(rf"\mu = \frac{{a+b}}{2} = {mean_theory:.3f}")

elif dist == "Exponential (Continuous)":
    beta = st.sidebar.slider("Rate β (β > 0)", 0.1, 5.0, 1.0)
    mean_theory = 1 / beta
    var_theory = 1 / (beta**2)
    pdf_func = lambda x: expon.pdf(x, scale=1 / beta)
    sample_func = lambda: np.random.exponential(scale=1 / beta, size=n)
    st.sidebar.latex(r"f(x|\beta) = \beta e^{-\beta x} \quad (x > 0)")
    st.sidebar.latex(f"\\mu = \\frac{{1}}{{\\beta}} = {mean_theory:.3f}")

else:  # Poisson
    lam = st.sidebar.slider("λ (rate)", 1, 20, 5)
    mean_theory = lam
    var_theory = lam
    pmf_func = lambda k: poisson.pmf(k, lam)
    sample_func = lambda: np.random.poisson(lam, n)
    st.sidebar.latex(
        r"f(x|\lambda) = \frac{e^{-\lambda}\lambda^x}{x!} \quad x=0,1,2,\dots"
    )
    st.sidebar.latex(rf"\mu = \lambda = {mean_theory}")

# Generate data
np.random.seed(42)
sample_means = []
for _ in range(n_sim):
    sample = sample_func()
    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Theoretical Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    if dist.startswith("Uniform"):
        x = np.linspace(a - 0.5, b + 0.5, 200)
        ax.plot(x, pdf_func(x), "r-", lw=2, label="PDF")
        ax.set_title("Uniform PDF")
    elif dist.startswith("Exponential"):
        x = np.linspace(0, mean_theory * 5, 200)
        ax.plot(x, pdf_func(x), "r-", lw=2, label="PDF")
        ax.set_title("Exponential PDF")
    else:
        k = np.arange(0, int(lam * 3) + 1)
        ax.stem(k, pmf_func(k), linefmt="r-", markerfmt="ro", basefmt=" ")
        ax.set_title("Poisson PMF")
    ax.axvline(
        mean_theory,
        color="blue",
        linestyle="--",
        label=f"Theoretical mean = {mean_theory:.3f}",
    )
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader(f"Sampling Distribution of Means (n = {n}, {n_sim} simulations)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        sample_means,
        bins=30,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax.axvline(
        mean_theory,
        color="red",
        linestyle="--",
        lw=2,
        label=f"True μ = {mean_theory:.3f}",
    )
    ax.set_title("Histogram of Sample Means")
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

st.success(
    f"✅ Law of Large Numbers: Sample means converge to theoretical mean **{mean_theory:.3f}**"
)
st.info(
    "As n increases → histogram of means becomes more bell-shaped → Central Limit Theorem in action!"
)

st.caption(
    "Built for CSEL 863 • Uses formulas directly from your lecture notes (Sections 2.3, 3.4 & Uniform example)"
)
