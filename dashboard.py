# dashboard.py
import streamlit as st
import pandas as pd
import time
import os

st.set_page_config(page_title="Federated Learning Dashboard", layout="wide")
st.title("Laryngeal FL - Training Dashboard")

@st.cache_data(ttl=5)
def load_data():
    if not os.path.exists("metrics.csv"):
        return pd.DataFrame(columns=["round", "client_id", "accuracy", "loss"])
    return pd.read_csv("metrics.csv")

placeholder = st.empty()
while True:
    df = load_data()
    if df.empty:
        st.warning("No data logged yet...")
        time.sleep(2)
        continue

    with placeholder.container():
        st.subheader("Training Metrics")
        st.line_chart(df.groupby("round")[["accuracy", "loss"]].mean())

        st.subheader("Per Client Accuracy")
        st.line_chart(df.pivot(index="round", columns="client_id", values="accuracy"))

        st.subheader("Latest Logs")
        st.dataframe(df.tail(10), use_container_width=True)

    time.sleep(5)
