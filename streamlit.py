import streamlit as st
import numpy as np
import pandas as pd
import requests

# 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
API_KEY = st.secrets["API_KEY"]
API_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
headers = {"Authorization": f"Bearer {API_KEY}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


with st.form(key="my_form"):
    labels = st.text_area("Give set of labels seperated with comma", " ")
    inputs = st.text_area(
        "Enter text for classification",
        "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
    )
    payload = {
        "inputs": inputs,
        "parameters": {"candidate_labels": list(labels.split(","))},
    }
    submitted = st.form_submit_button("Classify")
    if submitted:
        output = query(payload)
        st.write(output)
