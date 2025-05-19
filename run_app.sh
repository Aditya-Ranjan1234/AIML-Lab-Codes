#!/bin/bash
echo "Installing required packages..."
pip install -r requirements.txt
echo "Starting Streamlit app..."
streamlit run app.py
