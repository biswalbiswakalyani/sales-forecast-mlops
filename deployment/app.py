import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Sales Forecast", layout="centered")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="BBiswal30/sales-forecast-model",
        filename="sales_forecast_model.pkl",
        repo_type="model"
    )
    return joblib.load(model_path)

model = load_model()

st.title("📈 Sales Forecasting App")
st.write("Predict sales using deployed ML model")

# -------- User Inputs --------
Product_Weight = st.number_input("Product Weight", 0.0, 50.0, 5.0)
Product_Sugar_Content = st.selectbox("Sugar Content", ["Low Sugar","Regular","No Sugar"])
Product_Allocated_Area = st.slider("Allocated Area Ratio", 0.0, 1.0, 0.05)
Product_Type = st.selectbox(
    "Product Type",
    ["Meat","Snack Foods","Hard Drinks","Dairy","Canned","Soft Drinks",
     "Health and Hygiene","Baking Goods","Bread","Breakfast","Frozen Foods",
     "Fruits and Vegetables","Household","Seafood","Starchy Foods","Others"]
)
Product_MRP = st.number_input("Product MRP", 1.0, 500.0, 50.0)

Store_Establishment_Year = st.number_input("Store Establishment Year", 1980, 2025, 2005)
Store_Size = st.selectbox("Store Size", ["Small","Medium","High"])
Store_Location_City_Type = st.selectbox("City Type", ["Tier 1","Tier 2","Tier 3"])
Store_Type = st.selectbox(
    "Store Type",
    ["Departmental Store","Supermarket Type1",
     "Supermarket Type2","Food Mart"]
)

# -------- Prediction --------
if st.button("Predict Sales"):
    input_df = pd.DataFrame([{
        "Product_Weight": Product_Weight,
        "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area,
        "Product_Type": Product_Type,
        "Product_MRP": Product_MRP,
        "Store_Establishment_Year": Store_Establishment_Year,
        "Store_Size": Store_Size,
        "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Forecasted Sales: ₹ {prediction:,.2f}")
