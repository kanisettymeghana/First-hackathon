import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("price_model.pkl", "rb"))

st.title("🧵 Textile Cost Prediction Dashboard (AI + ML)")

st.write("Enter the details below to predict the production cost per unit.")

product_type = st.selectbox("Product Type", ["Cotton", "Silk", "Polyester", "Wool"])
labor_hours = st.number_input("Labor Hours Used", min_value=0.0, step=0.1)
inventory_level = st.number_input("Inventory Level", min_value=0.0, step=1.0)
lead_time = st.number_input("Lead Time (Days)", min_value=0.0, step=1.0)
defect_rate = st.number_input("Defect Rate (%)", min_value=0.0, step=0.1)
market_demand = st.number_input("Market Demand Index", min_value=0.0, step=1.0)
supply_efficiency = st.number_input("Supply Chain Efficiency Score", min_value=0.0, step=0.1)

input_data = pd.DataFrame({
    'labor_hours_used': [labor_hours],
    'inventory_level': [inventory_level],
    'lead_time_days': [lead_time],
    'defect_rate_percentage': [defect_rate],
    'market_demand_index': [market_demand],
    'supply_chain_efficiency_score': [supply_efficiency],
    f'product_type_{product_type}': [1]
})

for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model.feature_names_in_]

if st.button("Predict Cost"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Production Cost per Unit: ₹{prediction[0]:,.2f}")

st.header("💡 Find Required Supply Chain Efficiency for Target Profit")

selling_price = st.number_input("Enter Selling Price per Unit (₹)", min_value=0.0, step=10.0)
desired_profit = st.number_input("Enter Desired Profit per Unit (₹)", min_value=0.0, step=10.0)

if st.button("Find Required Efficiency Score"):
    
    target_cost = selling_price - desired_profit
    
    
    efficiency_range = np.linspace(0, 100, 200)  
    best_efficiency = None
    best_diff = float('inf')
    
    for eff in efficiency_range:
        test_input = pd.DataFrame({
            'labor_hours_used': [labor_hours],
            'inventory_level': [inventory_level],
            'lead_time_days': [lead_time],
            'defect_rate_percentage': [defect_rate],
            'market_demand_index': [market_demand],
            'supply_chain_efficiency_score': [eff],
            f'product_type_{product_type}': [1]
        })
        for col in model.feature_names_in_:
            if col not in test_input.columns:
                test_input[col] = 0
        test_input = test_input[model.feature_names_in_]
        predicted_cost = model.predict(test_input)[0]
        diff = abs(predicted_cost - target_cost)
        if diff < best_diff:
            best_diff = diff
            best_efficiency = eff
    
    if best_efficiency is not None:
        st.success(f"✅ To achieve ₹{desired_profit:.2f} profit per unit, "
                   f"you need a supply chain efficiency score of **{best_efficiency:.2f}**.")
    else:
        st.error("Couldn't find a suitable efficiency score.")
