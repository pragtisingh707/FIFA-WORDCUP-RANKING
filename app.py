import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="FIFA Ranking Prediction",
    page_icon="⚽",
    layout="centered"
)

# Custom CSS
st.markdown(
"""
<style>

/* Background */
.stApp {
    background-color: #EAF6FF;
}

/* Main Heading RED */
h1 {
    color: red;
    text-align: center;
    font-weight: bold;
}

/* Sub Heading */
h2, h3 {
    color: #5D4037;
}

/* Input Box */
div[data-testid="stNumberInput"] input {
    background-color:white;
    color:#333333;
    border:2px solid #90CAF9;
    border-radius:10px;
}

/* Button */
.stButton > button {
    width:100%;
    background-color:#8B4513;
    color:white;
    font-size:18px;
    font-weight:bold;
    height:45px;
    border-radius:12px;
}

/* Button Hover */
.stButton > button:hover {
    background-color:#5D4037;
    color:white;
}

/* Result Box */
.result {
    background-color:white;
    padding:25px;
    margin-top:20px;
    border-radius:15px;
    text-align:center;
    font-size:25px;
    font-weight:bold;
    color:#8B4513;
    border:3px solid #90CAF9;
}
</style>
""",
unsafe_allow_html=True
)

# Load Model
with open("fifa_linear_model.pkl","rb") as file:
    model = pickle.load(file)

# Title
st.title("⚽ FIFA Ranking Points Prediction")
st.write("Machine Learning Model - Linear Regression")

st.subheader("Enter Team Performance Details")

# Inputs
previous_points = st.number_input("Previous Points", min_value=0.0)
cur_year_avg = st.number_input("Current Year Average", min_value=0.0)
last_year_avg = st.number_input("Last Year Average", min_value=0.0)
two_year_avg = st.number_input("Two Year Average", min_value=0.0)
three_year_avg = st.number_input("Three Year Average", min_value=0.0)
confederation = st.number_input("Confederation Code", min_value=0)

# Prediction
if st.button("Predict FIFA Points"):

    input_data = np.array([
        previous_points,
        cur_year_avg,
        last_year_avg,
        two_year_avg,
        three_year_avg,
        confederation
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]

    # Result Box
    st.markdown(
    f"""
    <div class="result">
    Predicted FIFA Total Points ⚽
    <br><br>
    {prediction:.2f}
    </div>
    """,
    unsafe_allow_html=True
    )

    # 📊 GRAPH SECTION
    st.subheader("📊 Input vs Predicted Visualization")

    chart_data = pd.DataFrame({
        "Features": [
            "Prev Points",
            "Cur Year Avg",
            "Last Year Avg",
            "2 Year Avg",
            "3 Year Avg",
            "Confed Code",
            "Predicted"
        ],
        "Values": [
            previous_points,
            cur_year_avg,
            last_year_avg,
            two_year_avg,
            three_year_avg,
            confederation,
            prediction
        ]
    })

    st.bar_chart(chart_data.set_index("Features"))

# Footer
st.markdown("---")
st.caption("FIFA Ranking Analysis Project | Data Analytics + Machine Learning")