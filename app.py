import streamlit as st
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import pandas as pd
import os

# Set paths
BASE_DIR = "C:/Netflix-Stock-Prediction"
MODEL_PATH = os.path.join(BASE_DIR, "models/netflix_lstm_model.h5")
DATA_PATH = os.path.join(BASE_DIR, "data/processed/netflix_stock_processed.csv")

# Custom objects for model loading
custom_objects = {
    'mse': MeanSquaredError(),
    'MeanSquaredError': MeanSquaredError()
}

# App setup
st.set_page_config(layout="wide")
st.title("Netflix Stock Predictor")

# Add custom CSS for theme colors and enhanced styling
st.markdown(
    """
    <style>
    /* Change background color */
    .css-1d391kg {
        background-color: #141414;
    }
    /* Change header color */
    .css-1v3fvcr {
        color: #4CAF50;
        font-weight: bold;
    }
    /* Style buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        cursor: pointer;
    }
    /* Style sliders */
    div[role="slider"] {
        background-color: #4CAF50 !important;
    }
    /* Style date input */
    input[type="date"] {
        border: 1px solid #4CAF50;
        border-radius: 4px;
        padding: 5px;
    }
    /* Style markdown headers */
    h2 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, index_col='Date')

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects
        )
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Main app
df = load_data()
model = load_model()

if model:
    st.markdown("<h2 style='color:#4CAF50;'>Welcome to Netflix Stock Predictor!</h2>", unsafe_allow_html=True)
    
    # Date range selector for stock prices
    st.subheader("Stock Prices")
    min_date = pd.to_datetime(df.index.min())
    max_date = pd.to_datetime(df.index.max())
    start_date, end_date = st.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
    else:
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date
        filtered_df = df.loc[start_str:end_str]
        st.line_chart(filtered_df['Close'])
    
    # Prediction UI
    st.subheader("Make Prediction")
    
    # Get last 60 days data
    recent_data = df['Close'].tail(60).values.reshape(-1, 1)
    scaler = None
    try:
        import joblib
        scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Scaler loading failed: {e}")
    
    if scaler is not None:
        scaled_data = scaler.transform(recent_data)
        
        if st.button("Predict Tomorrow's Closing Price"):
            prediction = model.predict(scaled_data.reshape(1, 60, 1))
            predicted_price = scaler.inverse_transform(prediction)[0][0]
            
            last_price = df['Close'].iloc[-1]
            change = ((predicted_price - last_price) / last_price) * 100
            
            st.success(f"**Predicted Closing Price:** ${predicted_price:.2f}")
            if change >= 0:
                st.markdown(f"<h3 style='color:green;'>Change from Today: +{change:.2f}%</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:red;'>Change from Today: {change:.2f}%</h3>", unsafe_allow_html=True)
    else:
        st.warning("Scaler not loaded, cannot make predictions.")
    
    # 30-day forecast
    st.subheader("30-Day Forecast")
    n_days = st.slider("Select days to forecast", 1, 30, 7)
    
    if st.button("Generate Forecast"):
        current_sequence = scaled_data.copy()
        forecast = []
        
        for _ in range(n_days):
            next_pred = model.predict(current_sequence.reshape(1, 60, 1))
            forecast.append(next_pred[0, 0])
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        last_date = pd.to_datetime(df.index[-1])
        dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_days+1)]
        
        forecast_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in dates],
            'Forecasted Price': forecast_prices.flatten()
        }).set_index('Date')
        
        st.line_chart(forecast_df)
        st.dataframe(forecast_df.style.format({'Forecasted Price': '{:.2f}'}))
        
        # Download button
        csv = forecast_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name='netflix_stock_forecast.csv',
            mime='text/csv',
        )
    
    import io
    import sys
    st.write("Model architecture:")
    stream = io.StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__
    summary_str = stream.getvalue()
    st.text(summary_str)
else:
    st.warning("Could not load model")
