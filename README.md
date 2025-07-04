# Netflix_Stock_Prediction_Sytstem

# Netflix Stock Price Prediction & Forecasting 🎥📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red)

An end-to-end machine learning project that predicts Netflix (NFLX) stock prices using LSTM neural networks, with a Streamlit web interface for interactive forecasting.

## Features ✨

- **LSTM Model**: Deep learning model trained on historical Netflix stock data
- **Interactive Dashboard**: Visualize predictions with Streamlit
- **30-Day Forecast**: Generate multi-day stock price projections
- **Live Data Integration**: Option to fetch current market data (via Yahoo Finance)

## Project Structure 🗂️
Netflix-Stock-Prediction/
├── data/
│ ├── raw/ # Original dataset (CSV/zip)
│ └── processed/ # Cleaned and processed data
├── models/ # Saved model files (.h5, .pkl)
├── notebooks/ # Jupyter notebooks
│ ├── 1_data_preparation.ipynb
│ └── 2_model_training.ipynb
├── app/ # Streamlit application
│ ├── app.py # Main application
│ └── utils.py # Helper functions
└── requirements.txt # Python dependencies

## Installation ⚙️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/netflix-stock-prediction.git
cd netflix-stock-prediction

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```
Usage 🚀
Data Preparation
Run the Jupyter notebook in order:

notebooks/2_model_training.ipynb

Run Streamlit App
```cd app
streamlit run app.py
The app will launch at http://localhost:8501

```
Technical Details 🛠️
Model Architecture: 2-layer LSTM with Dropout

Training Data: Netflix stock data 2010-2023

Metrics: MSE (Mean Squared Error)

Frontend: Streamlit with Plotly charts

Contributing 🤝
Pull requests are welcome! For major changes, please open an issue first.

License 📄
MIT
```

### Key Features of This README:
1. **Badges** - Visual indicators for technologies used
2. **Clear Structure** - Organized sections with emojis
3. **Installation Guide** - Step-by-step setup instructions
4. **Visual Hierarchy** - Proper markdown formatting
5. **Technical Details** - Shows the ML stack clearly

### To Add:
1. Replace placeholder screenshot with actual app screenshot
2. Update the GitHub URL with your actual repository link
3. Add your dataset source/attribution if required
4. Include any special instructions for your specific setup

Would you like me to modify any particular section or add more details about specific components?
```
