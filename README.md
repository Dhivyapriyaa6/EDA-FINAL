# Flood Forecasting & Resilient Farming System for Disaster-Prone Indian Regions
A data-driven platform designed to predict floods, analyze rainfall trends, and support climate-resilient agriculture in disaster-prone regions of India.

## Project Goal
Our Goal is to build an integrated system that can:
1) Predict flood risk using long-term rainfall data
2) Support farmers with region-specific climate insights
3) Provide actionable dashboards for rainfall trends
4) Enable AI-based future rainfall forecasting
5) Assist policymakers with flood-vulnerability analytics

## System Overview
### Data Sources
Multi-dataset integration from Kaggle, including:
1) District-wise Rainfall Normals
2) Flood Risk Dataset (India)
3) Historical Rainfall (1901–2015)
4) Historical Crop Recommendation

### Data Processing Pipeline
1) Missing value imputation (mean/median)
2) Duplicate removal
3) Feature generation (season indicators, rainfall deviation, numeric encoding)
4) Merging rainfall & flood datasets at district level

### Visual Analytics
The system provides complete EDA including:
1) Flood Risk Heatmaps
2) Monthly & Seasonal Rainfall Patterns
3) Box Plots, Radar Charts
4) Donut Charts for seasonal distribution

These visualizations help identify region-specific trends.

## AI/ML Model
### LSTM-based Rainfall Forecasting System
### Model Highlights
1) Time-series forecasting architecture
2) Learns long-term rainfall patterns
3) Captures monsoon seasonality, anomalies & extreme events
4) Customizable forecast horizon (default: 12 months)

### Performance Indicators
1) RMSE: 15–45 mm
2) MAE: 12–35 mm
3) Directional Accuracy: 80–85%
4) Confidence Interval: ±15% shaded visualization

## Installation

git clone https://github.com/Dhivyapriyaa6/EDA-FINAL.git

cd flood-forecasting

python -m venv venv

source venv/bin/activate     # Windows: ./venv/Scripts/activate

pip install -r requirements.txt

streamlit run app.py

## Sample Output Screenshots
<img width="868" height="411" alt="image" src="https://github.com/user-attachments/assets/4c0783e2-07c4-46ad-bc25-c788edb94e52" />
<img width="940" height="439" alt="image" src="https://github.com/user-attachments/assets/277e3cbc-fa46-4158-9612-c1c733010d5f" />
