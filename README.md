<div align="center"> 
<!-- <img src="docs/assets/logo.png" alt="Riesgo Fichajes Logo" width="200"/> --> 

# Riesgo Fichajes - Liga 1 ML 

**A Machine Learning pipeline and Streamlit application for predicting transfer risk in Peru's Liga 1.** 

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div> 

--- 

## 📖 Description 

**Riesgo Fichajes** is an academic Machine Learning project designed to augment human football scouting with quantitative risk signals. By analyzing historical pre-transfer technical and tactical performance data, the system predicts whether a new signing ("fichaje") in Peru's top football division (Liga 1) will be a high-risk or low-risk investment. 

The core predictive engine is a `scikit-learn` RandomForest classifier trained on normalized, per-90-minute statistical metrics. The project features a complete offline data scraping pipeline and an interactive online Streamlit application equipped with SHAP explainability and Groq-powered LLM narrative scouting reports. 

> **⚠️ Current Status:** The repository is currently undergoing a major structural refactoring sprint to resolve "Split-Brain" architecture (DRY violations), lock dependencies, and implement rigorous automated testing. 

## 🗂️ Project Structure 

The repository is logically divided into data extraction, offline model training, and the online web application. 

- **`scripts/`**: The 5-phase offline ML pipeline. 
  - *Phase 1:* Player ID collection via Transfermarkt. 
  - *Phase 2:* Raw statistics extraction via SofaScore. 
  - *Phase 3:* Feature engineering and normalization (per-90 metrics). 
  - *Phase 4:* Labeling logic (expanding-window z-score + inactivity rules). 
  - *Phase 5:* Model training, walk-forward cross-validation, and `.joblib` export. 
- **`web_streamlit/`**: The interactive Streamlit web application. Contains the UI routing (`app.py`), live prediction logic, and LLM integrations. 
- **`core/`** *(Upcoming)*: A unified library for shared scraping clients, feature engineering mathematics, and central constants to prevent Training-Serving Skew. 
- **`datos_entrada/` & `datos_salida/`**: Directories containing raw transfer lists, intermediate processed files, and the final training CSVs. 
- **`model_artifacts/`**: Serialized production models, hyperparameters, and evaluation plots. 

## 📊 Data Sources 

This project relies on a custom multi-stage web scraping pipeline. No static datasets are used. 
- [**Transfermarkt**](https://www.transfermarkt.com/): Scraped for seasonal transfer lists, player biographical data (age, nationality), and club history. 
- [**SofaScore**](https://www.sofascore.com/): Scraped via REST API to collect granular, season-long technical-tactical statistics for each player. 

## ⚙️ Installation & Setup 

Ensure you have Python 3.11+ installed. We recommend using a virtual environment. 

```bash 
# 1. Clone the repository 
git clone [https://github.com/axelstefans/riesgo-fichajes-liga1-ml.git](https://github.com/axelstefans/riesgo-fichajes-liga1-ml.git) 
cd riesgo-fichajes-liga1-ml 

# 2. Create and activate a virtual environment (Windows) 
python -m venv venv 
.\venv\Scripts\activate 

# For Linux/macOS: 
# python3 -m venv venv 
# source venv/bin/activate 

# 3. Install the required dependencies 
pip install -r requirements.txt