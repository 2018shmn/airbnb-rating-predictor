# Airbnb Rating Predictor

## Overview

Machine learning project that predicts Airbnb accommodation review scores (0-5, continuous scale) using listing features to provide insights into customer satisfaction drivers and listing optimization strategies.

## Problem Definition

**Objective:** Predict review_scores_rating using property and host characteristics

**Impact:** Understanding ratings enable:
- Quality assessment and improvement recommendations
- Listing optimization strategies
- Revenue optimization through promoting high-potential listings

## Dataset

28,000+ Airbnb listings with features including:

- **Review Metrics:** Overall rating, cleanliness, location, communication
- **Host Information:** Response rates, verification, listing count
- **Property Details:** Room type, availability, pricing

### Key Data Characteristics: 
Left-skewed target distribution (most ratings >4.0), systematic missing values, strong correlation between review subcategories.

## Methodology

### 1. Exploratory Data Analysis:
Visualized target variable distribution, found missing data patterns, and identified top predictive features through correlation matrix, correlation heatmaps, and scatter plots

### 2. Data Preprocessing/Feature Engineering:
- Median imputation for numerical features
- Boolean to numerical conversion
- Removal of high-missing rate text columns and uninformative features
- One-hot encoding for categorical variables with ≤10 categories

### 3. Model Development & Evaluation:

#### Models: 
Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor

#### Hyperparameter Optimization:
Grid Search with 5-fold Cross-Validation for ensemble methods

#### Metrics: 
Mean Absolute Error (MAE), Root Mean Square Error (RMSE), R² Score

## Results

### Model Performance:
Ensemble methods (Gradient Boosting and XGBoost) significantly outperformed baseline models. Gradient Boosting model achieved the highest R2 Score of 0.81 and improved prediction performance by 81% over baseline (mean predictor), outperforming Linear Regression by ~10% in variance explained.

### Key Predictors: 
Review subcategories (strongest), price, availability, host verification, number of listings

## Tech Stack
```python
pandas, numpy, scipy          # Data processing
scikit-learn, xgboost         # Machine learning
matplotlib, seaborn           # Visualization
```

## Applications
- **Host Guidance:** Actionable insights for rating improvement
- **Quality Control:** Identify underperforming listings
- **Market Analysis:** Rating patterns across different property types
- **Platform Optimization:** Data-driven decision making

## Future Enhancements
- Text analysis of reviews and descriptions
- Geographic and temporal modeling
- Real-time Prediction API integration

## Project Structure

```
airbnb-rating-predictor/
├── airbnbListingsData.zip/airbnbListingsData.csv
├── Airbnb_Predictor.ipynb
├── README.md
```
## Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
scipy>=1.7.0
```
---
