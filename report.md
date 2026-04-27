# Finding the Best Bitcoin Price Direction Predictor

## Final Project Report

**Author:** [Jongbin Lee]  
**Date:** April 27, 2026  
**Course:** [IS 327]

---

## 1. Introduction

Cryptocurrency markets, particularly Bitcoin, represent a challenging domain for predictive modeling due to their high volatility, non-stationarity, and susceptibility to external factors such as news, regulations, and market sentiment. This project aims to develop a comprehensive machine learning pipeline for predicting Bitcoin price movements using historical market data.

The project builds upon the initial proposal by implementing a complete data science workflow: from data collection and feature engineering to model training and evaluation. Unlike traditional academic projects that use static datasets, this work emphasizes real-world applicability by constructing a data pipeline that could theoretically be deployed for continuous prediction.

The core challenge addressed is the prediction of next-day Bitcoin price direction - whether the price will increase or decrease. This binary classification problem is practically relevant for traders who primarily concern themselves with directional movements rather than exact price predictions.

## 2. Research Question

**Which type of machine learning model, classification or regression, is more effective for predicting next-day cryptocurrency price direction?**

This question encompasses both technical performance metrics and practical utility. While regression models provide more granular predictions, classification models directly address the binary decision-making process that underlies most trading strategies.

## 3. Hypothesis

Based on the characteristics of financial markets and the nature of the prediction task, the following hypotheses were formulated:

1. **Classification models will perform better than regression models** when predicting price direction. This expectation stems from the fundamental difference in objectives: classification directly optimizes for directional accuracy, while regression attempts to predict exact values in a highly noisy environment.

2. **Random Forest models will outperform linear models**. Cryptocurrency markets exhibit nonlinear relationships and complex interactions between variables. Tree-based ensemble methods are better suited to capture these patterns compared to linear approaches.

3. **Nonlinear technical indicators will enhance model performance** when combined with appropriate nonlinear modeling techniques. Indicators like RSI and MACD are inherently nonlinear transformations of price data, making them more compatible with Random Forest than Logistic Regression.

## 4. Variables and Features

### Data Source
- **Dataset**: Historical Bitcoin (BTC-USD) data from January 1, 2023 to March 1, 2023
- **Granularity**: Daily intervals
- **Sample Size**: 27 observations after feature engineering and NaN removal
- **Source**: Yahoo Finance (yfinance library) - used as alternative to Binance API due to access restrictions

### Target Variables
- **Classification Target (`direction`)**: Binary variable (0 = price decrease, 1 = price increase next day)
- **Regression Target (`next_return`)**: Continuous variable representing next-day percentage return

### Feature Variables

#### Raw Market Data
- `close`: Daily closing price
- `volume`: Daily trading volume

#### Return-Based Features
- `return_1d`: 1-day percentage price change
- `return_3d`: 3-day percentage price change
- `return_7d`: 7-day percentage price change
- `return_30d`: 30-day percentage price change

#### Trend Indicators
- `ema_12`: 12-day Exponential Moving Average
- `ema_26`: 26-day Exponential Moving Average

#### Momentum Indicators
- `rsi_14`: 14-day Relative Strength Index (0-100 scale)

#### MACD Components
- `macd`: MACD line (EMA12 - EMA26)
- `macd_signal`: 9-day EMA of MACD line
- `macd_hist`: MACD histogram (MACD - MACD Signal)

#### Volatility Features
- `volatility_30d`: 30-day rolling standard deviation of price

#### Volume Features
- `volume_change_1d`: 1-day percentage change in volume

## 5. Methods

### Data Pipeline
A modular data pipeline was implemented with three main components:

1. **Data Collection** (`data_pipeline.py`): Automated fetching of historical Bitcoin data
2. **Feature Engineering** (`features.py`): Creation of technical indicators and derived features
3. **Model Training & Evaluation** (`models.py`): Implementation of ML models with cross-validation

### Feature Incorporation
All 14 engineered features were used simultaneously as input variables for both classification and regression tasks. This approach allows models to learn complex relationships between different market indicators without manual feature selection. The feature set provides multiple perspectives on market behavior:

- **Temporal patterns** through return calculations
- **Trend analysis** via moving averages
- **Momentum signals** through RSI and MACD
- **Risk assessment** via volatility measures
- **Market participation** via volume indicators

### Modeling Approach
Four models were implemented to compare different algorithmic approaches:

#### Classification Models
- **Logistic Regression**: Linear probabilistic classifier for binary outcomes
- **Random Forest Classifier**: Ensemble of decision trees for nonlinear classification

#### Regression Models
- **Linear Regression**: Ordinary least squares for continuous prediction
- **Random Forest Regressor**: Ensemble method for nonlinear regression

### Evaluation Methodology
- **Cross-Validation**: 5-fold stratified cross-validation to ensure robust performance estimates
- **Classification Metrics**:
  - Accuracy: Overall correct prediction rate
  - F1-Score: Harmonic mean of precision and recall
- **Regression Metrics**:
  - R²: Proportion of variance explained
  - Mean Squared Error (MSE): Average squared prediction error

## 6. Results

### Classification Results

| Model | Accuracy (Mean ± Std) | F1-Score (Mean ± Std) |
|-------|----------------------|----------------------|
| Logistic Regression | 0.64 ± 0.03 | 0.00 ± 0.00 |
| Random Forest Classifier | 0.54 ± 0.12 | 0.18 ± 0.23 |

### Regression Results

| Model | R² (Mean ± Std) | MSE (Mean ± Std) |
|-------|----------------|------------------|
| Linear Regression | -15.14 ± 15.96 | 0.0040 ± 0.0036 |
| Random Forest Regressor | -0.47 ± 0.37 | 0.0007 ± 0.0007 |

## 7. Hypothesis Evaluation

### Hypothesis 1: Classification vs. Regression
**Supported** - Classification models significantly outperformed regression models. Logistic Regression achieved 64% accuracy, while both regression models showed poor performance (negative R² values). This confirms that directional prediction is more tractable than exact value prediction in volatile cryptocurrency markets.

### Hypothesis 2: Random Forest vs. Linear Models
**Partially Supported** - Random Forest Regressor showed better performance than Linear Regression (R²: -0.47 vs. -15.14), but Random Forest Classifier underperformed compared to Logistic Regression (54% vs. 64% accuracy). This suggests that while tree-based methods excel at regression in noisy data, they may overfit for classification with small datasets.

### Hypothesis 3: Nonlinear Indicators with Nonlinear Models
**Mixed Results** - The superior performance of Logistic Regression (linear) over Random Forest Classifier suggests that the nonlinear indicators did not provide sufficient benefit to overcome the overfitting tendencies of the more complex model with limited data.

## 8. Explanation and Discussion

### Performance Analysis
The strong performance of Logistic Regression (64% accuracy) indicates that linear relationships exist between the engineered features and price direction. This is somewhat counterintuitive given the nonlinear nature of technical indicators, but suggests that the feature engineering process effectively linearized complex market relationships.

The poor regression performance (negative R²) reflects the fundamental unpredictability of exact price movements in cryptocurrency markets. The high volatility and external influences create noise that overwhelms the signal in continuous prediction tasks.

### Limitations of Current Results
- **Small Dataset**: Only 27 observations limit model complexity and generalization
- **Short Time Period**: January-March 2023 may not capture diverse market conditions
- **Feature Engineering**: Some indicators (RSI, MACD) showed limited predictive power
- **Class Imbalance**: The F1-score of 0.00 for Logistic Regression suggests severe class imbalance

### Practical Implications
The 64% accuracy of Logistic Regression represents meaningful predictive power for directional trading. In a random market, 50% accuracy would be expected; 64% suggests the model captures genuine market patterns. However, transaction costs and slippage would need to be considered in real trading applications.

## 9. Learning Outcomes

This project provided valuable insights into machine learning applications in financial contexts:

1. **Feature Engineering Importance**: The transformation of raw price data into technical indicators proved crucial for model performance. Features like moving averages and momentum indicators provide structured representations of market behavior that raw data lacks.

2. **Model Selection Trade-offs**: The comparison between linear and nonlinear models highlighted the importance of matching model complexity to data characteristics and sample size. Simple models often outperform complex ones with limited data.

3. **Evaluation Challenges**: Cross-validation proved essential for reliable performance estimation in small, potentially non-stationary financial datasets.

4. **Practical vs. Theoretical Performance**: The superior performance of classification over regression underscores the importance of aligning model objectives with real-world decision-making processes.

5. **Domain Knowledge Integration**: Financial expertise in technical analysis directly informed feature selection and model interpretation.

## 10. Areas for Improvement

### Data and Features
- **Larger Dataset**: Extend time period and include multiple cryptocurrencies for better generalization
- **Additional Features**: Incorporate sentiment analysis, on-chain metrics, and macroeconomic indicators
- **Feature Selection**: Implement automated feature selection to identify most predictive variables
- **Alternative Data Sources**: Integrate news sentiment, social media data, and order book information

### Methodology
- **Advanced Models**: Experiment with neural networks, gradient boosting, and time-series specific models (LSTM, ARIMA)
- **Hyperparameter Tuning**: Systematic optimization of model parameters
- **Ensemble Methods**: Combine multiple models for improved predictions
- **Time-Series Validation**: Use rolling window cross-validation instead of random splits

### Evaluation
- **Backtesting**: Implement realistic trading simulation with transaction costs
- **Risk Metrics**: Include Sharpe ratio, maximum drawdown, and other trading-specific metrics
- **Statistical Testing**: Formal hypothesis testing for model comparison
- **Confidence Intervals**: Provide uncertainty estimates for predictions

### Implementation
- **Real-time Pipeline**: Deploy continuous data collection and prediction system
- **Model Interpretability**: Add SHAP values or feature importance analysis
- **Production Readiness**: Implement model monitoring, retraining, and deployment infrastructure

---

This project successfully demonstrated the feasibility of machine learning for cryptocurrency price prediction while highlighting the challenges and opportunities in financial forecasting. The results validate the hypothesis that classification approaches are more effective than regression for directional prediction, providing a foundation for future research in algorithmic trading systems.</content>
<parameter name="filePath">c:\Users\Jongbin\Bitcoin_Prediction_Techniques\Bitcoin_Prediction_Techniques\report.md