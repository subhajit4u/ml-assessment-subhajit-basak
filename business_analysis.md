# Part B: Business Case Analysis

## B1. Problem Formulation

### (a) Problem Definition

This is a supervised machine learning regression problem.

- **Target Variable:** items_sold  
- **Input Features:** transaction_date, store_id, store_size, location_type, promotion_type, is_weekend, is_festival, competition_density  
- **Problem Type:** Regression  

The goal is to predict the number of items sold based on store attributes, time-based factors, and promotional strategies.

---

### (b) Why items_sold instead of revenue

Using **items_sold** is more reliable than revenue because revenue can be influenced by price variations, discounts, and inflation.

Items sold reflects **true customer demand** and is not distorted by pricing strategies.

This highlights a key principle in ML:
> The target variable should directly represent the business objective and avoid external noise.

---

### (c) Alternative Modeling Strategy

Instead of a single global model, a better approach is:

- Build **store-specific models**, OR  
- Use a **hierarchical / segmented model** (e.g., by location_type)

This is because different stores respond differently to promotions due to demographics, location, and customer behavior.

---

## B2. Data and EDA Strategy

### (a) Data Joining Strategy

The data comes from:
- Transactions table  
- Store attributes  
- Promotion details  
- Calendar data  

These would be joined using:
- `store_id`
- `transaction_date`

**Final dataset grain:**  
One row per **store per day**

**Aggregations:**
- Total items_sold per day
- Average basket size
- Promotion applied per day

---

### (b) EDA Strategy

1. **Sales Distribution Plot**
   - To check skewness and outliers
   - Helps decide transformations

2. **Sales vs Promotion Type**
   - To understand effectiveness of different promotions

3. **Time Series Plot (Sales over time)**
   - To identify seasonality and trends

4. **Correlation Heatmap**
   - To identify relationships between numerical features

These insights help in feature engineering and model selection.

---

### (c) Handling Promotion Imbalance

Since 80% of transactions have no promotion:

- The model may become biased toward non-promotion cases
- It may underestimate the impact of promotions

**Solutions:**
- Use stratified analysis
- Apply weighting to promotion data
- Evaluate model separately on promotion vs non-promotion

---

## B3. Model Evaluation and Deployment

### (a) Train-Test Split and Metrics

- Use **time-based split** (train on past, test on future)
- Random split is inappropriate because it causes data leakage

**Metrics:**
- RMSE → measures large errors
- MAE → measures average error

**Interpretation:**
- Lower RMSE/MAE = better prediction accuracy

---

### (b) Explaining Different Recommendations

Feature importance helps explain model decisions:

- In December → festivals → promotions more effective  
- In March → lower demand → discounts preferred  

Thus, recommendations vary due to:
- Seasonality  
- Customer behavior  
- External factors  

---

### (c) Deployment Strategy

1. Save model using:
   - `joblib` or `pickle`

2. Monthly process:
   - Load new data
   - Apply same preprocessing
   - Generate predictions

3. Monitoring:
   - Track RMSE/MAE over time
   - Detect performance drop

4. Retraining:
   - Retrain model when performance degrades or new patterns emerge
