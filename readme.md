# 📊 GDP per Capita vs Happiness Index - Linear Regression

> Understanding the relationship between economic prosperity and human well-being through univariate linear regression

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue?style=flat-square&logo=python)
![ML](https://img.shields.io/badge/MachineLearning-Linear%20Regression-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 Overview

This project implements **univariate linear regression from scratch** to explore the relationship between **GDP per capita** and **life satisfaction (happiness index)** across different countries. Rather than using scikit-learn, the regression parameters are calculated manually using mathematical formulas to develop a deep understanding of how linear regression works under the hood.

### 🎯 Key Objective

> Prove the correlation between economic wealth and human happiness by implementing the foundational mathematics of linear regression from first principles.

---

## 🤔 Understanding the Problem

### The Research Question
*"Does a country's GDP per capita influence the happiness level of its citizens?"*

### Dataset Overview

- **Target Variable (Y)**: Life Satisfaction (Cantril Ladder Score: 0-10)
  - Based on survey responses from the World Happiness Report
  - Measures self-reported well-being across countries
  
- **Feature Variable (X)**: GDP per Capita (in international dollars, 2021 prices)
  - Adjusted for inflation and purchasing power parity
  - Represents average economic output per person

- **Data Source**: Our World in Data, World Bank, OECD, IMF
- **Time Period**: 2011-2025
- **Coverage**: Multiple countries with representative samples

---

## 🧮 Mathematical Foundation

### Simple Linear Regression Model

The goal is to find the best-fit line through the data points:

$$\hat{y} = b_0 + b_1 \cdot x$$

Where:
- $\hat{y}$ = predicted life satisfaction
- $x$ = GDP per capita
- $b_1$ = slope (coefficient)
- $b_0$ = intercept (y-intercept)

### Calculating the Slope

The slope quantifies how much life satisfaction changes for each unit increase in GDP per capita:

$$b_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

**Step-by-step breakdown:**
1. Calculate the mean of X: $\bar{x} = \frac{\sum x_i}{n}$
2. Calculate the mean of Y: $\bar{y} = \frac{\sum y_i}{n}$
3. Calculate deviations from mean for each point
4. Multiply deviations together for numerator: $(x_i - \bar{x})(y_i - \bar{y})$
5. Square deviations for denominator: $(x_i - \bar{x})^2$
6. Divide sum of numerator by sum of denominator

### Calculating the Intercept

Once the slope is determined, find where the line crosses the y-axis:

$$b_0 = \bar{y} - b_1 \cdot \bar{x}$$

This ensures the regression line passes through the point $(\bar{x}, \bar{y})$

### Cost Function (Mean Squared Error)

To evaluate how well our regression line fits the data, we use the **Mean Squared Error (MSE)** cost function:

$$J(b_0, b_1) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

Where:
- $m$ = number of data points
- $\hat{y}_i$ = predicted value for point $i$
- $y_i$ = actual value for point $i$
- The difference $(\hat{y}_i - y_i)$ is called the **residual** or prediction error

**What it measures:**
- The average squared distance between predicted and actual values
- Lower cost = better fit (predictions closer to actual data)
- Squaring the errors penalizes large mistakes more heavily
- The factor of $\frac{1}{2m}$ normalizes the cost across different dataset sizes

This cost function quantifies how well the regression line explains the relationship between GDP and happiness.

### Making Predictions

For any given GDP value, predict the life satisfaction:

$$\hat{y}_{\text{new}} = b_0 + b_1 \cdot x_{\text{new}}$$

---

## 📁 Project Structure

```
MachineLearning/
│
├── housing_prices.py                 # Main regression analysis script
│   ├── Loads GDP vs Happiness data
│   ├── Calculates slope and intercept
│   ├── Evaluates model cost (MSE)
│   ├── Generates predictions
│   ├── Visualizes results with scatter + regression line
│   └── Makes predictions for new countries
│
├── linear_regression_parameters.py    # Helper functions module
│   ├── mean()                        # Calculates arithmetic mean
│   ├── mean_difference()             # Computes deviations from mean
│   ├── multiply_mean_differences_and_sum()  # Numerator calculation
│   └── mean_diff_square()            # Denominator calculation
│
├── cost_function.py                  # Cost evaluation module
│   └── calculate_cost()              # Computes Mean Squared Error (MSE)
│
├── gdp-vs-happiness.csv              # Dataset with countries' data
│   ├── Entity (Country name)
│   ├── GDP per capita
│   └── Life satisfaction
│
├── gdp-vs-happiness.metadata.json     # Data documentation & sources
│   ├── Data collection methodology
│   ├── Column descriptions
│   ├── Data sources and citations
│   └── Processing notes
│
└── README.md                          # This file

```

### File Descriptions

#### `linear_regression_parameters.py` (Utility Functions)
Contains modular helper functions for mathematical operations:
- `mean(arr)`: Computes the average of an array
- `mean_difference(arr, mean)`: Returns array of deviations from the mean
- `muliply_mean_differences_and_sum(arrX, arrY)`: Calculates sum of products of paired deviations
- `mean_diff_square(arr)`: Computes sum of squared values

#### `cost_function.py` (Cost Evaluation)
Implements the Mean Squared Error (MSE) cost function:
- `calculate_cost(x_train, y_train, b, w)`: Computes how well the regression line fits the data
  - Takes predicted parameters (intercept `b`, slope `w`) and actual data
  - Returns the average squared error between predictions and actual values
  - Lower cost indicates a better-fitting regression model

**Why a separate cost function?**
- **Model Evaluation**: Quantifies prediction accuracy
- **Optimization**: Can be minimized to find best parameters (basis for gradient descent)
- **Comparison**: Allows comparing different models objectively
- **Reusability**: Can be used with different regression approaches

#### `housing_prices.py` (Main Script)
The primary execution file that:
- Loads the GDP vs Happiness dataset using pandas
- Implements the linear regression pipeline
- Calculates the best-fit line parameters
- Evaluates the cost function to measure model quality
- Generates visualizations using matplotlib
- Makes predictions on new data (e.g., Cyprus)

#### `gdp-vs-happiness.csv`
Real-world dataset containing:
- Country/region names
- GDP per capita (in international $)
- Life satisfaction scores
- Multiple years of data for temporal analysis

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib
```

### Execution
```bash
python housing_prices.py
```

### Expected Output

1. **Cost Function**: Displays the Mean Squared Error for the fitted model
   ```
   Cost function evaluates to 15102121326.38
   ```

2. **Regression Equation**: Shows the best-fit line formula
   ```
   Best fit line equation is: y(hat) = 4.7 + 0.00003163 x_i
   ```

3. **Visualization**: Shows a scatter plot of actual values with the fitted regression line

4. **Prediction**: Makes a prediction for a new country
   ```
   Predict the happiness index of country 'Cyprus' having a GDP per capita of 37655
   Happiness Index is 5.873208109174731
   ```

---

## 💡 Key Insights

### What the Model Reveals

1. **Positive Correlation**: There is a clear positive relationship between GDP and happiness
   - As GDP per capita increases, life satisfaction tends to increase
   
2. **Non-Linear Pattern**: The relationship isn't perfectly linear
   - Very wealthy nations show diminishing returns in happiness gains
   - Poorer nations see larger happiness increases with GDP growth

3. **Predictive Power**: The model can predict happiness levels for countries based on their economic output

### Limitations

- **Correlation ≠ Causation**: A strong GDP-happiness relationship doesn't prove money causes happiness
- **Oversimplification**: Many factors influence happiness (healthcare, education, freedom, relationships, etc.)
- **Univariate Model**: Uses only one feature; more sophisticated models (multivariate regression) would be more accurate
- **Time Lag**: Economic changes may take time to affect reported well-being
- **Cultural Differences**: Happiness reporting varies across cultures and value systems

---

## 🎓 Learning Outcomes

By implementing linear regression from scratch, you'll understand:

✅ **Core Concepts**
- How regression finds patterns in data
- The meaning of slope and intercept in real-world context
- Why we minimize squared errors

✅ **Mathematical Skills**
- Computing mean and deviations
- Understanding covariance and variance
- Deriving the normal equation for regression parameters

✅ **Programming Practices**
- Modular function design
- Data loading and preprocessing with pandas
- Visualization best practices with matplotlib
- NumPy for numerical computations

✅ **ML Fundamentals**
- The difference between actual vs predicted values
- Training and inference phases
- How to make predictions on new data

---

## 🔄 Algorithm Walk-Through

Here's what happens when you run the script:

```
1. Load CSV Data
   └─ Read GDP per capita and Life satisfaction columns

2. Calculate Slope (b1)
   ├─ Find mean of X (GDP values)
   ├─ Find mean of Y (Happiness values)
   ├─ Compute deviations: (X - X̄) and (Y - Ȳ)
   ├─ Calculate numerator: Σ(X - X̄)(Y - Ȳ)
   ├─ Calculate denominator: Σ(X - X̄)²
   └─ b1 = numerator / denominator

3. Calculate Intercept (b0)
   ├─ Already have b1 and means
   └─ b0 = Ȳ - b1 * X̄

4. Generate Predictions
   ├─ For each training point: ŷᵢ = b0 + b1 * xᵢ
   └─ For new countries: ŷ = b0 + b1 * x_new

5. Evaluate Cost Function
   ├─ Calculate residuals: (ŷᵢ - yᵢ) for each point
   ├─ Square each residual
   ├─ Sum all squared residuals
   └─ MSE = (1/2m) * Σ(ŷᵢ - yᵢ)²

6. Visualize Results
   ├─ Plot actual values as red X markers
   ├─ Plot predicted values as blue line
   └─ Add labels and legend

7. Display Results
   └─ Print equation, cost, and specific predictions
```

---

## 📊 Data Sources & Attribution

- **Life Satisfaction Data**: Wellbeing Research Centre (2026) – World Happiness Report
- **GDP Data**: Eurostat, OECD, IMF, and World Bank (2026) – World Development Indicators
- **Data Platform**: [Our World in Data](https://ourworldindata.org/grapher/gdp-vs-happiness)

**Citation**:
- Wellbeing Research Centre (2026). "Self-reported life satisfaction." World Happiness Report 2026.
- World Bank (2026). "GDP per Capita – World Bank – In constant international-$." World Development Indicators.

---

## 🤝 Next Steps & Extensions

To enhance this project:

1. **Add R² Score**: Measure goodness of fit
   - Formula: $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

2. **Multivariate Regression**: Add more features (education, health, etc.)

3. **Statistical Tests**: Perform t-tests on the slope coefficient

4. **Cross-Validation**: Split data into train/test sets

5. **Residual Analysis**: Check if model assumptions are violated

6. **Time Series**: Analyze how the relationship changes year-over-year

7. **Regional Analysis**: Build separate models for different world regions

---

## 📚 Resources to Learn More

- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSA)
- [StatQuest with Josh Starmer - Linear Regression](https://www.youtube.com/watch?v=PwZucgF2-nE)
- [A Complete Guide to Linear Regression in Python](https://realpython.com/linear-regression-in-python-with-scikit-learn/)
- [Our World in Data - Happiness and Life Satisfaction](https://ourworldindata.org/happiness-and-life-satisfaction)

---

<div align="center">

**Made with ❤️ for understanding Machine Learning fundamentals**

*"The best way to understand machine learning is to implement it yourself."*

</div>
