# House-Price-Prediction-Model
# House Price Prediction using Deep Learning

This repository contains the code for the **"House Prices - Advanced Regression Techniques"** competition on **Kaggle**. The objective of this competition is to predict house prices based on various features such as lot size, number of rooms, and neighborhood details. Below is a detailed description of the code and its workflow.

---

## Code Overview

The code is written in **Python** using **Jupyter Notebook** and leverages popular data science libraries such as **pandas, numpy, seaborn, and matplotlib** for data processing and visualization. The workflow includes **data preprocessing, feature engineering, model training, and prediction generation**.

---

## Steps in the Code

### 1. Importing Libraries
The necessary libraries are imported:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```
- **Pandas** for data handling.
- **NumPy** for numerical computations.
- **Matplotlib & Seaborn** for data visualization.

---

### 2. Loading Data
The training and test datasets are loaded into DataFrames:
```python
df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")
```
- `df1` contains **training data** (features + target variable `SalePrice`).
- `df2` contains **test data** (features only, without `SalePrice`).

---

### 3. Data Preprocessing
Since the test dataset lacks the target variable (`SalePrice`), it is added with a default value of `0` to facilitate concatenation:
```python
if 'SalePrice' not in df2.columns:
    df2['SalePrice'] = 0
df = pd.concat([df1, df2], axis=0)
```
- Merges **training and test datasets** to apply consistent preprocessing.

---

### 4. Handling Missing Values
Missing values are handled separately for numerical and categorical columns:
```python
# Filling missing numerical values with column mean
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Filling missing categorical values with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
```
- **Numerical columns**: Filled with the column mean.
- **Categorical columns**: Filled with the most frequent value (mode).

---

### 5. Feature Engineering
New features are created to enhance the model's performance:
```python
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath'])
```
- **TotalSF**: Sum of basement, first, and second floor areas.
- **TotalBath**: Total number of bathrooms, combining full and half baths.

---

### 6. Encoding Categorical Variables
Categorical features are transformed into numerical values using **Label Encoding**:
```python
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label.fit_transform(df[col])
```
- **Encodes categorical features** for model compatibility.

---

### 7. Splitting Data
The combined dataset is split back into **training and test sets**:
```python
train, test = df[:df1.shape[0]], df[df1.shape[0]:]
test = test.drop(columns='SalePrice')
```
- **Train set**: Contains the `SalePrice` target variable.
- **Test set**: Excludes `SalePrice` for prediction.

---

### 8. Model Training
A **Deep Learning model** (e.g., a simple neural network) can be trained:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_train = train.drop(columns=['SalePrice'])
y_train = train['SalePrice']

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```
- Uses a **simple feedforward neural network** with **ReLU activation**.
- Compiles with **Adam optimizer** and **Mean Squared Error (MSE) loss function**.

---

### 9. Generating Predictions
The trained model generates predictions on the test dataset:
```python
y_pred = model.predict(test)
final = pd.DataFrame()
df_dummy = pd.read_csv("test.csv")
final['Id'] = df_dummy['Id']
final['SalePrice'] = y_pred.flatten()
final.to_csv('submission.csv', index=False)
```
- **Predictions are saved** in a `submission.csv` file for Kaggle submission.

---

## Key Features

 **Feature Engineering**: New features like `TotalSF` and `TotalBath` improve model performance.
 **Handling Missing Values**: Uses mean for numerical and mode for categorical data.
 **Deep Learning Model**: Implements a neural network using TensorFlow/Keras.
 **Submission Ready**: Generates predictions in a **Kaggle-compatible format**.

---

## How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```
2. **Ensure the dataset files (`train.csv` and `test.csv`) are placed in the correct directory**.
3. **Run the Jupyter Notebook** to preprocess the data, train the model, and generate predictions.
4. **The predictions will be saved in `submission.csv`**.

---

## Dependencies

Ensure you have the following installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
```

---

## Future Improvements

🔹 **Hyperparameter tuning** to optimize the deep learning model.
🔹 **Experimenting with other models** like XGBoost or CatBoost.
🔹 **Additional feature engineering** to improve predictive accuracy.



## Contributions

Feel free to fork the repository and contribute improvements! 🚀

