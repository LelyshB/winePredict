from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Оценка модели
    print("Test Data:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_test)}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_test)}")
    print(f"R^2 Score: {r2_score(y_test, y_pred_test)}")

    print("Training Data:")
    print(f"Mean Squared Error: {mean_squared_error(y_train, y_pred_train)}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_pred_train)}")
    print(f"R^2 Score: {r2_score(y_train, y_pred_train)}")

    # Графики для визуализации
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, color='blue')
    plt.plot([min(y_train), max(y_train)], [min(y_pred_train), max(y_pred_train)], color='red') # линия регрессии
    plt.xlabel('True Values [Quality]')
    plt.ylabel('Predictions [Quality]')
    plt.title('Prediction vs True (Training Data)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, color='green')
    plt.plot([min(y_test), max(y_test)], [min(y_pred_test), max(y_pred_test)], color='red') # линия регрессии
    plt.xlabel('True Values [Quality]')
    plt.ylabel('Predictions [Quality]')
    plt.title('Prediction vs True (Test Data)')


    # График для визуализации тренировочных и тестовых данных
    plt.figure(figsize=(10, 6))

    plt.scatter(y_train, y_pred_train - y_train, color='blue', label='Training Data')
    plt.scatter(y_test, y_pred_test - y_test, color='green', label='Test Data')
    plt.hlines(y=0, xmin=min(y), xmax=max(y), colors='red', linestyles='dashed')
    plt.xlabel('True Values [Quality]')
    plt.ylabel('Residuals')
    plt.legend()
    plt.title('Residual Plot')

    plt.show()

    return y_pred_train, y_pred_test, y_train, y_test

