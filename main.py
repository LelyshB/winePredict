from data_loader import load_data
from model import train_and_evaluate
from data_visualization import prepare_data
import pandas as pd
from evaluation import plot_results

def main():
    # Шаг 1: Загрузка данных
    df = pd.read_csv("winequality-red.csv", delimiter=";")

    # Шаг 2: Подготовка данных
    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']
    target = 'quality'
    X, y = prepare_data(df, features, target)

    # Шаг 3: Обучение и оценка модели
    y_pred_train, y_pred_test, y_train, y_test = train_and_evaluate(X, y)

if __name__ == "__main__":
    main()




