import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(y_test, y_pred, y_train, y_pred_train):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_train, y=y_pred_train)
    sns.lineplot(y_train, y_train, color='red')
    plt.title("Training Data")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_test, y=y_pred)
    sns.lineplot(y_test, y_test, color='red')
    plt.title("Test Data")

    plt.show()
