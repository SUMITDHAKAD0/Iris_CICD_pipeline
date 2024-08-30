import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_distributions(X_train):
    os.makedirs('artifacts/plots', exist_ok=True)
    plt.figure(figsize=(10, 6))
    X_train.hist(bins=30, figsize=(10, 8), layout=(2, 2))
    plt.tight_layout()
    plt.savefig('artifacts/plots/feature_distributions.png')
    plt.close()


def plot_model_performance(y_test, predictions):
    os.makedirs('artifacts/plots', exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Model Performance')
    plt.savefig('artifacts/plots/model_performance.png')
    plt.close()
