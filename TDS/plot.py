# data visualizations
import seaborn as sns
import matplotlib.pyplot as plt


def plot_line(Y_pred, Y):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=Y_pred, y=Y, ax=ax)
    sns.lineplot(x=Y_pred, y=Y_pred, ax=ax, color='black')
    ax.set_xlabel("SalePrice")
    plt.show()


def corr_map(X):
    X_corr = X.corr()
    sns.heatmap(X_corr, annot=False, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.show()
    # print corr
    high_corr = X_corr[X_corr > 0.8]
    high_corr.stack()

