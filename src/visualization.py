import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_elbow(wcss, max_k=15):
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()

def plot_clusters_pca(X, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='viridis')
    plt.title("Clusters Visualized with PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def cluster_profile(X_df, labels):
    df = X_df.copy()
    df["Cluster"] = labels
    return df.groupby("Cluster").mean()
