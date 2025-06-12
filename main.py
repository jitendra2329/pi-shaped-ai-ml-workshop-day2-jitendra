from src import data_loader, preprocess, clustering, evaluation, visualization

if __name__ == "__main__":
    # Load data
    df = data_loader.load_data("data/Wholesale customers data.csv")

    print("Initial Data Preview:")
    print(df.head())

    # Preprocess
    X = preprocess.select_features(df)
    X_scaled = preprocess.scale_features(X)

    # Elbow method
    wcss = clustering.find_optimal_k(X_scaled)
    visualization.plot_elbow(wcss)

    # Choose k (e.g., from elbow graph)
    optimal_k = 5
    model, labels = clustering.train_kmeans(X_scaled, optimal_k)

    # Evaluation
    score = evaluation.evaluate_clustering(X_scaled, labels)
    print(f"Silhouette Score: {score:.2f}")

    # Visualizations
    visualization.plot_clusters_pca(X_scaled, labels)

    # Cluster insights
    cluster_means = visualization.cluster_profile(X, labels)
    print("\nCluster Profiles:")
    print(cluster_means)
