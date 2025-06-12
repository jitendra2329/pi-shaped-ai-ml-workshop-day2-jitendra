from sklearn.metrics import silhouette_score

def evaluate_clustering(X, labels):
    return silhouette_score(X, labels)
