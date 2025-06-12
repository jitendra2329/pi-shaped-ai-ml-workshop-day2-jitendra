from sklearn.cluster import KMeans


def find_optimal_k(X, max_k=15):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

def train_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return model, labels
