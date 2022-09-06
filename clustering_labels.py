from sklearn.cluster import KMeans
X = data.values

# 10 clusters
n_clusters = 20
# Runs in parallel All CPUs

# Train K-Means.
kmeans = KMeans(n_clusters = n_clusters, n_init = 20).fit(X)
kmeans.labels_
target = kmeans.labels_
pd.DataFrame(target)[0].value_counts()

# Data Scaling
rb.fit(data)
data = rb.transform(data)

data.shape, target.shape