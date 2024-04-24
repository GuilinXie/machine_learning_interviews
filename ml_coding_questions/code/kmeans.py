# https://domino.ai/blog/getting-started-with-k-means-clustering-in-python
# https://www.youtube.com/watch?v=s_mgVRfLysY
# https://zhuanlan.zhihu.com/p/387621659

# KMeans steps:
# 1. Decide how many clusters you want, i.e. choose k
# 2. Randomly assign a centroid to each of the k clusters
# 3. Calculate the distance of all observation to each of the k centroids
# 4. Assign observations to the closest centroid
# 5. Find the new location of the centroid by taking the mean of all the observations in each cluster
# 6. Repeat steps 3 - 5 until the centroids do not change position


# Classifying questions:
# 1. Do I need to generate data?
# 2. What's the dimension of the data?


# Time complexity
# O(m * n * k * d)
# m: steps
# n: number of data points
# k: number of centroids
# d: dimensions

# Follow up:
# 1: Detect converge automatically, without hardcode the step number: use finished
# 2: How to choose K?
#     1. Required
#     2. Observe
#     3. Elbow method: Plot K over loss, x = K, y = loss, when the loss goes flat, then the K at this point would be a good one. Like an elbow-shape.
#     4. Gap statistic: Check out at https://zhuanlan.zhihu.com/p/387621659
# 3： How to choose K centroids in a better way?
#     1. kmeans++ : https://zhuanlan.zhihu.com/p/78798251



import numpy as np
from matplotlib import pyplot as plt

# Solution 1: Iterate


def data_generator():
    class_1_data = np.random.randn(100, 2) + np.array([3, 4])
    class_2_data = np.random.randn(100, 2) + np.array([10, -4])
    class_3_data = np.random.randn(100, 2) + np.array([-5, 0])
    return np.concatenate([class_1_data, class_2_data, class_3_data], axis = 0)

# Visualize the data
# data = data_generator()
# plt.scatter(data[:, 0], data[:, 1])
# plt.show()

def kmeans(data, K):
    D = data.shape[1]   # D = 2, number of dimension
    N = data.shape[0]   # N = 300, number of 
    category = np.zeros(N)   # The category of each data point
    centroid = np.random.randn(K, D)    # Stores the centroid of each category
    centroid_copy = centroid.copy()
    
    # To detect converge automatically
    finished = False
    
    while not finished:
    # for i in range(10):
        # Step-1: update the category for each data point
        for j in range(N):
            nearest_centroid = None
            nearest_centroid_distance = float("inf")
            
            for k in range(K):
                dist_j_k = np.linalg.norm(centroid[k] - data[j])
                if dist_j_k < nearest_centroid_distance:
                    nearest_centroid_distance = dist_j_k
                    nearest_centroid = k
            category[j] = nearest_centroid
            
            
        # Data Visualization
        plt.scatter(x=data[:, 0], y=data[:, 1], c=category)
        plt.plot(centroid[:, 0], centroid[:, 1], "r+")
        plt.show()

        # Step-2: update the centroid of each category based on current means
        
        # Learning rate, how fast to update centroid
        # Better for visualization, and better for stability. The centroids are moving slower.
        lr = 0.5    
        
        for j in range(K):
            new_centroid = np.mean(data[category == j], axis=0)   # Use a mask to filt data
            centroid[j] = (1 - lr) * centroid[j] + lr * new_centroid
        
        # new_centroid - old_centroid
        mean_update = np.linalg.norm(np.linalg.norm(centroid - centroid_copy, axis=0).reshape(-1))     # .reshape(-1) to flat the matrix to 1 row
        centroid_copy = centroid.copy()
        if mean_update < 0.001:
            finished = True

             
        # Step-3: repeat step-1 and step-2 until converging
    
    
# Solution 2: Recursive

def kMeans(points, center_list, k):
    # Store inter-mediate results
    clusters = [[] for _ in range(k)]
    
    for i in points:
        # Compute distance from observation to centroids
        distance = [np.sqrt(sum(np.square(np.array(i) - np.array(j)))) for j in center_list]
        # Assign the point to the closest centroid
        clusters[distance.index(min(distance))].append(i)
    # Calculate new centroids
    new_center_list = np.array([sum(np.array(i)) / len(i) for i in clusters])
    
    if not (new_center_list == center_list).all():
        clusters, center_list = kMeans(points, new_center_list, k)
    return clusters, center_list

def randCenter(dataset, k):
    temp = []
    while len(temp) < k:
        index = np.random.randint(0, len(dataset) - 1)
        if index not in temp:
            temp.append(index)
    return np.array([dataset[i] for i in temp])
    


# Elbow Method to find K
# Advantage: Can be used in analysis high-dimension data
from sklearn.cluster import KMeans

def elbow_analysis(data):
    sse = []
    potential_k = list(range(1, 10))
    
    for k in potential_k:
        km = KMeans(n_clusters=k)
        km.fit(data)
        sse.append(km.inertia_)
    
    plt.figure(figsize=(6, 6))
    plt.plot(potential_k, sse, '-o')
    plt.xlabel(r"Number of clusters *k*")
    plt.ylabel("Sum of squared distance")
    plt.show()

    
if __name__ == "__main__":
    data = data_generator()
    # kmeans(data, 3)
    elbow_analysis(data)
    
    
    
        

        
        

