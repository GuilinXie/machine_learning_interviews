# KMeans steps:
# 1. Decide how many clusters you want, i.e. choose k
# 2. Randomly assign a centroid to each of the k clusters
# 3. Calculate the distance of all observation to each of the k centroids
# 4. Assign observations to the closest centroid
# 5. Find the new location of the centroid by taking the mean of all the observations in each cluster
# 6. Repeat steps 3 - 5 until the centroids do not change position
# https://domino.ai/blog/getting-started-with-k-means-clustering-in-python

import numpy as np

# Solution 1: Recursive

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


        
        
    
        
        

