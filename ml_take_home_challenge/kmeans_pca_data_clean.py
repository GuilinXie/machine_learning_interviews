import numpy as np       
import pandas as pd       
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# %matplotlib inline
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

#################################
# Step1: Read .csv file
data = './data/live.csv'
df = pd.read_csv(data)

#################################
# Step2: Clean Data
df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)   # Drop all null columns
## Filter duplicates
duplicated_data = df[df['status_id'].duplicated() == True]                # Check duplicated_data
df = df.drop_duplicates(subset='status_id', keep='last')                       

df.drop(['status_id', 'status_published'], axis=1, inplace=True)     # Drop categorical features seem like identifier

# Generate more features
df['all_reaction_count'] = df.iloc[:, -6:].sum(axis=1)
df['reaction_match'] = df.apply(lambda x: x['all_reaction_count'] == x['num_reactions'], axis=1)
df['reaction_comment_ratio'] = df.num_reactions / df.num_comments
df['reaction_share_r'] = df.num_reactions / df.num_shares

print(df.isnull().sum())
print(df[df.reaction_comment_ratio.isnull()].head())

## Replace not usable values
df.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)

print(df.isnull().sum())

#################################
# Step3: Define features X and target y
X = df.drop('status_type', axis=1, inplace=False)
y = df['status_type']

#################################
# Step3: Transform X and y

# numerical_columns = list(df)
numerical_columns = list(X)
numerical_columns.remove("reaction_match")

categorical_columns = ['reaction_match']

preprocessor = ColumnTransformer([
# ('onehot', OneHotEncoder(sparse=False), ['major']),
('onehot', OneHotEncoder(sparse=False), categorical_columns),
# ('ordinal', OrdinalEncoder(), ['status_type']),
# ('discretizer', KBinsDiscretizer(n_bins=3), ['age']),
# ('scale', StandardScaler(), ['age']),
('scale', MinMaxScaler(), numerical_columns),
# ('pass', 'pass_through', ['column1']),
# ('drop', 'drop', ['status_type']),
],
remainder='passthrough')


# # Step2: Define features X and target y
# X = df[df.columns[2:]]
# y = df[['status_type']]


X_prep = preprocessor.fit_transform(X)


le = LabelEncoder()
y = le.fit_transform(y)

# feature_names = preprocessor.get_feature_names_out()    # OrdinalEncoder does not support get_feature_names_out yet
# columns_new = list(df)
columns_new = preprocessor.get_feature_names_out()

X_new = pd.DataFrame(X_prep, columns=columns_new)

# Apply PCA to reduce dimension
pca = PCA()
pca_result = pca.fit_transform(X_new)

# To Check for PCA's k
# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel("Principle Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.show()


num_components = 2
pca_result_selected = pca_result[:, :num_components]

pca_df = pd.DataFrame(data=pca_result_selected, columns=[f'PC{i}' for i in range(1, num_components + 1)])

# kmeans = KMeans(n_clusters=4, random_state=0)
# kmeans.fit(X_new)
# print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
# labels = kmeans.labels_
# correct = sum(y == labels)
# print('Accuracy Score: {0:0.2f}'.format(correct / float(y.size)))

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(pca_df)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
pca_df['Cluster'] = kmeans.labels_
correct = sum(y == pca_df['Cluster'])
print('Accuracy Score: {0:0.2f}'.format(correct / float(y.size)))


pca_df.plot.scatter(x='PC1', y='PC2', c='Cluster', colormap='viridis', figsize=(16, 5), title="PCA Components with K-means Clustering")
# plt.scatter(x='PC1', y='PC2', c=list(pca_df['Cluster'].map(color_map)
# plt.figure(figsize=(16, 6))
# sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
# plt.title("PCA Components with kmeans clustering")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='cluster')
plt.show()


# # Elbow Method to find optimal K

# cs = []
# for k in range(1, 11):
#     kms = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kms.fit(X_new)
#     cs.append(kms.inertia_)

# plt.plot(range(1, 11), cs)
# plt.title("The Elbow Method")
# plt.xlabel("Number of clusters *k*")
# plt.ylabel("Sum of squared distance")
# plt.show()


