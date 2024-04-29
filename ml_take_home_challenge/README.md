# EDA | Exploratory Data Analysis

## Numerical | Categorical | Label

```
data = ‘./data/file.csv’
df = pd.read_csv(data)

pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", None)
```

1. ### Check the number of samples and features

   ```
   df.shape
   ```
2. ### Preview the dataset

   ```
   df.head())
   ```
3. ### View summary of the dataset

   ```
   df.info()    # eg: 'object' -> character feature, 'int64' -> numerical feature
   ```
4. ### Drop unwanted columns

   ```
   df.drop(['column1', 'column2'], axis=1, inplace=True)
   ```
5. ### Remove duplicates for unique identifiers | a certain column

   1. ```
      duplicated_df = df[df['status_id'].duplicated() == True]

      duplicated_df.head()
      duplicated_df.tail()

      df[df.status_id == '245666232']   # Check for a specific id

      df_no_dps = df.drop_duplicates(subset='status_id', keep='last')
      df_no_dps.shape
      ```
6. ### Check for missing values in dataset

   ```
   df.isnull().sum()

   df[df.col_name.isnull()]     # Filter out np.NaN
   ```
7. ### Replace wrong value

   1. ```
      df.replace([np.inf, -np.inf, np.NaN], 0.0, inplace=True)    # replace the whole df

      df.column1.replace([np.Nan], 0.0, inplace=True)             # replace just column1
      ```
8. ### Numerical features

   1. View the statistical summary
      1. ```
         df.describe()	# count, min, max, mean, std, 25%, 50%, 75% percentile
         ```
   2. View a column's min/max
      1. ```
         df1.col1.min()
         df1.col1.max()
         ```
   3. Create a new column column1 = sum(column2, column3, column4)
      1. ```
         df['all_reaction_count'] = df.iloc[:, -6:].sum(axis=1)      # the sum of last 6 columns, get a new column as 'all_reaction_count'



         # Check the reason behind mismatch
         df_react_mismatch['diff_react'] = df_react_mismatch.num_reaction - df_react_mismatch
         ```
   4. Create a new column using / divide
      1. ```
         df['ratio'] = df.num_reactions / df.num_comments
         ```
   5. Create a new column using lambda
      1. ```
         df['reaction_match'] = df.apply(lambda x: x['num_reactions'] == x['all_reaction_count'], axis=1)
         ```
   6. Filter rows according to True/False column to create a new df
      1. ```
         df_react_mismath = df[df.reaction_match == False]
         ```
   7. Filter rows with a df's column based on another df column's values
      1. ```
         df1[df1['status_id'].isin(list(df2.status_id.values))]
         ```
   8. Filter rows with one column  between two conditions
      1. ```
         df1 = df[(df.ratio > 0) & (df.ratio <= 2)]

         df1 = df[(df['col1_name'] != 'link') & (df['col2_name'] != 'statUs')]
         ```
   9. Filter columns by column names
      1. ```
         df1 = df[['col1', 'col2', 'col3']]       # By column names

         df1 = df[df.columns[:-4]]                # By filtering out the last 4 columns. Drop the last 4 columns
         ```
   10. Set one column's value to another column's value
       1. ```
          df.num_reactions = df.all_reaction_count
          ```
   11. Draw plot line
       1. ```
          df.ratio.plot(kind='line', figsize=(16, 5))
          ```
   12. Draw plot scatter for two columns
       1. ```
          df.plot.scatter(x='col1_name', y='col2_name', figsize=(16,5), title='col1_name vs. col2_name')
          ```
   13. Draw plot scatter for three columns (the third column is the color)
       1. ```
          import seaborn as sns

          plt.figure(figsize=(16, 6))
          sns.scatterplot(x='col1_name', y='col2_name', hue='col3_name', data=df, palette='viridis')   # hue is the color column
          plt.title("title_name")
          plt.xlabel('x_name')
          plt.ylabel('y_name')
          plt.legend(title='legend_name')   # To show which color represents which
          plt.show()
          ```
   14. Draw plot scatter for three columns(the third column is the color)
       1. ```
          df.plot.scatter(x='col1_name', y='col2_name', c='col3_name', colormap='viridis')   # The difference from sns.scatterplot is that this legend works for contiguous value of the third column.
          ```
   15. TODO
9. ### Categorical features

   1. Explore one by one
      1. ```
         df['status_id'].unique()        # Check how many unique characters
         len(df['status_id'].unique())    # Check if it is unique identifier or usable features

         df.nunique()        # Check all columns' uniques
         ```
   2. Draw bar chart for value_counts()
      1. ```
         st_ax = df.status_type.value_counts().plot(kind='bar', figsize=(10, 5), title='Status Type')
         st_ax.set(xlabel='Status_type', ylabel='Count')
         ```
   3. TODO
   4. Convert categorical feature into Integers
      1. ```
         from sklearn.preprocessing import OneHotEncoder
         from sklearn.compose import ColumnTransformer

         ct = ColumnTransformer([('onehot', OneHotEncoder(sparse=False), ['gender', 'major'])])
         ct.fit_transform(df)

         ```
10. ### Numerical + Categorical features Preprocessing Pipeline

    ```
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder,StandardScaler, MinMaxScaler, KBinsDiscretizer
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(sparse=False), ['major']),
    ('ordinal', OrdinalEncoder(), ['gender']),
    ('discretizer', KBinsDiscretizer(n_bins=3), ['age']),
    ('scale', StandardScaler(), ['age']),
    ('pass', 'pass_through', ['column1']),
    ('drop', 'drop', ['column2']),
    ],
    remainder='passthrough'
    )

    preprocessor.fit_transform(df)
    ```
11. ### Preprocessor Save & Load in Deployment

    1. ```
       import pickle

       # Save
       with open(file_path, 'wb') as f:
       	pickle.dump(ct, f)

       # Load
       with open(file_path, 'rb') as f:
       	ct = pickle.load(f)
       ```
12. ### Encode label y

    1. ```
       y = df['status_type']
       le = LabelEncoder()
       y = le.fit_transform(y)
       ```
13. ### Save clean data

    1. ```
       clean_data_df.to_csv('./data/clean_data.csv', index=False)
       ```
14. 
15. fdfa

## Text

## Image

## Time Series

# Feature Selection

## Tree based

```
from sklearn.ensemble import ExtraTreesClassifier

arr_new = df.values       # Pandas to array
Y_ch = df[:, 1:2]         # Get target column
X_ch = df[:, 3:12]        # Get features' column

model_sl = ExtraTreesClassifier()
model_sl.fit(X_ch, Y_ch)     # Train the model
print(model_sl.feature_importance_)    # Get the feature importance
```

# Data Split

```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncode()
le.fit(Y_ch.tolist())
Y_ch = le.transform(Y_ch.tolist())

X_train, x_test, y_train, y_test = train_test_split(X_ch[:, 3:8], Y_ch, test_size=0.25, random_state=0)
```

# Model Training & KFold Crossvalidation

```
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import KFold, cross_val_score

# Initialize models
models = {}
models["LDA"] = LinearDiscriminantAnalysis()
models['NB'] = MultinomialNB()
models['KNN'] = KNeighborsClassifier()

# Train models
results = []
for key in models:
	kfold = KFold(10)    # 10 fold
	cv_result = cross_val_score(models[key], x_train, y_train, scoring = 'neg_mean_squared_error', cv=kfold)
	results.append(cv_result)
	print('%s:%f(%f)'%(key, cv_result.mean(), cv_result.std()))		# 输出均方误差的均值（离0越近越好）、标准方差M o de z z

```

# Model optimization

## KNN

```
from sklearn.model_selection import GridSearchCV

param_grid_knn = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8]}   
model_knn_ty = KNeighborsClassifier()  
kfold = KFold(10)
grid = GridSearchCV(
		stimator=model_knn_ty,
		param_grid=param_grid_knn,
		scoring='neg_mean_squared_error',
		cv=kfold)
grid_result = grid.fit(x_train, y_train)
print("Best: %s. Using: %s" %(grid_result.best_score_, grid_result.best_params_))
```

## Naive Bayes

```
from sklearn.model_selection import GridSearchCV

param_grid_nb = {'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}   
model_nb_ty = MultinomialNB()  
kfold = KFold(10)
grid = GridSearchCV(
		stimator=model_nb_ty,
		param_grid=param_grid_nb,
		scoring='neg_mean_squared_error',
		cv=kfold)  
grid_result = grid.fit(x_train, y_train)
print("Best: %s. Using: %s" %(grid_result.best_score_, grid_result.best_params_))
```

## LDA

```
from sklearn.model_selection import GridSearchCV
param_grid_lda = {'solver': ['lsqr', 'eigen']}
model_lda_ty = LinearDiscriminantAnalysis(shrinkage='auto')
kfold = KFold(10)
grid = GridSearchCV(
		stimator=model_lda_ty,
		param_grid=param_grid_lda,
		scoring='neg_mean_squared_error',
		cv=kfold)  
grid_result = grid.fit(x_train, y_train)
print("Best: %s. Using: %s" %(grid_result.best_score_, grid_result.best_params_))
```

## Filnal Model

```
model_last = {}
models_last['LDA'] = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')   
models_last['NB'] = MultinomialNB(alpha=0)
models_last['KNN'] = KNeighborsClassifier(n_neighbors=5)  

...
```



# Reference

https://www.kaggle.com/code/prashant111/k-means-clustering-with-python

https://zhuanlan.zhihu.com/p/562538185

https://www.kaggle.com/code/jaganadhg/fb-live-selling-data-analysis

https://zhuanlan.zhihu.com/p/112794090
