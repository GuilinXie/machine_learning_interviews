# Table of Contents

* [ML Coding Questions](#ML-Coding-Questions)
* [ML System Design](#ML-System-Design)
* [ML Take Home Challenge](#ML-Take-Home-Challenge)
* [References](#References)


# ML Coding Questions

| ID | Topic              | Solution                      | Knowledge Notes                                                                                                                                 | Reference                                                                                                                                                                                 |
| -- | ------------------ | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | Batch Norm         | Python, Numpy, nn.BatchNorm2d | mean, population standard deviation, sample standard deviation,<br />numpy, nn.BatchNorm2d(num_features=2, eps=1e-5)                            | [Enzo_Mi Bilibili](https://www.bilibili.com/video/BV11s4y1c7pg/?spm_id_from=333.999.0.0&vd_source=cce459bd59b16eaede26f2352c4eb26c)  <br />[AI zhihu](https://zhuanlan.zhihu.com/p/269465213) |
| 2  | Layer Norm         | Python, Numpy, nn.LayerNorm   | nn.LayerNorm(normalized_shape=3)                                                                                                                | [Enzo_Mi Bilibili](https://www.bilibili.com/video/BV1UG411f7DL/?spm_id_from=333.788&vd_source=cce459bd59b16eaede26f2352c4eb26c)                                                              |
| 3  | Dice Loss          | Python, PyTorch               | torch.sum(), F1_score                                                                                                                           | [zhihu](https://zhuanlan.zhihu.com/p/269592183)                                                                                                                                              |
| 4  | Focal Loss         | Python                        | Cross Entropy, Binary Cross Entropy, F.softmax, torch.sigmoid, torch.log,<br />torch.sum(dim=1), torch.ones, torch.view, torch.pow, F.one_hot() | [focal loss definition](https://zhuanlan.zhihu.com/p/49981234)<br />[focal loss implementation](https://zhuanlan.zhihu.com/p/308290543)                                                         |
| 5  | KMeans             | Python, Numpy                 | Recursive, np.random.randint, np.sqrt, np.square, np.sum, np.all(), min(list), list.index                                                       | [zhihu](https://zhuanlan.zhihu.com/p/293096829)                                                                                                                                              |
| 6  | Reservoir Sampling | Python                        | random.randint,                                                                                                                                 |                                                                                                                                                                                           |
| 7  | cumSum             | Python                        | torch.cumsum, np.cumsum, sum                                                                                                                    |                                                                                                                                                                                           |
|    |                    |                               |                                                                                                                                                 |                                                                                                                                                                                           |


# ML System Design

## ML design framework

1. Clarifying requirements
2. Framing the problem as an ML task
3. Data Preprocessing

   1. Data Preparation
   2. Data Exploration Analysis
   3. Feature Engineering
   4. Train/ Test split
   5. Distributed
4. Model Development

   1. Training
   2. Evaluation
   3. Tuning
   4. Tracking / Logging
   5. Versioning
5. Test

   1. Code
   2. Data
   3. Models
6. Deployment and serving

   1. CI/CD workflows
7. Monitoring and infrastructure

## Clarifying Requirements

### Business Objective

1. To increase the number of orders?
2. To increase the total revenue, sales by 20%?
3. To increase the clicks?
4. To increase the usage time, users retention?

### Features the system needs to support

1. Any interactions, such as 'Like', 'Dislike', 'Click' etc. These could be used as natural labels.

### Data

1. Data Sources, csv?
2. Data volumn, large, small?
3. Is the data labeled?

### Constraints

1. How much computing power is available?
2. Cloud-based system or local device?
3. Is the model expected to improve automatically over time?
4. One model or more models?

### Scale of the system

1. How many users do we have?
1. How many items, such as videos, clothes, are we dealing with?
1. What's the rate of growth of these metrics?

### Performance

* How fast must the prediction be, real-time or not real-time?
* Accuracy vs Latency, which is more important?
* Precision vs Recall, which is more important?

## Frame the Problem as ML Task

### ML objective from Business objective

A ML objective is one that ML models can solve.

| Application                                          | Business Objective                                  | ML Objective                                     |
| ---------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------ |
| Event ticket selling app                             | Increase ticket sales                               | Maximize the number of event registration        |
| Video streaming app                                  | Increase user engagement                            | Maximize the time users spend watching videos    |
| Ad click prediction system                           | Increase user clicks                                | Maximize click-through rate                      |
| Harmful content detection in a social media platform | Improve the platform's safety                       | Accurately predict if a given content is harmful |
| Friend recommendation system                         | Increase the rate at which users grow their network | MaxImize the number of formed connections        |

### Input & Output ｜ ML Category

* Output is a continuous numerical value
  * Regression
* Output is a discrete class label
  * Binary classification
  * Multiclass classification
* With labeled data
  * Supervised learning
* Without labeled data
  * Unsupervised learning
* Input is structured data
  * Decision Tree, Linear Model, SVM, KNN, Naive Bayes, Deep Learning
* Input is unstructured data
  * Deep Learning

## Data Preprocessing

### Data Preparation

1. How clean is the data?
2. Who generates the data, user-generated or system-generated?
3. Data source
   1. SQL
      1. Relational
         1. MySQL
         2. PostgreSQL
   2. NoSQL
      1. Key/Value
         1. Redis
         2. DynamoDB
      2. Document
         1. MongoDB
         2. CounchDB
      3. Column-based
         1. Cassandra
         2. HBase
      4. Graph
         1. Neo4J
4. Data Types
   1. Structured : follows a predefined schema
      1. Numerical
         1. Discrete
         2. Continuous
      2. Categorical
         1. Ordinal
         2. Nominal
         3. Scale
   2. Unstructured: No data schema
      1. Image
      2. Text
      3. Video
      4. Audio

### Data Exploration Analysis

1. Class imbalance
2. Number of features
3. Feature types: discrete, continuous
4. Feature statistics: mean, min, max, median, number of missing data
5. Are there any bias in the data? What kinds of biases are present? How do we correct i

### Feature Engineering

1. **For both Continuous & Categorical features**

   1. Handling missing data

      1. Deletion

         1. Row Deletion: if a data point has many missing values
         2. Column Deletion: if a feature has many missing values
      2. Imputation

         1. Defaults
         2. Mean, Median, or Mode(the most common value)
   2. Discretization( Bucketing)

      1. Convert a continuous feature into a categorical feature
      2. Reduce the number of categories for a categorical feature
2. **For only Continuous features**

   1. Feature scaling: scaling features to have a standard range and distribution

      1. Normalization (min-max scaling)
         z = (x - x_min) / (x_max - x_min)
         Normalization doesn't change the distribution of the feature.
      2. Standardization (Z-score normalization)

         1. z = (x- u) / sigma
            Standardization changes the distribution of the feature
      3. Log scaling

         1. z = log(x)
         2. Mitigate the skewness of a feature, and enable the optimization algorithm to converge faster
3. **For only Categorical features**
   convert categorical features into numeric representation

   1. integer encoding
      1. This method is useful if the integer values have a natural relationship with each other(ordinal), such as excellent/good/bad -> 1/2/3
   2. one-hot encoding
      1. red/greed/blue ->100/010/001
   3. embedding encoding
      1. An embedding is a mapping of a categorical feature into a N-dimensional vector.
      2. This method is useful when the number of unique values the feature takes is very large. In this case, one-hot encoding is not a good option because it leads to very large vector sizes.



TO BE CONTIN


# ML Take Home Challenge



TO BE DONE



# References

1. Book, Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications. By Chip Huyen
2. Book, Machine Learning System Design Interview: An Insider Guide. By Ali Aminian & Alex Xu
3. Github, [Machine Learning Interviews](https://github.com/alirezadir/Machine-Learning-Interviews).
4. Book, Designing Data-Intensive Application. By Martin Kleppmann
