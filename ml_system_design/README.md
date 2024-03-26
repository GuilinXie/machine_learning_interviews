


# ML design framework

1. Clarifying requirements
2. Framing the probelm as an ML task
3. Data Preprocessing

   1. Data Preparation
   2. Data Exploration Analysis
   3. Feature Engineering
   4. Collect Training Labels
   5. Handle Class Imbalance
   6. Train/ Test Split
4. Model Development

   1. Training
   2. Evaluation
   3. Tuning
   4. Tracking / Logging
   5. Versioning
   6. Distributed Training
5. Test

   1. Code
   2. Data
   3. Models
6. Deployment and serving

   1. CI/CD workflows (TODO)
7. Monitoring and infrastructure

# Clarifying Requirements

### Business Objective

1. To increase the number of orders?
2. To increase the total revenue, sales by 20%?
3. To increase the clicks?
4. To increase the usage time, user retention?

### Features the system needs to support

1. Any interactions, such as 'Like', 'Dislike', 'Click' etc. These could be used as natrual labels.

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
2. How many items, such as videos, clothes, are we dealing with?
3. What's the rate of growth of these metrics?

### Performance

* How fast must the predition be, real-time or not real-time?
* Accuracy vs Latency, which is more important?
* Precision vs Recall, which is more important?

# Frame the Problem as ML Task

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
  * Supervisied learning
* Without labeled data
  * Unsupervisied learning
* Input is structured data
  * Decision Tree, Linear Model, SVM, KNN, Naive Bayes, Deep Learning
* Input is unstructured data
  * Deep Learning

# Data Preprocessing

## Data Preparation

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

## Data Exploration Analysis

1. Class imbalance
2. Number of features
3. Feature types: discrete, continuous
4. Feature statistics: mean, min, max, median, number of missing data
5. Are there any bias in the data? What kinds of biases are present? How do we correct i

## Feature Engineering

1. **For both Continuous & Categorical features**

   1. Handling missing data

      1. Deletion

         1. Row Deletion: if a data point has many missing values
         2. Column Deletion: if a feature has many missing values
      2. Imputation

         1. Defaults
         2. Mean, Median, or Mode(the most common value)
   2. Deiscretization( Bucketing)

      1. Convert a continuous feature into a categorical feature
      2. Reduce the number of categories for a categorical feature
2. **For only Continuous features**

   1. Feature scaling: scaling features to have a standard range and distribution

      1. Normalization (min-max scaling)
         z = (x - x_min) / (x_max - x_min)
         Normalizaiton doesn't change the distribution of the feature.
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
4. **For Images**

   Steps

   1. Resize
   2. Normalization
   3. Data augmentation (Userful for class imbalance, and the model can learn more complex pattern)
      1. Random crop
      2. Rescale
      3. Random saturation / brightness
      4. Vertical / horizontal flip
      5. Rotation / translation
      6. Mix up
      7. Mosaic
      8. Add noise
5. **For Text**

   1. Text Normalization        E.g. "A person is walking in the street." -> "A person walk in the street"
      1. Lowercasing
      2. Punctuation removal
      3. Trim whitespaces
      4. Normalization Form KD: TODO
      5. Strip accents
      6. Lemmatization and stemming
   2. Tokenization            ['a', 'person', 'walk', 'in', 'the', 'street']
      1. Word tokenization
      2. Subword tokenization
      3. Character tokenization
   3. Tokens to IDs          [33, 20, 8, 25, 70]
      1. Lookup table
      2. Hashing
   4. Text encoder
      1. Statistical methods
         1. BoW
         2. TF-IDF
      2. ML-based methods
         1. Embedding (lookup) Layer
         2. Word2vec
         3. Transformer-based models
6. For video

   1. Decode frames
   2. Sample frames
   3. Preprocess like images
      1. Resize
      2. Scale, normalize, correcting color mode
7. TODO

## Collect Labels

1. Hand labeling
   1. How good is the label
2. Natural labeling
   1. How do we get them
   2. How do we receive user feedback on the system
   3. How long does it take to get natural labels

## Handle Class Imbalance

1. Resampling training data
   1. Oversampling
   2. Downsampling
   3. Stratified sampling
   4. Importance weighted sampling
   5. Reservoir sampling
   6. Convenience sampling
2. Altering loss function
   1. Class weighted loss
   2. Class balanced loss
   3. Focal Loss

## Train / Test Split

Divide the dataset into training, validation, test splits, often in 70%, 20%, 10%.

# Model Development

## Model Selection

### Choose the best ML Algorithm and Architecture

1. Establish a simple baseline
   1. E.g. To recommend the most popular videos, for a video recommendation system.
2. Experiment with simple models
   1. E.g. Apply a ML algorithms that are quick to train, such as Logistic Regression
3. Switch to more complex models
   1. Deep neural networks
4. Use an ensemble of models for more accurate predictions
   1. Bagging
   2. Boosting
   3. Stacking

### Considerations

1. The amount of data the model needs to train on
2. Training speed, the time it takes to train
3. Inference speed,
4. Computation resource limitation
5. Model's interpretability
6. Can the model be deployed on a user's device
7. Transfer training or training from scratch
8. How many parameters of the model? How large is the model? How much memory is needed?
9. Hyperparameters to choose and hyperparameters tuning techques
10. For neural networks, discuss

    1. Typical architectures/blocks
       1. ResNet
       2. Transformer-based architectures.
       3. etc.
    2. Choice of hyperparameters
       1. The number of hidden layers
       2. The number of neurons
       3. Activation functions

### Pros & Cons of Models

| Model                                    | Pros                                                                                                                                          | Cons                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Stability                                                                   | Use Cases                                                                                                                                       |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Linear Regression                        | 1. simple,<br /> fast training,<br /> interpretable                                                                                          | 1. Assume Linearity<br />2. Sensitive to Outliers<br />3. Assume features are independent, <br />features that have high correlation can result in an unreliable model<br />4. Assume Homoskedacity. <br />constant variance around the mean for x and y<br />5. Can not determine feature importance for high correlation features<br />6. Number of observations Greater than the number of features<br />7. Assume each observation is independent<br />8. Each feature is distributed normally<br />9. x/y are related linearly | Not Stable                                                                  | With Lasso(L1) and Ridge(L2)<br />1. Data Preprocessing<br />2. Handle outliers<br />3.Dimensionality reduction<br />Remove correlated features |
| Logistic Regression<br />LR              | 1. Fast, simple<br />2. See the positive/negative <br />relation between x/y<br />3. Interpret model coefficients <br />as feature importance | 1. Number of observations greater than the number of features<br />2. Linear boundaries, assume a linearity bwtween x/y<br />3. Predict discrete numbers<br />4. Assume average or no multicollinearity between features<br />5. x/y are related to the log odds, log(p/(1-p))                                                                                                                                                                                                                                                        | Not Stable                                                                  | 1. Binary classification<br />2. Multiclass classification                                                                                      |
| Dicision Tree                            | 1. Not require normalization of data<br />2. Not require scaling of data<br />3. Not require dealing with missing data<br />4. Interpretable  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Not Stable                                                                  |                                                                                                                                                 |
| Random Forest<br />RF                    |                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Stable                                                                      |                                                                                                                                                 |
| Gradient Boosted Decision Tree<br />GBDT |                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Stable                                                                      |                                                                                                                                                 |
| Support Vector Machine<br />SVM          |                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Not Stable                                                                  |                                                                                                                                                 |
| Naive Bayes                              |                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                             |                                                                                                                                                 |
| Factorization Machine<br />FM            |                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                             |                                                                                                                                                 |
| Neural Networks                          |                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Not Stable<br />Solutions:<br />1. Add noise while training<br />2. Dropout |                                                                                                                                                 |

## Model Training

### Choose Loss Function

A loss function is used to measure how accurate the model is at predicting an expected outcome.

The optimization algorithm is used to update the model's parameters during the training process in order to minimize the loss, by backpropagation.

1. What loss function to use?
   1. Cross-entropy
   2. Binary cross-entropy
   3. Focal Loss Cross-entropy
   4. Focal Loss Binary cross-entropy
   5. Weighted loss for class-imbalance
   6. Mean Square Error (MSE)
   7. Mean Absolute Error (MAE)

### Choose Regularization

1. L1
2. L2
3. K-fold CV
4. dropout
5. Entropy Regularization

### Choose Optimizer

1. SGD
2. AdaGrad
3. Momentum
4. RMSProp
5. Adam

### Choose Activation functions

Which to choose and why to choose this: TODO

1. ELU
2. ReLU
3. Tanh
4. Sigmoid

### Bias/Variance Trade-off

TODO

### Overfitting and underfitting

1. How to detect overfitting and underfitting
   1. TODO
2. How to solve overfitting
   1. TODO
3. How to solve underfitting
   1. TODO

### 3 steps for training NN

1. Forward propagation
2. Loss function
3. Backward propagation

### From scratch vs fine-tuning

### How often to retrain

daily, weekly, monthly, or yearly

### Distributed training

Data parallelism

model parallelism

## Model Evaluation

Evaluate model's performance based on different metrics.

1. Online metrics
   1. Which metrics are important for measuring the effectiveness of ML system online?
   2. How do these metrics relate to the business objective?
2. Offline metrics
   1. Which metrics are good to evaluate the model's predictions during the development phase?
3. Fairness and bias
   1. Does the model have the potential for bias across different attributes such as age, gender, race, etc.?
      1. How to detect model's bias
         1. TODO
   2. How would you fix this?
   3. What happens if someone with malicious intent gets access to your system?

### Offline evaluation

To measure how close the predictions are to the ground truth values.

Metrics to use and their trade-offs: TODO

| Task                        | Offline Metrics                                                          |
| --------------------------- | ------------------------------------------------------------------------ |
| Classification              | Precision, Recall, F1 score, Accuracy, ROC-AUC, PR-AUC, Confusion Matrix |
| Regression                  | MSE, MAE, RMSE                                                           |
| Object Detection            | mAP, AP, Precision, Recall, metrics based on object's size and class     |
| Ranking                     | Precision@k, recall@k, MRR, mAP, nDCG                                    |
| Image Generation            | FID, Inception score                                                     |
| Natural Language Processing | BLEU, METEOR, ROUGE, CIDEr, SPICE                                        |

### Online evaluation

To measure the model perforMs in production after deployment.

Usually these metrics are tied to business objectives.

| Problem                   | Online metrics                                                             |
| ------------------------- | -------------------------------------------------------------------------- |
| Ad click prediction       | Click-through rate, revenue increase, etc.                                 |
| Harmful content detection | TODO                                                                       |
| Video recommendation      | Click-through rate, total watch time, number of completed videos, etc.     |
| Friend recommendation     | Number of requests sent per day, number of requests accepted per day, etc. |

# Model Deployment

## Cloud vs. on-device deployment

TODO

## Model compression

To make a model smaller, it can reduce

1. the inference latency
2. model size

### Knowledge distillation

To train a small model (student) to mimic a larger model (teacher).

### Pruning

To find the least useful parameters and set them to zero.

Then we can have a sparser model, and we can store it more efficiently.

### Quantization

Usually, the model is represented with 32-bit floating numbers.

Here, we can can reduce the model size by using fewer bits, such as half model with 16-bit floating numbers.

It can happen during training or post-training.

## Test in production

### Shadow deployment

Deploy the new model in parallel with the existing model.

Each incoming request is routed to both models, but only the existing model's prediction is served to the user.

Pros: minimize the risk of unreliable predictions

cons: expensive

### A/B testing

Deploy the new model in parallel with the existing model.

A portion of the traffic is routed to the newly developed model, such as 5% of the traffic.

Two things to keep in mind for A/B testing:

1. The traffic routed to each model has to be random.
2. A/B test should be run on a sufficient number of data points in order for the results to be legitimate.

### Batch/Online Prediction

| Type              | Pros                                                              | Cons                                                                                                              | Use cases                                                                                 |
| ----------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Batch Prediction  | No worry about inference time, as it is pre-computed periodically | 1. Less responsive to the chaning preference of users<br />2. Only possible if we know in advance what to compute | 1. Need to process a large volume of data<br />2. The results are not needed in real time |
| Online Prediction |                                                                   | 1. May take too long to generate predictions                                                                      | 1. We do not know what to compute in advance                                              |

### Real-time features

Is real-time access to features possible?

What are the challenges?

TODO

# Monitor

## Why a system fails in production

### Data Distribution Shift

Solutions:

1. Train on large datasets
2. Regularly retrain the model using labeled data from the new distribution

## What to Monitor

### Operation-related metrics

To ensure the system is up and running

1. Average serving time
2. thoughput
3. the number of prediction requests
4. CPU/GPU utilization

### ML specific metrics

1. Input/ouput monitoring
2. Input/output distribution
3. Accuracy of the model: expect to be in a specific range
4. Model versions


# References

1. Book, Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications. By Chip Huyen
2. Book, Machine Learning System Design Interview: An Insider Guide. By Ali Aminian & Alex Xu
3. Github, [Machine Learning Interviews](https://github.com/alirezadir/Machine-Learning-Interviews).
4. Book, Designing Data-Intensive Application. By Martin Kleppmann
