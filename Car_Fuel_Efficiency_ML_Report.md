# **Car Fuel Efficiency Prediction Using Machine Learning**

Submitted by:
B6737139 - Patrick Oliver Montano
B6741068 - Paing Nyein Oo
B6737115 - Sokvisal Leng

## **1. Background and Significance**


Fuel efficiency is an important factor in vehicle performance, transportation cost, and
environmental impact. A car with higher fuel efficiency consumes less fuel to travel the
same distance, which reduces fuel expenses and lowers fuel consumption. Fuel
efficiency is commonly measured using Miles Per Gallon (MPG), which represents how
many miles a vehicle can travel using one gallon of fuel.


In real-world vehicle analysis, MPG is affected by several characteristics, including
cylinders, engine displacement, horsepower, vehicle weight, acceleration, model year,
and origin. These variables do not always affect MPG in a simple linear way. For
example, a heavier car with a larger engine usually has lower MPG, but model year and
vehicle origin may also influence the result because of differences in technology and
manufacturing design.


Machine learning is useful for this problem because it can learn patterns from historical
vehicle data and use those patterns to estimate MPG for new vehicles. This project
uses the Auto MPG dataset and compares several regression models to predict car fuel
efficiency.

## **2. Objectives**


The main objective of this project is to develop a machine learning model that predicts
car fuel efficiency based on vehicle characteristics.


The specific objectives are:


  - To study the relationship between vehicle input features and MPG.

  - To clean and prepare the Auto MPG dataset for machine learning.

  - To train multiple regression models for MPG prediction.

  - To compare model performance using MSE, RMSE, R² Score, and CV-RMSE.

  - To identify the best-performing model for fuel efficiency prediction.

  - To analyze the impact of each input feature on the predicted MPG.

  - To test sample predictions using the trained models.

  - To tune the SVR model hyperparameters and compare R² results.


## **3. Scope of the Project**

This project focuses on supervised machine learning regression for car fuel efficiency
prediction. The target variable is mpg, while the original input features are:


  - cylinders

  - displacement

  - horsepower

  - weight

  - acceleration

  - model year

  - origin


The project includes dataset collection, data cleaning, preprocessing, training,
evaluation, feature analysis, hyperparameter tuning, and sample prediction. It does not
include real-time vehicle sensor data, modern fuel price prediction, mobile application
deployment, or web deployment.

## **4. Methodology Based on the Steps of Machine Learning**


**Step 0: Determine Machine Learning Requirements**


The first step was to define the machine learning problem clearly. The project is a
regression problem because the output, MPG, is a continuous numerical value.


The machine learning requirements are:


Requirement Description
Problem type Regression
Target output MPG
Input features Cylinders, Displacement, Horsepower,
Weight, Acceleration, Model Year, Origin
Dataset Auto MPG dataset
Evaluation metrics MSE, RMSE, R² Score, CV-RMSE
Main goal Build a model that predicts MPG with low
error and high R²


**Step 1: Gathering Data**


The dataset used in this project is the Auto MPG dataset from the UCI Machine
Learning Repository. It contains vehicle information and fuel efficiency values.


The original dataset columns are:


Column Description
mpg Fuel efficiency in miles per gallon


Column Description
cylinders Number of engine cylinders


displacement Engine displacement


horsepower Engine horsepower


weight Vehicle weight


acceleration Acceleration value


model_year Vehicle model year


origin Vehicle origin category


car_name Vehicle name


The project code loads the dataset directly from the UCI source:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower',
'weight', 'acceleration', 'model_year', 'origin', 'car_name']

data = pd.read_csv(url, names=column_names, na_values='?', sep=r'\s+', comment='\t')


After removing missing values, the dataset contained 392 usable records.


**Step 2: Preparing the Data**


The dataset was prepared before training the models. Missing horsepower values were
represented by “?”, so they were converted into NaN and removed using dropna(). The
car_name column was also removed because it is a text field and was not used as a
numerical model input.


data = data.dropna().drop(['car_name'], axis=1)

X = data.drop(['mpg'], axis=1)
y = data['mpg']


The origin feature is categorical, so it was converted using one-hot encoding:


X = pd.get_dummies(X, columns=['origin'], drop_first=True)


After encoding, the model used these numerical input columns:


  - cylinders

  - displacement

  - horsepower

  - weight

  - acceleration


  - model_year

  - origin_2

  - origin_3


The original report still describes seven vehicle inputs because origin_2 and origin_3 are
encoded versions of the single original origin feature.


**Step 3: Choosing a Model**


Several regression models were selected and compared:


Model Reason for Use
Linear Regression Simple baseline model
Ridge Regression Linear model with L2 regularization
Lasso Regression Linear model with L1 regularization
Elastic Net Combines L1 and L2 regularization
Random Forest Captures nonlinear patterns using decision
trees
Gradient Boosting Builds strong predictions from sequential
weak learners
SVR with RBF Kernel Captures nonlinear relationships using
support vectors
Polynomial Regression Degree 2 Adds nonlinear polynomial feature
interactions


The purpose of comparing different models was to identify which approach predicted
MPG most accurately. The best test performance in the printed results was achieved by
SVR with the RBF kernel.


**Step 4: Training**


The dataset was split into training and testing sets using an 80:20 split:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


After cleaning, the dataset had 392 records:


Split Number of Records Percentage
Training set 313 80%
Testing set 79 20%


The 80:20 split was selected because it gives the model enough training data while still
keeping a reasonable number of testing samples. Since the dataset is small, a 90:10
split gives a smaller test set and can make evaluation less reliable. A 70:30 split gives


more test data but reduces training data. Therefore, 80:20 is a balanced choice for this
project.


The SVR split comparison was:


Train: Test Split Test Samples RMSE R² CV-RMSE
90:10 40 1.859 0.9419 2.784
85:15 59 2.122 0.9231 2.772
80:20 79 2.225 0.9030 2.919
75:25 98 2.307 0.8945 2.901
70:30 118 2.358 0.8949 2.840


Although 90:10 produced a higher test R², it only used 40 test samples. The 80:20 split
is more balanced and gives a more dependable evaluation for the report.


_Feature Scaling_


Feature scaling was applied using StandardScaler:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


The purpose of scaling is to put all features on a similar scale. This is important
because features such as weight and horsepower have much larger values than
acceleration or origin. SVR, Linear Regression, Ridge, Lasso, and ElasticNet can be
affected by feature scale, so scaling helps the model train more fairly and improves
stability.


_Model Training Code_


The SVR model was trained using an RBF kernel:


**from** sklearn.svm **import** SVR

svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)

y_pred = svr_model.predict(X_test_scaled)


_SVR Parameters Used in Training_


Parameter Value Meaning
Kernel, K RBF Allows the model to learn
nonlinear relationships


Parameter Value Meaning
C 100 Controls the penalty for
prediction errors


gamma 0.1 Controls how far the
influence of one training
point reaches


epsilon 0.1 Defines the margin where
small errors are ignored


bias term 23.4216 The fitted SVR intercept,
stored in
svr_model.intercept_


The RBF kernel function is:


2

𝐾(𝑥𝑖, 𝑥𝑖, ) = 𝑒 [(−𝛾||𝑥][𝑖][−𝑥][𝑖][||] )


_Support Vectors_


The trained SVR model used 297 support vectors out of 313 training samples, which is
94.89% of the training set.


A support vector is a training data point that directly affects the final SVR prediction
boundary. In SVR regression, support vectors are usually the points outside the epsilon
margin or on the margin. These points are important because the model uses them to
form the prediction function. Points inside the epsilon margin have less influence
because their prediction error is considered acceptable.


The support-vector count can be printed with:


print(len(svr_model.support_))
print(svr_model.support_vectors_.shape[0])


**Step 5: Evaluation**


The models were evaluated using MSE, RMSE, R² Score, and CV-RMSE.


Model MSE RMSE R² Score CV-RMSE
Linear Regression 10.602 3.256 0.7923 3.445
Ridge Regression 10.656 3.264 0.7912 3.442
Lasso Regression 10.795 3.286 0.7885 3.429
ElasticNet 10.926 3.305 0.7859 3.471


Model MSE RMSE R² Score CV-RMSE
Random Forest 5.691 2.386 0.8885 3.005
Gradient Boosting 6.702 2.589 0.8687 2.916
SVR with RBF Kernel 4.951 2.225 0.9030 2.919
Polynomial Regression Degree 2 7.657 2.767 0.8500 2.904


The SVR with RBF Kernel achieved the best test result, with:


Metric Value
MSE 4.951
RMSE 2.225
R² Score 0.9030
CV-RMSE 2.919


The R² Score of 0.9030 means the SVR model explained about 90.30% of the variation
in MPG on the test set.


_Actual vs Predicted Result_


The actual vs predicted graph shows how close each model’s predicted MPG values
are to the real MPG values. Points closer to the red diagonal line represent better
predictions.


Actual vs Predicted MPG - All Models


The model comparison graph also shows that SVR had the lowest test RMSE and the
highest test R².


The observed relationships are:


Input Feature Relationship with MPG
cylinders Negative relationship. Cars with more
cylinders usually have lower MPG.
displacement Strong negative relationship. Larger
engines usually reduce MPG.
horsepower Strong negative relationship. Higher
horsepower often means more fuel
consumption.
weight Strong negative relationship. Heavier cars
usually have lower MPG.
acceleration Weak positive relationship. Cars with
higher acceleration values may show
slightly higher MPG.
model year Positive relationship. Newer model years
tend to have better MPG.


Input Feature Relationship with MPG
origin Origin groups 2 and 3 show higher
average MPG than origin group 1.


_Feature Importance Ranking_


Because SVR does not provide built-in tree-style feature importance, permutation
importance was used. This measures how much the model’s R² decreases when a
feature is shuffled. A larger decrease means the feature is more important.


The SVR feature importance ranking was:


Rank Feature Importance
1 horsepower 0.3725
2 displacement 0.3719
3 model year 0.3359
4 cylinders 0.1623
5 weight 0.0891
6 origin 0.0806
7 acceleration 0.0388


The most influential features were horsepower, displacement, and model year. This is
reasonable because engine power, engine size, and vehicle technology generation
strongly affect fuel efficiency.


**Step 6: Hyperparameter Tuning**


Hyperparameter tuning was performed for the SVR model using cross-validation R².
The main parameters tested were C, gamma, and epsilon.


Parameter Purpose
C Controls how strongly the model penalizes
prediction errors
gamma Controls the influence range of each
training point in the RBF kernel
epsilon Defines the no-penalty error margin
around the prediction


The tuning graph is shown below.


SVR Hyperparameter Tuning Results


_Tuning C_


The tuning test changed to C while keeping gamma=0.1 and epsilon=0.1.


C Mean CV R²
1 0.8185
10 0.8684
50 0.8679
100 0.8623
200 0.8511


The best single test for C was C=10, with a mean CV R² of 0.8684.


_Tuning Gamma_


The tuning test changed gamma while keeping C=100 and epsilon=0.1.


gamma Mean CV R²
scale 0.8550
0.01 0.8701
0.05 0.8718
0.1 0.8623
0.2 0.8423


The best single test for gamma was gamma=0.05, with a mean CV R² of 0.8718.


_Turning Epsilon_


The tuning test changed epsilon while keeping C=100 and gamma=0.1.


epsilon Mean CV R²
0.01 0.8614
0.1 0.8623
0.2 0.8629
0.5 0.8597


The best single test for epsilon was epsilon=0.2, with a mean CV R² of 0.8629.


_Combined Grid Search Result_


A combined grid search tested multiple values of C, gamma, and epsilon together. The
best cross-validation result was:


C gamma epsilon Mean CV R²
50 0.05 0.5 0.8799


The top combined tuning results were:


Rank C gamma epsilon Mean CV R²
1 50 0.05 0.5 0.8799
2 50 0.05 0.1 0.8772
3 50 0.05 0.2 0.8795
4 200 0.01 0.5 0.8754
5 100 0.01 0.5 0.8750


Based on cross-validation, the best tuned SVR parameters were:


SVR(kernel='rbf', C=50, gamma=0.05, epsilon=0.5)


This parameter set was selected by cross-validation because it had the highest mean
CV R². However, the original SVR settings used in the printed project output, C=100,
gamma=0.1, and epsilon=0.1, achieved the highest reported test R² of 0.9030. For a strict


machine learning workflow, the cross-validation result should be used to choose
parameters, then the final model should be evaluated once on the test set.


The comparison between the original SVR and the tuned SVR was:



SVR
Version C gamma epsilon



SVR Test CVVersion C gamma epsilon RMSE Test R² CV R² RMSE

Original 100 0.1 0.1 2.225 0.9030 0.8623 2.919
project
SVR



Test
RMSE Test R² CV R²



100 0.1 0.1 2.225 0.9030 0.8623 2.919



Best CVtuned SVR



50 0.05 0.5 2.323 0.8942 0.8769 2.779



The original SVR had the best test R² in the current project output, while the tuned SVR
had the better cross-validation R² and CV-RMSE. Therefore, the final selected model for
the report remains SVR with RBF Kernel, and the recommended tuned version for
future retraining is C=50, gamma=0.05, and epsilon=0.5.


**Step 7: Prediction**


After training, the models were used to predict MPG for a sample vehicle.


The sample input was:


Feature Value
cylinders 8
displacement 307
horsepower 130
weight 3504
acceleration 12
model year 70
origin 1


The sample prediction results were:


Model Predicted MPG
Linear Regression 14.82
Ridge Regression 14.84
Lasso Regression 15.01
Elastic Net 14.88
Random Forest 16.27
Gradient Boosting 15.25
SVR with RBF Kernel 17.36
Polynomial Regression Degree 2 14.68


The final selected SVR model predicted:


17.36 MPG

## **5. Results Analysis**


The results show that the simple linear models performed reasonably but were weaker
than the nonlinear models. Linear Regression, Ridge, Lasso, and ElasticNet had RMSE
values around 3.25 to 3.31 and R² values around 0.78 to 0.79. These models could
identify general relationships, but they could not capture more complex nonlinear
patterns in the data.


Tree-based models performed better. Random Forest achieved an RMSE of 2.386 and
an R² Score of 0.8885, while Gradient Boosting achieved an RMSE of 2.589 and an R²
Score of 0.8687. These models performed well because they can capture nonlinear
interactions between features.


The best test result was achieved by SVR with RBF Kernel. It produced an RMSE of
2.225 and an R² Score of 0.9030. This means the model predicted MPG with the
smallest test error and explained about 90.30% of the variation in MPG.


The relationship graphs show that cylinders, displacement, horsepower, and weight
generally have negative relationships with MPG. This means larger, heavier, and more
powerful vehicles usually consume more fuel. Model year has a positive relationship
with MPG, which suggests that newer vehicles became more fuel efficient. Origin also
has an effect, with origin groups 2 and 3 showing higher average MPG than origin group
1.


The feature importance ranking showed that horsepower, displacement, and model year
were the strongest predictors for the SVR model. This supports the idea that engine
power, engine size, and vehicle generation are major factors in fuel efficiency.

## **6. Conclusion**


This project successfully developed and compared machine learning models for car fuel
efficiency prediction. The Auto MPG dataset was cleaned, prepared, scaled, and used
to train several regression models.


Among all tested models, SVR with RBF Kernel achieved the best test performance,
with an RMSE of 2.225 and an R² Score of 0.9030. The model also predicted 17.36
MPG for the sample vehicle input.


The project showed that machine learning can effectively predict fuel efficiency using
vehicle characteristics. The most important features were horsepower, displacement,
and model year. In general, larger engines, higher horsepower, and heavier vehicles
were associated with lower MPG, while newer model years were associated with better
MPG.


Future improvements could include using a larger modern vehicle dataset, testing more
advanced models, applying more detailed hyperparameter tuning, adding real-world
vehicle features, and deploying the model as a simple web application.

## **References**


UCI Machine Learning Repository. Auto MPG Dataset.
https://archive.ics.uci.edu/ml/datasets/auto+mpg


