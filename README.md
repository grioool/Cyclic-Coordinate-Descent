# Cyclic-Coordinate-Descent
The aim of the project is to implement Cyclic Coordinate Descent (CCD) algorithm for parameter
estimation in regularized logistic regression with l1 (lasso) penalty and compare it with standard
logistic regression model without regularization.

## Task 1
The aim of the first task is to collect and prepare data sets for conducting experiments.
1. Find 4 different real datasets corresponding to classification problem with binary class
variable containing numerical variables.
* You can use repositories: https://archive.ics.uci.edu/, https://www.openml.org/ or
other sources.
* Non-standard, interesting datasets will be appreciated. You can convert multi-class
datasets to binary datasets by combining classes (for example treat the majority class
as positive and assign the remaining observations to the negative class.).
* Please choose the datasets with large number of features, the number of variables
should be at least 50% of the number of observations. If it is not, you can add
dummy variables that are copies of the original variables with permuted values.
* Prepare datasets to run logistic regression algorithms. This includes:
▪ filling in missing values,
▪ removing collinear variables,
2.  Generate Synthetic dataset using the following procedure (p,n,d,g are treated as parameters
which will vary in the experiments):
* Generate binary class variable (Y=0 or Y=1) from Bernoulli distribution with class
prior probability p.
* Generate feature vector X, such that X|Y=0 follows d-dimensional multivariate
normal distribution with mean (0,…,0) and covariance matrix S with 𝑆[ⅈ, 𝑗] = 𝑔|ⅈ−𝑗|
,
whereas X|Y=1 follows d-dimensional multivariate normal distribution with mean
(1,1/2,1/3,…1/d) and covariance matrix 𝑆[ⅈ, 𝑗] = 𝑔|ⅈ−𝑗|
.
* Generate n observations using the above steps.

## Task 2
The aim of the second task is to implement regularized logistic regression using algorithm CCD1 (call
the method LogRegCCD). The implementation description can be found in the article
https://www.jstatsoft.org/article/view/v033i01 The most relevant description is in section 3.
However, the preceding sections also contain descriptions necessary to perform the implementation,
in particular formulas (4), (5) and (10). It is not necessary to use all the speed-up tricks described in
the article, although speeding up the method will count as a plus.
* The input of the method: training data (features X_train and labels y_train), validation data
(features X_valid and labels y_valid).
* The optimal value of the lambda parameter should be selected by optimizing the appropriate
measure on the validation set. The user should be able to choose the following measures:
recall, precision, F-measure, balanced accuracy (for a threshold of 0.5), area under the ROC
curve, area under the sensitivity-precision curve.
* The following methods should be implemented:
o fit(X_train,y_train),
o validate(X_valid, y_valid, measure),
o predict_proba(X_test),
o plot(measure, ...), which produces plot showing how the given evaluation measure
changes with lambda,
o plot_coefficients(...), which produces plot showing the coefficient values as function
of lambda parameter.

## Task 3
The aim of this task is to conduct experiments in which we will evaluate the performance of the
implemented method and compare it with the standard implementation of logistic regression
without regularization.
As the baseline, use the sklearn implementation:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
(LogisticRegression with penalty=’None’ and default value of the remaining parameters).
1.  Analyze how parameters n,p,d, g affect the performance of LogisticRegression and
LogRegCCD methods based on synthetic data. As an evaluation measure, consider ROC AUC
and Balanced Accuracy (for a threshold of 0.5).
2. Compare:
- the performance of the two methods on real datasets using ROC AUC, Recall-Precision AUC,
F-measure and Balanced Accuracy as the evaluation measures.
- coefficient values obtained in these two methods

# Requirements for code
* Instructions to run algorithm (README)
* Script, notebook to run the algorithm for new data
* Documentation of the functions - docstrings
* Readability of code and use of good practices for granularity of functions
* Saving all files (plots, tables) used for the report in separate files

# Instructions to run algorithm
