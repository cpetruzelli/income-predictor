In this analysis we will try to obtain the best possible model to classify income based on several numerical and categorical variables. The data comes from the UCI Machine Learning Repository, and can be found at the following link: 
https://archive.ics.uci.edu/ml/datasets/adult
This data was extracted from the 1994 Census Bureau Database by Ronny Kohavi and Barry Becker. 

This analysis is broken into several different parts:
1. We will first search for any discrepancies in the feature variables.
2. We will search for missing values and decide whether to replace them or remove them. Removing missing values is not always an option, and it may drastically effect the training dataset, which will greatly effect the accuracy of our model. 
3. We will look at the correlation between all of our feature values. We will also perform some feature reduction to see if we can safely remove some features. 
4. There are several categorical variables in our dataset. We will combine some of our classes into a single class to avoid repitition of values of the same class with a different name. 
5. We will convert the categorical variables to dummy variables using the get_dummies method of the pandas package.
6. We will use univariate feature selection and recursive feature elimination with cross validation to choose the best features for our model. 
