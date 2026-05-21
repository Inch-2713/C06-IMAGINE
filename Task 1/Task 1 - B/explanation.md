  #  MACHINE LEARNING 
It is a field of computer science where computers learn patterns from data instead of being programmed with rules. Instead of writing step by step instructions, we train a model using data so it can make predictions on new, unseen data as well.

#	Difference between Traditional Programming and Machine Learning 
-	Traditional Programming 
In this we give: Input Data + Rules -> Output 
(ex: program to calculate area, input – radius, rules – area = pi*r^2, output = area)
-	Machine Learning 
Instead of rules, computer learns the rules from data set

#	What is a Dataset?
-	It is a collection of data used to train and evaluate ml models (usually given as a table).
-	Ex: Each row = data example, Each column = one variable

#	Features vs label
-	Features – Input variables used for prediction (ex: predict house price – size, bedrooms etc)
-	Labels – Output we want to predict 
-	Ex: For a particular size of the house and number of bedrooms it contains, the predicted price is the label

#	Training Data vs Testing Data
ML model must be evaluated on unseen data, so the data is split into :
-	Training Data – used to train model 
-	Testing data – sued to evaluate the model’s performance
Ex- dataset-1000 rows, training-800, testing-200

#	Model as Function Approximation
-	Model is a mathematical function ( y=f(x))
-	x-input features, y-predicted output


## SUPERVISED LEARNING
- Model learns from labeled data.

#	Regression 
-	Predicts continuous numerical values. (ex: house price, temperature, etc)
-	Evaluated by mean squared error or mean absolute error

#	Classification 
-	Predicts categories or classes. (ex: spam detection, image classification, etc)
-	Evaluated by accuracy, precision etc

# Real world applications of supervised learning
-	Spam detection
-	Credit card fraud detection
-	House price prediction
-	Medical diagnosis

#	Hypothesis
-	It is the function the model uses to make predictions
-	Different weights and biases are assigned which are changed accordingly to make the model better
-	h(x) = w1x1 + w2x2 + b (x-inputs, w-weights, b-bias)

#	Prediction Error
-	It is the difference between the actual value and predicted value

#	Optimization
-	Finding the best model parameters, adjusting weights
-	It is done through gradient descent

 
## LOSS FUNCTION
- Measures how bad the model predictions are, low loss implies better model 

#	Mean squared error 
-	Average of squared values of prediction errors
-	Removes negative error
-	Used for regression

#	Binary Cross entropy
-	Measures how far predicted probability 

#	What does loss function measure?
-	Measures the difference between the model’s predictions and the actual values
-	It is a measure of how bad the model predictions are.

#  Why does minimizing loss improve performance?
-	When loss decreases, prediction error decreases and model predictions become closer to the actual values
-	Model learns from the data

#	Why are different losses used for different tasks?
-	Different loss functions are designed to measure error for each type of problem in the best way
-	Regresssion predicts continuous values, uses mean square error 
-	Classification predicts probabilities, uses cross entropy.

 
## GRADIENT DESCENT
#	What is an optimization problem?
-	Finding parameters that minimize or maximize something
-	In ml, it is minimizing loss

#	Why do we compute gradients?
-	Gradient tells us direction of increase of loss function
-	If we move in the opposite direction, loss decreases
-	It helps us know how to change parameters to reduce error

#	Moving in negative gradient direction
-	We move in the opposite direction of gradient, which reduces loss
-	This step is repeated again and again

#	Learning rate 
-	It controls how big each update is.
-	Small learning rate – slow learning, stable
-	Large learning rate – faster learning, may overshoot minimum
 
## NUMPY AND PANDA 
Roll of NumPy and Pandas in ML workflow:
# NumPy
Used for:
-	Store data in arrays
-	Vectorized operations
-	Linear algebra
-	Statistical functions
# Panda 
Used for:
-	Loading CSV files
-	Loading excel sheets
-	Loading dataset
