#Regression Template
#Importing the dataset
dataset = read.csv('Data.csv')

#Encoding categorical data
# dataset$Age = ifelse(is.na(dataset$Age), 
#                      ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), 
#                      dataset$Age)
# dataset$Salary = ifelse(is.na(dataset$Salary), 
#                         ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), 
#                         dataset$Salary)
# 
# dataset$Country = factor(dataset$Country, 
#                          levels = c('France', 'Spain', 'Germany'), 
#                          labels = c(0, 1, 2))
# dataset$Purchased = factor(dataset$Purchased, 
#                            levels = c('No', 'Yes'), 
#                            labels = c(0, 1))

#Splitting data set into training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split== TRUE)
test_set = subset(dataset, split== FALSE)

#Feature Scaling
#training_set[, 2:3] = scale(training_set[, 2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting regression model to the dataset
#Create your regressor here

#Predicting a new result with
y_pred = predict(poly_reg, data.frame(Level = 6.5)

#Visualizing the linear regression results
library(ggplot2)
#For High Resolution Curves
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() + geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = data.frame(Level = X_grid))), color = 'blue') + 
  ggtitle('Truth or Bluff (Regression)') + xlab('Level') + ylab('Salary')




                                      


