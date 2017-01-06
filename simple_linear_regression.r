#Simple Linear Regression
#Importing the dataset
dataset = read.csv('Salary_Data.csv')


#Splitting data set into training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split== TRUE)
test_set = subset(dataset, split== FALSE)

#Feature Scaling
#training_set[, 2:3] = scale(training_set[, 2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting simple linear regression model in R
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

#Predicting the test set results
y_predict = predict(regressor, newdata = test_set)

#Visualizing the training set results
install.packages('ggplot2')
library(ggplot2)
ggplot() + geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
                      color = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') + xlab('Years of Experience') +
  ylab('Salary')

#Visualizing the test set results
library(ggplot2)
ggplot() + geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
                      color = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Test Set)') + xlab('Years of Experience') +
  ylab('Salary')
