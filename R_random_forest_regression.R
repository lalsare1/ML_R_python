#Decision Tree Regression Model
#Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#Create your regressor
install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1], y = dataset$Salary, ntree = 500)

#Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

#Visualizing the decision tree model
library(ggplot2)
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') + 
  geom_line(aes(x = X_grid, y = predict(regressor, newdata = data.frame(Level = X_grid))),
            color = 'blue') +
  ggtitle('Truth or Bluff (Random Forest Regression Model)') +
  xlab('Level')+
  ylab('Salary')