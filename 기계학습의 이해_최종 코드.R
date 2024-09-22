# Load the necessary libraries
install.packages("xgboost")
install.packages("e1071")
install.packages("randomForest")
install.packages("caret")
install.packages("catboost")
library(e1071)
library(randomForest)
library(caret)
library(xgboost)
library(glmnet)

library(lightgbm)
library(dplyr)

library(catboost)
library(readxl)
# Read in the data
data <- data <- read_xlsx('df2.xlsx')
data = as.matrix(data)
head(data)

# Set the windoww size and prediction horizon
windoww <- nrow(data)-100 
predict_ahead <- 1

# Initialize vectors to store predictions and errors

rmse.mat = NULL 
# Loop over the data using a rolling windoww approach
for (i in windoww:530) {
  print(i)
  
  ### ready 
  err.mod = NULL 
  
  ### partition 
  train_data <- data[1:i, ]
  head(train_data)
  test_data <- data[(i + 1):(i + predict_ahead),,drop=F]
  head(test_data)
  X_train <- train_data[,!(colnames(train_data)%in%c('????????','?????正?'))]
  y_train <- train_data[,colnames(train_data)=='????????']
  X_test <- test_data[,!(colnames(test_data)%in% c('????????','?????正?')),drop=F]
  y_test <- test_data[,colnames(test_data)=='????????',drop=F]
  
  ################# 1 glm 
  fit_glm = glm(y_train~.,data=data.frame(y_train,X_train))
  err.mod = c(err.mod,(y_test-predict.glm(fit_glm,newdata=data.frame(X_test)))^2)
  
  
  ###############  2 lasso 
  fit_lasso <- glmnet(X_train, y_train, alpha = 1)  
  e.mat = NULL 
  for(j in 1:30){
    ttrain_data <- data[1:(i-j), ]
    ttest_data <- data[(i-j+1):(i-j+predict_ahead),,drop=F]
    X_ttrain <- ttrain_data[,!(colnames(ttrain_data)%in%c('????????','?????正?'))]
    y_ttrain <- ttrain_data[,colnames(ttrain_data)=='????????']
    X_ttest <- ttest_data[,!(colnames(ttest_data)%in% c('????????','?????正?'))]
    y_ttest <- ttest_data[,colnames(ttest_data)=='????????']
    fitt_lasso <- glmnet(X_ttrain, y_ttrain, alpha = 1,lambda=fit_lasso$lambda)
    e.mat = rbind(e.mat,y_ttest-predict.glmnet(fitt_lasso,newx=X_ttest))
  }
  opt = which.min(colMeans(e.mat^2))
  err.mod = c(err.mod,(y_test-predict.glmnet(fit_lasso,newx=X_test)[,opt])^2)
  
  ############## 3 ridge 
  fit_ridge <- glmnet(X_train, y_train, alpha = 0)  
  e.mat = NULL 
  for(j in 1:30){
    ttrain_data <- data[1:(i-j), ]
    ttest_data <- data[(i-j+1):(i-j+predict_ahead),,drop=F]
    X_ttrain <- ttrain_data[,!(colnames(ttrain_data)%in%c('????????','?????正?'))]
    y_ttrain <- ttrain_data[,colnames(ttrain_data)=='????????']
    X_ttest <- ttest_data[,!(colnames(ttest_data)%in% c('????????','?????正?'))]
    y_ttest <- ttest_data[,colnames(ttest_data)=='????????']
    fitt_ridge <- glmnet(X_ttrain, y_ttrain, alpha = 0,lambda=fit_ridge$lambda)
    e.mat = rbind(e.mat,y_ttest-predict.glmnet(fitt_ridge,newx=X_ttest))
  }
  opt = which.min(colMeans(e.mat^2))
  err.mod = c(err.mod,(y_test-predict.glmnet(fit_ridge,newx=X_test)[,opt])^2)
  
  #########################  4 elastic net
  fit_elasticnet  <- glmnet(X_train, y_train, alpha = 0.01) 
  e.mat = NULL 
  for(j in 1:30){
    ttrain_data <- data[1:(i-j), ]
    ttest_data <- data[(i-j+1):(i-j+predict_ahead),,drop=F]
    X_ttrain <- ttrain_data[,!(colnames(ttrain_data)%in%c('????????'))]
    y_ttrain <- ttrain_data[,colnames(ttrain_data)=='????????']
    X_ttest <- ttest_data[,!(colnames(ttest_data)%in% c('????????'))]
    y_ttest <- ttest_data[,colnames(ttest_data)=='????????']
    fitt_elasticnet <- glmnet(X_ttrain, y_ttrain, alpha = 0.01,lambda=fit_elasticnet$lambda)
    e.mat = rbind(e.mat,y_ttest-predict.glmnet(fitt_elasticnet,newx=X_ttest))
  }
  
  opt = which.min(colMeans(e.mat^2))
  err.mod = c(err.mod,(y_test-predict.glmnet(fit_elasticnet,newx=X_test)[,opt])^2)
  
  
  ############## 5 SVM 
  svm_model <- svm(y_train ~ ., data=data.frame(X_train, y_train))
  predictions_svm <- predict(svm_model, newdata=data.frame(X_test))
  err.mod <- c(err.mod, (y_test - predictions_svm)^2)

  # svm_model <- svm(y_train ~ ., data=data.frame(X_train, y_train))
  # predictions_svm <- predict(svm_model, newdata=data.frame(X_test))
  # 
  # svm_tune_grid <- expand.grid(
  #   cost =  c(0.1, 1,5, 10),
  #   gamma= c(0.01, 0.1, 1)
  # )
  # 
  # tuned_results <- tune(svm, y_train ~ ., data = data.frame(X_train, y_train), 
  #                       ranges = svm_tune_grid)
  # 
  # best_parameters <- tuned_results$best.parameters
  # print(best_parameters)
  # 
  # svm_model_best <- svm(y_train ~ ., data=data.frame(X_train, y_train), 
  #                       cost=best_parameters$cost, gamma=best_parameters$gamma)
  # predictions_svm_best <- predict(svm_model_best, newdata=data.frame(X_test))
  # err.mod <- c(err.mod, (y_test - predictions_svm_best)^2)
  
  ################ 6  XGBoost 
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dtest <- xgb.DMatrix(data = X_test, label = y_test)
  params <- list(objective = "reg:squarederror", booster = "gbtree")
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100)
  predictions_xgb <- predict(xgb_model, newdata = dtest)
  err.mod <- c(err.mod, (y_test - predictions_xgb)^2)
  
  
  
  ############### 7  randomforest
  rf_model <- randomForest(y_train ~ ., data = data.frame(X_train, y_train), ntree = 500)
  tuneGrid <- expand.grid(
    mtry = c(2, 3, 4, 5)  # mtry ?? ??舛
  )
  trainControl <- trainControl(method="cv", number=5)
  tunedModel <- train(
    y_train ~ .,
    data = data.frame(X_train, y_train),
    method = "rf",
    tuneGrid = tuneGrid,
    trControl = trainControl
  )
  bestMtry <- tunedModel$bestTune$mtry
  rf_model <- randomForest(y_train ~ ., data = data.frame(X_train, y_train), mtry = bestMtry)
  predictions <- predict(rf_model, newdata = data.frame(X_test))
  err.mod <- c(err.mod, (y_test - predictions)^2)
  
  ################ 8 lightgbm

  param_grid <- expand.grid(
    max_depth = c(5, 10, 15),
    learning_rate = c(0.01, 0.05, 0.1),
    nrounds = c(100, 200, 300)
  )
  best_params <- NULL
  min_rmse <- Inf
  
  for (i in 1:nrow(param_grid)) {
    params <- list(
      objective = "regression",
      metric = "rmse",
      max_depth = param_grid$max_depth[i],
      learning_rate = param_grid$learning_rate[i],
      nrounds = param_grid$nrounds[i]
    )
    
    lgb_model <- lgb.train(
      params = params,
      data = lgb.Dataset(data = as.matrix(X_train), label = as.vector(y_train)),
      nrounds = params$nrounds,
      verbose = -1
    )
    
    predictions <- predict(lgb_model, as.matrix(X_test))
    
    rmse <- sqrt(mean((predictions - y_test)^2))
    
    if (rmse < min_rmse) {
      min_rmse <- rmse
      best_params <- params
    }
  }
  
  print(best_params)
  final_lgb_model <- lgb.train(
    params = best_params,
    data = lgb.Dataset(data = as.matrix(X_train), label = as.vector(y_train)),
    nrounds = best_params$nrounds,
    verbose = -1
  )
  predictions_lgb <- predict(final_lgb_model, as.matrix(X_test))
  err.mod <- c(err.mod, (y_test - predictions_lgb)^2)
  
  
  ########### final 
  rmse.mat = rbind(rmse.mat,err.mod)
}


sqrt(colMeans(rmse.mat))
boxplot(sqrt(rmse.mat),
        ylim = c(0, 2000), 
        names = c("GLM", "Lasso", "Ridge", "SVM", "XGBoost", "RF","lightGBM"),
        main = "Boxplot of Square Root of RMSE Values(converted_file_???????拿???)",
        xlab = "Models",
        ylab = "Square Root of RMSE")
