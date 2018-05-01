# =======================================================
#    Allstate Claims Severity, Done by group 38
# =======================================================

#install.packages("xgboost")
#install.packages("mlr")
#install.packages("FeatureHashing")

library(mlr)
library(xgboost)
library(data.table)
library(parallelMap)
library(FeatureHashing)
library(BBmisc)


# Set the working directory
setwd("C:/Users/Asus-Laptop/Documents/ML Assignment")

# create xgboost learner for mlr package
makeRLearner.regr.xgboost.latest = function() {
  makeRLearnerRegr(
    cl = "regr.xgboost.latest", #class of regression
    package = "xgboost",
    par.set = makeParamSet(
      # makeDiscreteLearnerParam(id = "booster", default = "gbtree", values = c("gbtree", "gblinear", "dart")), #uncomment for feature importance
      makeNumericLearnerParam(id = "eta", default = 0.3, lower = 0, upper = 1),
      makeNumericLearnerParam(id = "obj_par", default = 2, lower = 0),
      makeIntegerLearnerParam(id = "max_depth", default = 6L, lower = 1L),
      makeNumericLearnerParam(id = "min_child_weight", default = 1, lower = 0),
      makeNumericLearnerParam(id = "subsample", default = 1, lower = 0, upper = 1),
      makeNumericLearnerParam(id = "colsample_bytree", default = 1, lower = 0, upper = 1),
      makeNumericLearnerParam(id = "lambda", default = 0, lower = 0),
      makeNumericLearnerParam(id = "alpha", default = 0, lower = 0),
      #uncomment the line below if you wish to find out best nrounds
      # makeIntegerLearnerParam(id = "early_stopping_rounds", default = 15L, lower = 1L, tunable = FALSE),
      makeNumericLearnerParam(id = "base_score", default = 0.5, tunable = FALSE),
      makeIntegerLearnerParam(id = "nthread", lower = 1L, tunable = FALSE),
      makeIntegerLearnerParam(id = "nrounds", default = 1L, lower = 1L),
      makeIntegerLearnerParam(id = "silent", default = 0L, lower = 0L, upper = 1L, tunable = FALSE),
      makeIntegerLearnerParam(id = "verbose", default = 1, lower = 0, upper = 2, tunable = FALSE),
      makeIntegerLearnerParam(id = "print_every_n", default = 1L, lower = 1L, tunable = FALSE, requires = quote(verbose == 1L))
    ),
    par.vals = list(nrounds = 1L, silent = 0L, verbose = 1L, obj_par = 2),
    properties = c("numerics", "factors", "weights"),
    name = "eXtreme Gradient Boosting",
    short.name = "xgboost",
    note = "All settings are passed directly, rather than through `xgboost`'s `params` argument. `nrounds` has been set to `1` and `verbose` to `0` by default."
  )
}

# create xgboost train and predict methods for mlr package
trainLearner.regr.xgboost.latest = function(.learner, .task, .subset, .weights = NULL,  ...) {
  data = getTaskData(.task, .subset, target.extra = TRUE)
  target = data$target
  #change features into vectors
  data = FeatureHashing::hashed.model.matrix( ~ . - 1, data$data)
  
  myobj = function(preds, dtrain, c) {
    labels = getinfo(dtrain, "label")
    x = preds-labels
    # introduce hyperparameter for objective function
    c = .learner$par.vals$obj_par
    grad = tanh(c*x)
    hess = c*sqrt(1-grad^2)
    return(list(grad = grad, hess = hess))
  }
  
  xgboost::xgboost(data = data, label = target, objective = myobj, ...)
}
predictLearner.regr.xgboost.latest = function(.learner, .model, .newdata, ...) {
  m = .model$learner.model
  data = FeatureHashing::hashed.model.matrix( ~ . - 1, .newdata)
  xgboost:::predict.xgb.Booster(m, newdata = data, ...)
}

train = fread("train.csv")
test = fread("test.csv")

# remove id
train[, id := NULL]
test[, id := NULL]

# transform target variable and use factor variables
train$loss = log(train$loss + 200)
test$loss = -99

# feature preprocess
dat = rbind(train, test)

char.feat = vlapply(dat, is.character)
char.feat = names(char.feat)[char.feat]

for (f in char.feat) {
  dat[[f]] = as.integer(as.factor(dat[[f]]))
}

dat = as.data.frame(dat)

# create task
train = dat[dat$loss != -99, ]
test = dat[dat$loss == -99, ]

# create mlr measure for log-transformed target
mae.log = mae
mae.log$fun = function (task, model, pred, feats, extra.args) {
  measureMAE(exp(pred$data$truth), exp(pred$data$response))
}

# create mlr train and test task
trainTask = makeRegrTask(data = as.data.frame(train), target = "loss")
testTask = makeRegrTask(data = as.data.frame(test), target = "loss")

# specify mlr learner with some nice hyperpars
set.seed(123)
lrn = makeLearner("regr.xgboost.latest")
# lrn = makeLearner("regr.xgboost") #use this if you want to get feature importance
lrn = setHyperPars(lrn,
                   base_score = 7.7,
                   subsample = 0.9990,
                   colsample_bytree = 0.45,
                   max_depth = 10,
                   lambda = 10,
                   min_child_weight = 2.5,
                   #uncomment the line below if you wish to find out best nrounds
                   # early_stopping_rounds = 5, 
                   alpha = 8,
                   nthread = 16,
                   nrounds = 800,
                   eta = 0.055,
                   print_every_n = 50
                   
)

## This is how you could do hyperparameter tuning with random search
# 1) Define the set of parameters you want to tune (here we use only 'obj_par')
ps = makeParamSet(
  makeNumericParam("obj_par", lower = 1.5, upper = 2)
#   makeNumericParam("eta", lower = 0.001, upper = 0.01),
# makeIntegerLearnerParam("max_depth", lower = 1, upper = 12 ),
# makeNumericLearnerParam("min_child_weight", lower = 0, upper = 5),
# makeNumericLearnerParam("subsample", lower = 0, upper = 1),
# makeNumericLearnerParam("colsample_bytree", lower = 0, upper = 1),
# makeNumericLearnerParam("lambda", lower = 0, upper = 12),
# makeNumericLearnerParam("alpha", lower = 0, upper = 12),
# makeNumericLearnerParam("base_score", lower = 1, upper = 12 ),
# makeIntegerLearnerParam("nthread", lower = 1, upper = 1000),
# makeIntegerLearnerParam("nrounds",lower = 1,upper = 1000),
# makeIntegerLearnerParam("silent", lower = 0L, upper = 1L),
# makeIntegerLearnerParam(id = "print_every_n", lower = 1, upper = 100)

)

# 2) Use 3-fold Cross-Validation to measure improvements
# rdesc = makeResampleDesc("CV", iters = 3)

# 3) Here we use random search (with 5 Iterations) to find the optimal hyperparameter
# ctrl =  makeTuneControlRandom(maxit = 5) 

# 4) now use the learner on the training Task with the 3-fold CV to optimize your set of parameters
# res = tuneParams(lrn, task = trainTask, resampling = rdesc,
# par.set = ps, control = ctrl, measures = mae.log)
# parallelStop()
# res$x

# 5) We fit model using the hyperparameter we found from 4)
set.seed(123)
lrn = setHyperPars(lrn, obj_par = 1.79)
mod = train(lrn, trainTask)

# 6) make Predtiction
pred = exp(getPredictionResponse(predict(mod, testTask))) - 200
summary(pred)
# getFeatureImportanceLearner(lrn,mod)
pred1 = exp(getPredictionResponse(predict(mod, trainTask))) - 200


submission = fread("sample_submission.csv", colClasses = c("integer", "numeric"))
submission$loss = pred
write.csv(submission, "kaggle_script.csv", row.names = FALSE)

#Get MAE
sqrt(mean((train$loss-pred1)^2))
mean(abs(train$loss-pred1))
# =======================================================
#                     Extra Experiments 
# =======================================================

# In order to see how well our above solution fare, we will be comparing it to
# other regression algorithms. 

# It is not mandatory to run the code below as it is just for comparison purposes.

# =======================================================
#                    Lasso Regression
# =======================================================

# library(Matrix)
# library(glmnet)
# 
# # load data
# train <- read.csv("train.csv",sep=",",header=T)
# test <- read.csv("test.csv",sep=",",header = T)
# 
# # separate target column
# y <- log(train[,"loss"]+200)
# train <- train[,-which(names(train)=="loss")]
# 
# # deal with factors (one-hot)
# full <- rbind(train,test)
# x.full <- sparse.model.matrix(~.,full)[,-1]
# x.train <- x.full[1:nrow(train),]
# x.test <- x.full[(nrow(train)+1):nrow(full),]
# rm(full,x.full)
# 
# # fit linear model with lasso penalty regularization
# linmod <- glmnet(x.train,y,
#                  family = "gaussian",
#                  alpha=1,
#                  lambda=exp(-8.383838))
# 
# # predict
# pred <- exp(predict(linmod,x.train))-200
# submission <- data.frame(id=test$id,loss=pred[,1])
# write.table(submission,file="submission.csv",sep=",",row.names=FALSE)

# =======================================================
#             H2O GBM and DEEP LEARNING MODEL
# =======================================================

# # Load libraries
# library(h2o)
# h2o.init(nthreads = 12)
# 
# # Read input data
# print("loading data")
# train <- h2o.importFile("../input/train.csv", destination_frame = "train.hex")
# test <- h2o.importFile("../input/test.csv", destination_frame = "test.hex")
# print("loading data - DONE")
# 
# print("splitting data")
# train <- train[, -1]
# train$loss <- log1p(train$loss)
# splits <- h2o.splitFrame(
#   data = train, 
#   ratios = c(0.8),
#   destination_frames = c("train.hex", "valid.hex"), seed = 1111
# )
# train <- splits[[1]]
# valid <- splits[[2]]
# print("splitting data - DONE")
# 
# submission <- test[, 1]
# test <- test[, -1]
# 
# features <- colnames(train)[-131]
# label <- "loss"
# 
# hyper_params = list( max_depth = seq(1,29,2) )
# print("train gbm")
# gbm_model <- h2o.gbm(features, label, training_frame = train, validation_frame = valid, ntrees=350,max_depth = 5)
# print("train gbm - DONE")
# 
# print("train deeplearning")
# dl_model <- h2o.deeplearning(features, label, training_frame = train, validation_frame = valid, model_id="dl_model_first", 
#                              #activation="Rectifier",  ## default
#                              #hidden=c(200,200),       ## default: 2 hidden layers with 200 neurons each
#                              epochs=50,
#                              variable_importances=T    ## not enabled by default
# )
# 
# print("train deeplearning - DONE")
# 
# print(h2o.mse(h2o.performance(gbm_model, valid = TRUE)))
# print(h2o.mse(h2o.performance(dl_model, valid = TRUE)))
# 
# submission$loss <- predict(gbm_model, newdata = test)
# 
# submission$loss <- expm1(submission$loss)
# 
# h2o.downloadCSV(submission, filename = "submission_h2o_gbm.csv")
# 
# submission$loss <- predict(dl_model, newdata = test)
# 
# submission$loss <- expm1(submission$loss)
# 
# h2o.downloadCSV(submission, filename = "submission_h2o_dl.csv")