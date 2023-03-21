
.onLoad <- function(libname, pkgname) {
  library('dplyr')
  library('readr')
  library('randomForest')
  library('e1071')
  library('readxl')
  library('xgboost')
  library('mice')
  library('neuralnet')
  library('glmnet')
  library('VIM')
  library('car')
  library('Metrics')
  library('psych')
  library('rpart')
  library('rpart.plot')
  library('xgboost')
  library('caret')
  library('ggplot2')
  library('corrplot')
  library('lightgbm')
  library('automl')
  library('class')
  library('factoextra')
  library('adabag')
  library('autokeras')
  library("rgl")
}




.onLoad()


#' My function description
#'
#' @export


drop_na <- function(data, method){
  data <- apply(data, 2, as.numeric)
  if (method == 'omit'){
    data <- data %>% na.omit()
  }
  else if (method == 'mean'){
    for(i in 1:ncol(data)) {
      data[, i][is.na(data[, i])] <- mean(data[, i], na.rm = TRUE)
    }
  }
  else if (method == 'median'){
    for(i in 1:ncol(data)) {
      data[, i][is.na(data[, i])] <- median(data[, i], na.rm = TRUE)
    }
  }
  else if (method == 'mode'){
    for(i in 1:ncol(data)) {
      data[, i][is.na(data[, i])] <- mode(data[, i])
    }
  }
  else{
    data <- data %>% na.omit()
  }
  data <- data %>% as.data.frame()
  return(data)
} # method = c('omit', 'mean', 'median', 'mode)



#' My function description
#'
#' @export
data_selection <- function(data, pos_X, pos_y, method){
  data_X <- data %>% select(pos_X)
  data_y <- data %>% select(pos_y)
  data <- cbind(data_X, data_y) # last column is predicted
  for(i in 1:length(colnames(data))){
    colnames(data)[i] <- gsub(" ", "_", colnames(data)[i])
  }
  drop_na(data, method)
}


#' My function description
#'
#' @export

data_into_five <- function(data){
  divid_number <- nrow(data)
  divid_number_20 <- floor(divid_number / 5)
  first_4 <- seq(1, 5 * divid_number_20, divid_number_20)
  divid_5_position <- list(first_4[1]:(first_4[2]-1), first_4[2]:(first_4[3]-1),
                           first_4[3]:(first_4[4]-1), first_4[4]:(first_4[5]-1),
                           first_4[5]:nrow(data))
  return(divid_5_position)
}


#' My function description
#'
#' @export
coefficient_of_determination <- function(true, pred){
  rss <- sum((pred - mean(true)) ** 2)
  tss <- sum((true - mean(true)) ** 2)
  rsq <- 1 - (rss/tss)
  return(rsq)
}



#' My function description
#'
#' @export
dist_plot <- function(data){
  print("HIST PLOT; Factor Var ONLY")
  new_data = data2[,which(sapply(data2, is.factor))]
  for(i in 1:ncol(new_data)){
    print(ggplot(data = new_data, aes(x = factor(new_data[[i]]), fill = new_data[[i]])) +
            geom_bar() +
            theme_bw() +
            xlab(colnames(new_data)[i])+
            guides(fill = guide_legend(title = colnames(new_data)[i])))
  }
}


#' My function description
#'
#' @export
violin_plot <- function(data){
  print("VIOLIN PLOT; Numerical Var ONLY")
  data = apply(data, 2, as.numeric) %>% as.data.frame()
  for(i in 1:ncol(data))
  {
    print(ggplot(data, aes(x = data[,i], y = " ")) +
            xlab('Value') +
            ylab(colnames(data)[i]) +
            geom_violin() +
            stat_summary(fun = "mean",
                         geom = "crossbar",
                         width = 1,
                         aes(color = "Mean")) +
            stat_summary(fun = "median",
                         geom = "crossbar",
                         width = 1,
                         aes(color = "Median")) +
            scale_colour_manual(values = c("dark red", "blue"), # Colors
                                name = "" )+
            theme_light() +
            theme(text = element_text(size = 20)))
  }
}



#' My function description
#'
#' @export
visual_summary1 <- function(data){
  fake_data <- apply(data, 2, as.numeric)
  fake_data_get_nonnum <- which(apply(fake_data, 2, is.na)  == TRUE, arr.ind = TRUE)
  if(nrow(fake_data_get_nonnum) == 0){
    print(violin_plot(fake_data))
    corrplot(cor(fake_data), method = 'pie', type = "upper", order = "hclust",
             tl.col = "black", tl.srt = 45)
  } else {
    non_numerical_data <- data[,unique(fake_data_get_nonnum[,2])]
    numerical_data <- data[,-unique(fake_data_get_nonnum[,2])]
    print(dist_plot(non_numerical_data))
    print(violin_plot(numerical_data))
    corrplot(cor(numerical_data), method = 'pie', type = "upper", order = "hclust",
             tl.col = "black", tl.srt = 45)
  }
}



#' My function description
#'
#' @export
get_info <- function(data){
  set.seed(123)
  ord = sample(1:nrow(data), nrow(data), replace = FALSE)
  data = data[ord,]
  data %>% summary() %>% print()
  SELECT_VARIABLES <- colnames(data)
  excat_position <- is.na(data) %>% which(arr.ind = TRUE)
  mis_data1 = as.data.frame(table(colnames(data)[excat_position[,2]]))
  rownames(mis_data1) <- mis_data1$Var1
  data %>% glimpse()
  print("——————————————————————————————————————————————————————————————————————————————————————")
  print("Number of NA Values in Each Variable :")
  print(mis_data1 %>% select(-1))
  percent_mis_data = mis_data1 %>% select(-1) / nrow(data) * 100
  for (i in 1:nrow(percent_mis_data)){
    percent_mis_data[i,1] = paste0(round(as.numeric(percent_mis_data[i,1]), 4), '%')
  }
  percent_mis_data
  print("——————————————————————————————————————————————————————————————————————————————————————")
  print("Percentage of NA Values in Each Variable :")
  print(percent_mis_data)
  print(as.data.frame(SELECT_VARIABLES))
}





#######################################################
#' My function description
#'
#' @export
llm <- function(data, order){
  linear_m <- lm(unlist(data[ncol(data)])  %>% as.numeric() ~ ., data = data[,1:ncol(data) - 1])
  print("~")
  print('T test')
  linear_m %>% summary() %>% print()
  print('~')
  print('F test')
  linear_m %>% anova() %>% print()
  print('~')
  print('VIF MultiConlinerity')
  vif(linear_m) %>% print()
  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()

  for(i in 1:5){
    train_data = data[-unlist(order[i]),] %>% as.data.frame()
    test_data = data[unlist(order[i]),] %>% as.data.frame()
    mod1 = lm(unlist(train_data[ncol(train_data)])  %>% as.numeric() ~ ., data = train_data[,1:ncol(train_data) - 1])
    pred = predict(mod1, newdata = test_data[,1:ncol(test_data) - 1]) %>% as.numeric()
    true = unlist(test_data[ncol(test_data)]) %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))
  }
  paste('Pearson Coefficient Correlation :', pearson_cor)  %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('Coefficient of Determination :', r_2)  %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('RMSE : ', rmseee)  %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('MAE : ', maeee) %>% print()
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}

#######################################################
#' My function description
#'
#' @export
lss0 <- function(data, order){
  set.seed(123)
  y = data[,ncol(data)]
  X = data[,-ncol(data)] %>% as.matrix()
  mod2 <- cv.glmnet(x = X, y = y, alpha = 1)
  mod2 %>% plot()
  plot(mod2$glmnet.fit,
       "lambda", label = TRUE)
  paste('The Min Value of Lambda for Lasso Reg is :', mod2$lambda.min) %>% print()
  mod2 <- glmnet(X, y, alpha = 1, lambda = mod2$lambda.min)
  print(mod2$beta)

  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()

  for(i in 1:5){
    train_data = data[-unlist(order[i]),] %>% as.data.frame()
    test_data = data[unlist(order[i]),] %>% as.data.frame()
    train_data_X = train_data[,-ncol(train_data)] %>% as.matrix()
    train_data_y = train_data[,ncol(test_data)]
    test_data_X = test_data[,-ncol(test_data)] %>% as.matrix()
    test_data_y = test_data[,ncol(test_data)]
    mod2_cl = glmnet(train_data_X, train_data_y, alpha = 1, lambda = mod2$lambda)
    pred = predict(mod2_cl, newx = test_data_X) %>% as.numeric()
    true = unlist(test_data_y) %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))
  }
  paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('Coefficient of Determination :', r_2) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('RMSE : ', rmseee) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('MAE : ', maeee) %>% print()
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}

#lsso
#####################################################
#' My function description
#'
#' @export
ridgee <- function(data, order){
  set.seed(123)
  y = data[,ncol(data)]
  X = data[,-ncol(data)] %>% as.matrix()
  mod3 <- cv.glmnet(x = X, y = y, alpha = 0)
  mod3 %>% plot()
  plot(mod3$glmnet.fit,
       "lambda", label = TRUE)
  paste('The Min Value of Lambda for Redge Reg is :', mod3$lambda.min) %>% print()
  mod3 <- glmnet(X, y, alpha = 0, lambda = mod3$lambda.min)
  print(mod3$beta)

  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()

  for(i in 1:5){
    train_data = data[-unlist(order[i]),] %>% as.data.frame()
    test_data = data[unlist(order[i]),] %>% as.data.frame()
    train_data_X = train_data[,-ncol(train_data)] %>% as.matrix()
    train_data_y = train_data[,ncol(test_data)]
    test_data_X = test_data[,-ncol(test_data)] %>% as.matrix()
    test_data_y = test_data[,ncol(test_data)]
    mod3_cl = glmnet(train_data_X, train_data_y, alpha = 0, lambda = mod3$lambda)
    pred = predict(mod3_cl, newx = test_data_X) %>% as.numeric()
    true = unlist(test_data_y) %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))
  }
  paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('Coefficient of Determination :', r_2) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('RMSE : ', rmseee) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('MAE : ', maeee) %>% print()
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
} # ridge




#######################################################

#' My function description
#'
#' @export
rf_reg <- function(data, order){
  set.seed(123)
  oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)
  train_data = data[oooorder,] %>% as.data.frame()
  test_data = data[-oooorder,] %>% as.data.frame()
  xxx <- c()
  mtry_record = c()
  ntree_record = c()
  set_mtry = c(as.numeric(round(log(ncol(train_data)-1))), as.numeric(round(log(ncol(train_data)-1))) + 1, as.numeric(round(log(ncol(train_data)-1))) + 2)
  set_ntree = seq(400, 1000, 100)
  for(i in 1: length(set_mtry)){
    for (j in 1: length(set_ntree)){
      rf_find_best <- randomForest(unlist(train_data[ncol(train_data)]) %>% as.numeric() ~ .,
                                   data = train_data[,1:ncol(train_data) - 1], ntree = set_ntree[j],
                                   mtry = set_mtry[i])
      pred = as.numeric(predict(rf_find_best, newdata = test_data[,1:ncol(test_data) - 1]))
      r_2 <- coefficient_of_determination(unlist(test_data[ncol(test_data)]) %>% as.numeric(), pred)
      xxx <- c(xxx, r_2)
      mtry_record = c(mtry_record, rf_find_best$mtry)
      ntree_record = c(ntree_record, rf_find_best$ntree)
      which(xxx == max(xxx))
    }
  }
  paste('The optimal ntree is :', ntree_record[which(xxx == max(xxx))]) %>% print()
  paste('The optimal mtry is :', mtry_record[which(xxx == max(xxx))]) %>% print()

  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()
  for(i in 1:5){
    train_data = data[-unlist(order[i]),] %>% as.data.frame()
    test_data = data[unlist(order[i]),] %>% as.data.frame()
    mod4 = randomForest(unlist(train_data[ncol(train_data)])  %>% as.numeric() ~ ., data = train_data[,1:ncol(train_data) - 1],
                        ntree = ntree_record[which(xxx == max(xxx))], mtry =mtry_record[which(xxx == max(xxx))], importance = TRUE)
    varImpPlot(mod4)
    pred = predict(mod4, newdata = test_data[,1:ncol(test_data) - 1]) %>% as.numeric()
    true = unlist(test_data[ncol(test_data)]) %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))

    paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Coefficient of Determination :', r_2) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('RMSE : ', rmseee) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('MAE : ', maeee) %>% print()
  }
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}
# rf



##########################################################

#' My function description
#'
#' @export
xgboost123 <- function(data, order){
  pearson_cor = c()
  xxx = c()
  rmseee = c()
  maeee = c()
  depthh = c()
  eeta = c()
  lambdaa = c()
  mod5_data <- data %>% as.matrix()
  set.seed(123)
  oooorder = sample(1:nrow(mod5_data), floor(0.75 * nrow(mod5_data)), replace = FALSE)
  train_data = mod5_data[oooorder,] %>% as.matrix()
  test_data = mod5_data[-oooorder,] %>% as.matrix()
  for(i in seq(3,11,2)){
    for(j in c(0.05, 0.1, 0.15)){
      for(k in c(1,2)){
        mod5 <-
          xgboost(
            data = train_data[, 1: ncol(train_data)-1],
            label = train_data[, ncol(train_data)],
            nrounds = 10000,
            objective = "reg:squarederror",
            early_stopping_rounds = 2,
            max_depth = i,
            lambda = k,
            eta = j,
            eval_metric = 'rmse',
            subsample = 0.9
          )
        pred = predict(mod5, newdata = test_data[,1:ncol(test_data) - 1]) %>% as.matrix()
        true = unlist(test_data[,ncol(test_data)]) %>% as.numeric()
        pearson_cor <- c(pearson_cor, cor(pred, true))
        xxx <- c(xxx, coefficient_of_determination(pred, true))
        rmseee <- c(rmseee, rmse(true, pred))
        maeee <- c(maeee, mae(true, pred))
        eeta <- c(eeta, j) # eta
        lambdaa <- c(lambdaa, k) # lambda
        depthh <- c(depthh, i) # max_depth
      }
    }
  }

  paste('The optimal lambda is :', lambdaa[which(xxx == max(xxx))]) %>% print()
  paste('The optimal eta is :', eeta[which(xxx == max(xxx))]) %>% print()
  paste('The optimal max_depth is :', depthh[which(xxx == max(xxx))]) %>% print()


  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()

  for(i in 1:5){
    train_data = mod5_data[-unlist(order[i]),] %>% as.matrix()
    test_data = mod5_data[unlist(order[i]),] %>% as.matrix()
    mod5 <-
      xgboost(
        data = train_data[, 1: ncol(train_data)-1],
        label = train_data[, ncol(train_data)],
        nrounds = 10000,
        objective = "reg:squarederror",
        early_stopping_rounds = 2,
        max_depth = lambdaa[which(xxx == max(xxx))],
        lambda = lambdaa[which(xxx == max(xxx))],
        eta = eeta[which(xxx == max(xxx))],
        eval_metric = 'rmse',
        subsample = 0.9
      )
    pred = predict(mod5, newdata = test_data[,1:ncol(test_data) - 1]) %>% as.matrix()
    true = unlist(test_data[,ncol(test_data)]) %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))

    paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Coefficient of Determination :', r_2) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('RMSE : ', rmseee) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('MAE : ', maeee) %>% print()
  }
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}



############################################################

#' My function description
#'
#' @export


lightgbm123 <- function(data, order){
  pearson_cor = c()
  xxx = c()
  rmseee = c()
  maeee = c()
  depthh = c()
  lrr = c()
  num_leaves = c()
  feature_frac = c()
  bagging_frac = c()
  boost = c()

  for(i in c(1, 3, 4, 5, 6, 7)){ # depth
    for (j in c(0.05, 0.1, 0.15)){ # lr
      for (k in c(2,4,8,16,32)){ # num_leaves
        for (a in c(0.8, 1.0)){ #feature_frac
          for (c in c(0.8, 0.9, 1.0)){ #bagging_frac
            for (b in c('goss', 'gbdt')){ #boosting

              mod6_data <- data
              set.seed(123)
              oooorder = sample(1:nrow(mod6_data), floor(0.75 * nrow(mod6_data)), replace = FALSE)
              train_data = mod6_data[oooorder,] %>% as.matrix()
              test_data = mod6_data[-oooorder,] %>% as.matrix()
              lgb_train = lgb.Dataset(train_data[, 1: ncol(train_data)-1], label = train_data[, ncol(train_data)])
              lgb_test = lgb.Dataset(test_data[, 1: ncol(test_data)-1], label = test_data[, ncol(test_data)])

              params = list(
                objective = "regression"
                , metric = "l2"
                , learning_rate = j
                , num_leaves = k
                , max_depth = i
                , boosting = b
                , feature_fraction = a
                , bagging_fraction = c
              )

              valids = list(test = lgb_test)

              model = lgb.train(
                params = params
                , data = lgb_train
                , nrounds = 2000L
                , valids = valids
                , early_stopping_round = 100
              )

              lgb.get.eval.result(model, "test", "l2")
              pred = predict(model, test_data[, 1: ncol(test_data)-1])
              true = unlist(test_data[,ncol(test_data)]) %>% as.numeric()

              pearson_cor <- c(pearson_cor, cor(pred, true))
              xxx <- c(xxx, coefficient_of_determination(pred, true))
              rmseee <- c(rmseee, rmse(true, pred))
              maeee <- c(maeee, mae(true, pred))
              depthh = c(depthh, i)
              lrr = c(lrr, j)
              num_leaves = c(num_leaves, k)
              feature_frac = c(feature_frac, a)
              bagging_frac = c(bagging_frac, c)
              boost = c(boost, b)
            }
          }
        }
      }
    }
  }
  paste('The optimal depth is :', depthh[which(xxx == max(xxx))]) %>% print()
  paste('The optimal learning_rate is :', lrr[which(xxx == max(xxx))]) %>% print()
  paste('The optimal num_leaves is :', num_leaves[which(xxx == max(xxx))]) %>% print()
  paste('The optimal feature_frac is :', feature_frac[which(xxx == max(xxx))]) %>% print()
  paste('The optimal bagging_frac is :', bagging_frac[which(xxx == max(xxx))]) %>% print()
  paste('The optimal boosting is :', boost[which(xxx == max(xxx))]) %>% print()


  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()

  for(i in 1:5){
    train_data = mod6_data[-unlist(order[i]),] %>% as.matrix()
    test_data = mod6_data[unlist(order[i]),] %>% as.matrix()
    lgb_train = lgb.Dataset(train_data[, 1: ncol(train_data)-1], label = train_data[, ncol(train_data)])
    lgb_test = lgb.Dataset(test_data[, 1: ncol(test_data)-1], label = test_data[, ncol(test_data)])

    params = list(
      objective = "regression"
      , metric = "l2"
      , learning_rate = lrr[which(xxx == max(xxx))[1]]                             # which(xxx == max(xxx))[1]
      , num_leaves = num_leaves[which(xxx == max(xxx))[1]]
      , max_depth = depthh[which(xxx == max(xxx))[1]]
      , boosting = boost[which(xxx == max(xxx))[1]]
      , feature_fraction = feature_frac[which(xxx == max(xxx))[1]]
      , bagging_fraction = bagging_frac[which(xxx == max(xxx))[1]]
    )

    valids = list(test = lgb_test)

    model = lgb.train(
      params = params
      , data = lgb_train
      , nrounds = 2000L
      , valids = valids
      , early_stopping_round = 100
    )


    pred = predict(model, as.matrix(test_data)[,-ncol(test_data)])
    true = unlist(test_data[,ncol(test_data)]) %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))

    paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Coefficient of Determination :', r_2) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('RMSE : ', rmseee) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('MAE : ', maeee) %>% print()
  }
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}


####################################################
#automl_for_reg
#' My function description
#'
#' @export
automl_cv <- function(data, order){
  set.seed(123)
  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()
  for(i in 1:5){
    set.seed(123)

    print(i)
    train_data = data[-unlist(order[i]),] %>% as.data.frame()
    test_data = data[unlist(order[i]),] %>% as.data.frame()
    pred_name = colnames(train_data)[ncol(train_data)]

    train_file <- paste0(tempdir(), "/train.csv")
    write.csv(train_data, train_file, row.names = FALSE)

    test_file_to_eval <- paste0(tempdir(), "/eval.csv")
    write.csv(test_data, test_file_to_eval, row.names = FALSE)

    class <- model_structured_data_regressor(max_trials = 20) %>%
      fit(
        train_file,
        pred_name,
        validation_data = list(test_file_to_eval, pred_name)
      )

    pred <- class %>% predict(test_file_to_eval)

    true = test_data[,ncol(test_data)] %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(true, pred))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))

    paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Coefficient of Determination :', r_2) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('RMSE : ', rmseee) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('MAE : ', maeee) %>% print()
  }

  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}

################



#' My function description
#'
#' @export
all_reg_together <- function(data, pos_X, pos_y, method){
  data <- data_selection(data,  pos_X, pos_y, method)
  order = data_into_five(data)
  lm_reg <- llm(data, order)
  lsso_reg <- lss0(data, order)
  ridge_reg <- ridgee(data, order)
  rf_reg <- rf_reg(data, order)
  xgb_reg <- xgboost123(data, order)
  auto_reg <- automl_cv(data, order)
  light_reg <- lightgbm123(data, order)
  reg_print_r2 <- function(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg){
    a <- mean(lm_reg[1] %>% unlist())
    b <- mean(lsso_reg[1] %>% unlist())
    c <- mean(ridge_reg[1] %>% unlist())
    d <- mean(rf_reg[1] %>% unlist())
    e <- mean(xgb_reg[1] %>% unlist())
    f <- mean(auto_reg[1] %>% unlist())
    g <- mean(light_reg[1] %>% unlist())
    result <- data.frame(c(a,b,c,d,e,f,g))
    colnames(result) <- "r_square"
    rownames(result) <- c('lm', 'lasso', 'ridge', 'rf', 'xgboost', "automl", 'lightgbm')
    return(result)
  }
  reg_print_R2 <- function(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg){
    a <- mean(lm_reg[2] %>% unlist())
    b <- mean(lsso_reg[2] %>% unlist())
    c <- mean(ridge_reg[2] %>% unlist())
    d <- mean(rf_reg[2] %>% unlist())
    e <- mean(xgb_reg[2] %>% unlist())
    f <- mean(auto_reg[2] %>% unlist())
    g <- mean(light_reg[2] %>% unlist())
    result <- data.frame(c(a,b,c,d,e,f,g))
    colnames(result) <- "R_square"
    rownames(result) <- c('lm', 'lasso', 'ridge', 'rf', 'xgboost', "automl", 'lightgbm')
    return(result)
  }
  reg_print_rmse <- function(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg){
    a <- mean(lm_reg[3] %>% unlist())
    b <- mean(lsso_reg[3] %>% unlist())
    c <- mean(ridge_reg[3] %>% unlist())
    d <- mean(rf_reg[3] %>% unlist())
    e <- mean(xgb_reg[3] %>% unlist())
    f <- mean(auto_reg[3] %>% unlist())
    g <- mean(light_reg[3] %>% unlist())
    result <- data.frame(c(a,b,c,d,e,f,g))
    colnames(result) <- "rmse"
    rownames(result) <- c('lm', 'lasso', 'ridge', 'rf', 'xgboost', "automl", 'lightgbm')
    return(result)
  }
  reg_print_mae <- function(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg){
    a <- mean(lm_reg[4] %>% unlist())
    b <- mean(lsso_reg[4] %>% unlist())
    c <- mean(ridge_reg[4] %>% unlist())
    d <- mean(rf_reg[4] %>% unlist())
    e <- mean(xgb_reg[4] %>% unlist())
    f <- mean(auto_reg[4] %>% unlist())
    g <- mean(light_reg[4] %>% unlist())
    result <- data.frame(c(a,b,c,d,e,f,g))
    colnames(result) <- "mae"
    rownames(result) <- c('lm', 'lasso', 'ridge', 'rf', 'xgboost', "automl", 'lightgbm')
    return(result)
  }
  r2 <- reg_print_r2(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg, light_reg)
  R2 <- reg_print_R2(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg, light_reg)
  rmse <- reg_print_rmse(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg, light_reg)
  mae <- reg_print_mae(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg, light_reg)
  return((cbind(r2,R2,rmse,mae) %>% round(3)))
  # The above is to save and print the result
}

#################################################################
#' My function description
#'
#' @export
llm_rg_scale <- function(data, order, trans_method){
  linear_m <- lm(unlist(data[ncol(data)])  %>% as.numeric() ~ ., data = data[,1:ncol(data) - 1])
  print("~")
  print('T test')
  linear_m %>% summary() %>% print()
  print('~')
  print('F test')
  linear_m %>% anova() %>% print()
  print('~')
  print('VIF MultiConlinerity')
  vif(linear_m) %>% print()
  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()






  for(i in 1:5){

    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data)
    test_data = predict(trans_model, test_data)


    mod1 = lm(train_label  %>% as.numeric() ~ ., data = train_data[,1:ncol(train_data) - 1])
    mod1 %>% summary()
    pred = predict(mod1, newdata = test_data[,1:ncol(test_data) - 1]) %>% as.numeric()


    a = test_data[,1:ncol(test_data) - 1]


    true = unlist(test_data[ncol(test_data)]) %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))
  }
  paste('Pearson Coefficient Correlation :', pearson_cor)  %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('Coefficient of Determination :', r_2)  %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('RMSE : ', rmseee)  %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('MAE : ', maeee) %>% print()
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}

#################################################################
#' My function description
#'
#' @export
lss0_rg_scale <- function(data, order,  trans_method){
  set.seed(123)

  train_label = data[,ncol(data)]
  train_data = data[,-ncol(data)]

  trans_model = preProcess(train_data, method = trans_method)
  train_data = predict(trans_model, train_data) %>% as.matrix()

  mod2 <- cv.glmnet(x = train_data, y = train_label, alpha = 1)
  mod2 %>% plot()
  plot(mod2$glmnet.fit,
       "lambda", label = TRUE)
  paste('The Min Value of Lambda for Lasso Reg is :', mod2$lambda.min) %>% print()
  mod2 <- glmnet(train_data, train_label, alpha = 1, lambda = mod2$lambda.min)
  print(mod2$beta)

  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()

  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data)
    test_data = predict(trans_model, test_data)


    mod2_cl = glmnet(train_data, train_label, alpha = 1, lambda = mod2$lambda)
    pred = predict(mod2_cl, newx = as.matrix(test_data)) %>% as.numeric()
    true = unlist(test_label) %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))
  }
  paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('Coefficient of Determination :', r_2) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('RMSE : ', rmseee) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('MAE : ', maeee) %>% print()
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}


#################################################################
#' My function description
#'
#' @export
ridgee_rg_scale <- function(data, order,  trans_method){
  set.seed(123)
  train_label = data[,ncol(data)]
  train_data = data[,-ncol(data)]

  trans_model = preProcess(train_data, method = trans_method)
  train_data = predict(trans_model, train_data) %>% as.matrix()

  mod3 <- cv.glmnet(x = train_data, y = train_label, alpha = 0)
  mod3 %>% plot()
  plot(mod3$glmnet.fit,
       "lambda", label = TRUE)
  paste('The Min Value of Lambda for Redge Reg is :', mod3$lambda.min) %>% print()
  mod3 <- glmnet(train_data, train_label, alpha = 0, lambda = mod3$lambda.min)
  print(mod3$beta)

  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()

  for(i in 1:5){
    set.seed(123)

    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data) %>% as.matrix()
    test_data = predict(trans_model, test_data) %>% as.matrix()

    mod3_cl = glmnet(train_data, train_label, alpha = 0, lambda = mod3$lambda)
    pred = predict(mod3_cl, newx = test_data) %>% as.numeric()
    true = unlist(test_label) %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))
  }
  paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('Coefficient of Determination :', r_2) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('RMSE : ', rmseee) %>% print()
  print("      ")
  print(" __________________________ ")
  print("      ")
  paste('MAE : ', maeee) %>% print()
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}

#################################################################
#' My function description
#'
#' @export
rf_rg_scale <- function(data, order, trans_method){

  set.seed(123)
  oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)

  train_data = data[oooorder,]
  train_label = train_data[,ncol(train_data)]
  train_data = train_data[,-ncol(train_data)]
  test_data = data[-oooorder,]
  test_label = test_data[,ncol(test_data)]
  test_data = test_data[,-ncol(test_data)]
  trans_model = preProcess(train_data, method = trans_method)
  train_data = predict(trans_model, train_data)
  test_data = predict(trans_model, test_data)

  xxx <- c()
  mtry_record = c()
  ntree_record = c()
  set_mtry = c(as.numeric(round(log(ncol(train_data)-1))), as.numeric(round(log(ncol(train_data)-1))) + 1, as.numeric(round(log(ncol(train_data)-1))) + 2)
  set_ntree = seq(400, 1000, 100)
  for(i in 1: length(set_mtry)){
    for (j in 1: length(set_ntree)){
      rf_find_best <- randomForest(train_label %>% as.numeric() ~ .,
                                   data = train_data, ntree = set_ntree[j],
                                   mtry = set_mtry[i])
      pred = as.numeric(predict(rf_find_best, newdata = test_data))
      r_2 <- coefficient_of_determination(test_label %>% as.numeric(), pred)
      xxx <- c(xxx, r_2)
      mtry_record = c(mtry_record, rf_find_best$mtry)
      ntree_record = c(ntree_record, rf_find_best$ntree)
      which(xxx == max(xxx))
    }
  }
  paste('The optimal ntree is :', ntree_record[which(xxx == max(xxx))]) %>% print()
  paste('The optimal mtry is :', mtry_record[which(xxx == max(xxx))]) %>% print()

  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()
  for(i in 1:5){
    set.seed(123)

    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data) %>% as.matrix()
    test_data = predict(trans_model, test_data) %>% as.matrix()


    mod4 = randomForest(train_label  %>% as.numeric() ~ ., data = train_data,
                        ntree = ntree_record[which(xxx == max(xxx))], mtry =mtry_record[which(xxx == max(xxx))], importance = TRUE)
    varImpPlot(mod4)
    pred = predict(mod4, newdata = test_data)
    true = test_label  %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))

    paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Coefficient of Determination :', r_2) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('RMSE : ', rmseee) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('MAE : ', maeee) %>% print()
  }
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}


#################################################################
#' My function description
#'
#' @export
xgboost_rg_scale <- function(data, order, trans_method){
  pearson_cor = c()
  xxx = c()
  rmseee = c()
  maeee = c()
  depthh = c()
  eeta = c()
  lambdaa = c()
  mod5_data <- data %>% as.matrix()

  set.seed(123)
  oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)

  train_data = data[oooorder,]
  train_label = train_data[,ncol(train_data)]
  train_data = train_data[,-ncol(train_data)]
  test_data = data[-oooorder,]
  test_label = test_data[,ncol(test_data)]
  test_data = test_data[,-ncol(test_data)]
  trans_model = preProcess(train_data, method = trans_method)
  train_data = predict(trans_model, train_data) %>% as.matrix()
  test_data = predict(trans_model, test_data)  %>% as.matrix()

  for(i in seq(3,11,2)){
    for(j in c(0.05, 0.1, 0.15)){
      for(k in c(1,2)){
        mod5 <-
          xgboost(
            data = train_data,
            label = train_label,
            nrounds = 10000,
            objective = "reg:squarederror",
            early_stopping_rounds = 2,
            max_depth = i,
            lambda = k,
            eta = j,
            eval_metric = 'rmse',
            subsample = 0.9
          )
        pred = predict(mod5, newdata = test_data) %>% as.matrix()
        true = test_label %>% as.numeric()
        pearson_cor <- c(pearson_cor, cor(pred, true))
        xxx <- c(xxx, coefficient_of_determination(pred, true))
        rmseee <- c(rmseee, rmse(true, pred))
        maeee <- c(maeee, mae(true, pred))
        eeta <- c(eeta, j) # eta
        lambdaa <- c(lambdaa, k) # lambda
        depthh <- c(depthh, i) # max_depth
      }
    }
  }

  paste('The optimal lambda is :', lambdaa[which(xxx == max(xxx))]) %>% print()
  paste('The optimal eta is :', eeta[which(xxx == max(xxx))]) %>% print()
  paste('The optimal max_depth is :', depthh[which(xxx == max(xxx))]) %>% print()


  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()

  for(i in 1:5){
    set.seed(123)

    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data) %>% as.matrix()
    test_data = predict(trans_model, test_data) %>% as.matrix()

    mod5 <-
      xgboost(
        data = train_data,
        label = train_label,
        nrounds = 10000,
        objective = "reg:squarederror",
        early_stopping_rounds = 2,
        max_depth = lambdaa[which(xxx == max(xxx))],
        lambda = lambdaa[which(xxx == max(xxx))],
        eta = eeta[which(xxx == max(xxx))],
        eval_metric = 'rmse',
        subsample = 0.9
      )
    pred = predict(mod5, newdata = test_data) %>% as.matrix()
    true = unlist(test_label) %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))

    paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Coefficient of Determination :', r_2) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('RMSE : ', rmseee) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('MAE : ', maeee) %>% print()
  }
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}

#################################################################
#' My function description
#'
#' @export
lightgbm_rg_scale <- function(data, order, trans_method){
  pearson_cor = c()
  xxx = c()
  rmseee = c()
  maeee = c()
  depthh = c()
  lrr = c()
  num_leaves = c()
  feature_frac = c()
  bagging_frac = c()
  boost = c()

  for(i in c(1, 3, 4, 5, 6, 7)){ # depth
    for (j in c(0.05, 0.1, 0.15)){ # lr
      for (k in c(2,4,8,16,32)){ # num_leaves
        for (a in c(0.8, 1.0)){ #feature_frac
          for (c in c(0.8, 0.9, 1.0)){ #bagging_frac
            for (b in c('goss', 'gbdt')){ #boosting

              set.seed(123)
              oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)

              train_data = data[oooorder,]
              train_label = train_data[,ncol(train_data)]
              train_data = train_data[,-ncol(train_data)]
              test_data = data[-oooorder,]
              test_label = test_data[,ncol(test_data)]
              test_data = test_data[,-ncol(test_data)]
              trans_model = preProcess(train_data, method = trans_method)
              train_data = predict(trans_model, train_data) %>% as.matrix()
              test_data = predict(trans_model, test_data)  %>% as.matrix()

              lgb_train = lgb.Dataset(train_data, label = train_label)
              lgb_test = lgb.Dataset(test_data, label = test_label)

              params = list(
                objective = "regression"
                , metric = "l2"
                , learning_rate = j
                , num_leaves = k
                , max_depth = i
                , boosting = b
                , feature_fraction = a
                , bagging_fraction = c
              )

              valids = list(test = lgb_test)

              model = lgb.train(
                params = params
                , data = lgb_train
                , nrounds = 2000L
                , valids = valids
                , early_stopping_round = 100
              )

              lgb.get.eval.result(model, "test", "l2")
              pred = predict(model, test_data)
              true = test_label %>% as.numeric()

              pearson_cor <- c(pearson_cor, cor(pred, true))
              xxx <- c(xxx, coefficient_of_determination(pred, true))
              rmseee <- c(rmseee, rmse(true, pred))
              maeee <- c(maeee, mae(true, pred))
              depthh = c(depthh, i)
              lrr = c(lrr, j)
              num_leaves = c(num_leaves, k)
              feature_frac = c(feature_frac, a)
              bagging_frac = c(bagging_frac, c)
              boost = c(boost, b)
            }
          }
        }
      }
    }
  }
  paste('The optimal depth is :', depthh[which(xxx == max(xxx))]) %>% print()
  paste('The optimal learning_rate is :', lrr[which(xxx == max(xxx))]) %>% print()
  paste('The optimal num_leaves is :', num_leaves[which(xxx == max(xxx))]) %>% print()
  paste('The optimal feature_frac is :', feature_frac[which(xxx == max(xxx))]) %>% print()
  paste('The optimal bagging_frac is :', bagging_frac[which(xxx == max(xxx))]) %>% print()
  paste('The optimal boosting is :', boost[which(xxx == max(xxx))]) %>% print()


  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()

  for(i in 1:5){
    set.seed(123)

    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data) %>% as.matrix()
    test_data = predict(trans_model, test_data) %>% as.matrix()

    lgb_train = lgb.Dataset(train_data, label = train_label)
    lgb_test = lgb.Dataset(test_data, label = test_label)

    params = list(
      objective = "regression"
      , metric = "l2"
      , learning_rate = lrr[which(xxx == max(xxx))[1]]                             # which(xxx == max(xxx))[1]
      , num_leaves = num_leaves[which(xxx == max(xxx))[1]]
      , max_depth = depthh[which(xxx == max(xxx))[1]]
      , boosting = boost[which(xxx == max(xxx))[1]]
      , feature_fraction = feature_frac[which(xxx == max(xxx))[1]]
      , bagging_fraction = bagging_frac[which(xxx == max(xxx))[1]]
    )

    valids = list(test = lgb_test)

    model = lgb.train(
      params = params
      , data = lgb_train
      , nrounds = 2000L
      , valids = valids
      , early_stopping_round = 100
    )


    pred = predict(model, as.matrix(test_data))
    true = test_label %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(pred, true))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))

    paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Coefficient of Determination :', r_2) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('RMSE : ', rmseee) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('MAE : ', maeee) %>% print()
  }
  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}

#################################################################
#' My function description
#'
#' @export
automl_rg_scale <- function(data, order, trans_method){

  pearson_cor = c()
  r_2 = c()
  rmseee = c()
  maeee = c()

  set.seed(123)
  oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)

  train_data = data[oooorder,]
  train_label = train_data[,ncol(train_data)]
  train_data = train_data[,-ncol(train_data)]
  test_data = data[-oooorder,]
  test_label = test_data[,ncol(test_data)]
  test_data = test_data[,-ncol(test_data)]
  trans_model = preProcess(train_data, method = trans_method)
  train_data = predict(trans_model, train_data) %>% as.matrix()
  test_data = predict(trans_model, test_data)  %>% as.matrix()
  for(i in 1:5){
    set.seed(123)

    print(i)
    train_data = data[-unlist(order[i]),] %>% as.data.frame()
    test_data = data[unlist(order[i]),] %>% as.data.frame()
    pred_name = colnames(train_data)[ncol(train_data)]

    train_file <- paste0(tempdir(), "/train.csv")
    write.csv(train_data, train_file, row.names = FALSE)

    test_file_to_eval <- paste0(tempdir(), "/eval.csv")
    write.csv(test_data, test_file_to_eval, row.names = FALSE)

    class <- model_structured_data_regressor(max_trials = 20) %>%
      fit(
        train_file,
        pred_name,
        validation_data = list(test_file_to_eval, pred_name)
      )

    pred <- class %>% predict(test_file_to_eval)

    true = test_data[,ncol(test_data)] %>% as.numeric()
    pearson_cor <- c(pearson_cor, cor(pred, true))
    r_2 <- c(r_2, coefficient_of_determination(true, pred))
    rmseee <- c(rmseee, rmse(true, pred))
    maeee <- c(maeee, mae(true, pred))

    paste('Pearson Coefficient Correlation :', pearson_cor) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Coefficient of Determination :', r_2) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('RMSE : ', rmseee) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('MAE : ', maeee) %>% print()
  }

  overall <- list(pearson_cor, r_2, rmseee, maeee)
  return(overall)
}



#######################################################################W##


all_reg_together_scaled <- function(data, pos_X, pos_y, method, trans_method){
  data <- data_selection(data, pos_X, pos_y, method)
  order = data_into_five(data)
  lm_reg <- llm_rg_scale(data, order, trans_method)
  lsso_reg <- lss0_rg_scale(data, order, trans_method)
  ridge_reg <- ridgee_rg_scale(data, order, trans_method)
  rf_reg <- rf_rg_scale(data, order, trans_method)
  xgb_reg <- xgboost_rg_scale(data, order, trans_method)
  auto_reg <- automl_rg_scale(data, order, trans_method)
  light_reg <- lightgbm_rg_scale(data, order, trans_method)
  reg_print_r2 <- function(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg){
    a <- mean(lm_reg[1] %>% unlist())
    b <- mean(lsso_reg[1] %>% unlist())
    c <- mean(ridge_reg[1] %>% unlist())
    d <- mean(rf_reg[1] %>% unlist())
    e <- mean(xgb_reg[1] %>% unlist())
    f <- mean(auto_reg[1] %>% unlist())
    g <- mean(light_reg[1] %>% unlist())
    result <- data.frame(c(a,b,c,d,e,f,g))
    colnames(result) <- "r_square"
    rownames(result) <- c('lm', 'lasso', 'ridge', 'rf', 'xgboost', "automl", 'lightgbm')
    return(result)
  }
  reg_print_R2 <- function(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg){
    a <- mean(lm_reg[2] %>% unlist())
    b <- mean(lsso_reg[2] %>% unlist())
    c <- mean(ridge_reg[2] %>% unlist())
    d <- mean(rf_reg[2] %>% unlist())
    e <- mean(xgb_reg[2] %>% unlist())
    f <- mean(auto_reg[2] %>% unlist())
    g <- mean(light_reg[2] %>% unlist())
    result <- data.frame(c(a,b,c,d,e,f,g))
    colnames(result) <- "R_square"
    rownames(result) <- c('lm', 'lasso', 'ridge', 'rf', 'xgboost', "automl", 'lightgbm')
    return(result)
  }
  reg_print_rmse <- function(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg){
    a <- mean(lm_reg[3] %>% unlist())
    b <- mean(lsso_reg[3] %>% unlist())
    c <- mean(ridge_reg[3] %>% unlist())
    d <- mean(rf_reg[3] %>% unlist())
    e <- mean(xgb_reg[3] %>% unlist())
    f <- mean(auto_reg[3] %>% unlist())
    g <- mean(light_reg[3] %>% unlist())
    result <- data.frame(c(a,b,c,d,e,f,g))
    colnames(result) <- "rmse"
    rownames(result) <- c('lm', 'lasso', 'ridge', 'rf', 'xgboost', "automl", 'lightgbm')
    return(result)
  }
  reg_print_mae <- function(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg){
    a <- mean(lm_reg[4] %>% unlist())
    b <- mean(lsso_reg[4] %>% unlist())
    c <- mean(ridge_reg[4] %>% unlist())
    d <- mean(rf_reg[4] %>% unlist())
    e <- mean(xgb_reg[4] %>% unlist())
    f <- mean(auto_reg[4] %>% unlist())
    g <- mean(light_reg[4] %>% unlist())
    result <- data.frame(c(a,b,c,d,e,f,g))
    colnames(result) <- "mae"
    rownames(result) <- c('lm', 'lasso', 'ridge', 'rf', 'xgboost', "automl", 'lightgbm')
    return(result)
  }
  r2 <- reg_print_r2(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg)
  R2 <- reg_print_R2(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg)
  rmse <- reg_print_rmse(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg)
  mae <- reg_print_mae(lm_reg,lsso_reg,ridge_reg,rf_reg,xgb_reg,auto_reg,light_reg)
  return((cbind(r2,R2,rmse,mae) %>% round(3)))
  # The above is to save and print the result
}





#####






# try to use mtcars to be the classification example
# remeber cyl as y
# rf, xgboost, adaboost, svm, knn, naivebayes, done






#' My function description
#'
#' @export

kme_pca <- function(data){
  pca <- prcomp(data[1:(ncol(data)-1)], scale = TRUE)
  pc1 <- pca$x[,1]
  pc2 <- pca$x[,2]
  pc3 <- pca$x[,3]
  tot = cbind(pc1, pc2, pc3)

  set.seed(123)
  kmeans_fit <- kmeans(tot, centers = length(unique(data[,ncol(data)])))
  cluster_assignments <- kmeans_fit$cluster
  cluster_centers <- kmeans_fit$centers

  tot = cbind(pc1, pc2, pc3) %>% as.data.frame()

  plot3d(tot$pc1, tot$pc2, tot$pc3,
         col = kmeans_fit$cluster, size = 2, type = "s",
         main = "K-means Clustering (k = 3) Based on First Three PCs ")

  widget <- rglwidget()
  widget


  plot3d(tot$pc1, tot$pc2, tot$pc3,
       col = data[,ncol(data)], size = 2, type = "s",
       main = "True Value Based on First Three PCs ")
  widget <- rglwidget()
  widget



}


####################################################################
#' My function description
#'
#' @export
class_KNN <- function(data, order){
  lv = levels(as.factor(unlist(data[ncol(data)])))
  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[i]),] %>% as.data.frame()
    test_data = data[unlist(order[i]),] %>% as.data.frame()
    pred = knn(train_data[,-ncol(train_data)], test_data[,-ncol(test_data)],
               cl = train_data[,ncol(train_data)], k = length(unique(train_data[,ncol(train_data)])))
    true = unlist(test_data[ncol(test_data)]) %>% as.numeric()

    confusion_matrix <- confusionMatrix(factor(pred, levels = lv), factor(true, levels = lv))
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('KNN Accuracy:', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('KNN p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('KNN Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}

##############################################
#' My function description
#'
#' @export
class_naive <- function(data, order){
  lv = levels(as.factor(unlist(data[ncol(data)])))
  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[5]),] %>% as.data.frame()
    test_data = data[unlist(order[5]),] %>% as.data.frame()
    modd <- naiveBayes(unlist(train_data[ncol(train_data)]) ~ ., data = train_data[,1:ncol(train_data) - 1])
    pred = predict(modd, newdata = test_data[,1:ncol(test_data) - 1])
    true = unlist(test_data[ncol(test_data)]) %>% as.numeric()

    confusion_matrix <- confusionMatrix(factor(pred, levels = lv), factor(true, levels = lv))
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('Naive Bayes Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Naive Bayes p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Naive Bayes Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}

##############################################
#' My function description
#'
#' @export
class_svm <- function(data, order){
  lv = levels(as.factor(unlist(data[ncol(data)])))
  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[i]),] %>% as.data.frame()
    test_data = data[unlist(order[i]),] %>% as.data.frame()
    modd = svm(unlist(train_data[ncol(train_data)])  %>% as.numeric() ~ ., data = train_data[,1:ncol(train_data) - 1],
               type = 'C-classification', kernel = 'sigmoid')
    pred = predict(modd, newdata = test_data[,1:ncol(test_data) - 1])
    true = unlist(test_data[ncol(test_data)]) %>% as.numeric()

    confusion_matrix <- confusionMatrix(factor(pred, levels = lv), factor(true, levels = lv))
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('SVM Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('SVM p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('SVM Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}


#########################################
#' My function description
#'
#' @export
class_rf <- function(data, order){
  set.seed(123)
  oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)
  train_data = data[oooorder,] %>% as.data.frame()
  test_data = data[-oooorder,] %>% as.data.frame()
  lv = levels(as.factor(unlist(data[ncol(data)])))
  xxx <- c()
  mtry_record = c()
  ntree_record = c()
  set_mtry = c(as.numeric(round(log(ncol(train_data)-1))), as.numeric(round(log(ncol(train_data)-1))) + 1, as.numeric(round(log(ncol(train_data)-1))) + 2)
  set_ntree = seq(400, 1000, 100)
  for(i in 1: length(set_mtry)){
    for (j in 1: length(set_ntree)){
      rf_find_best <- randomForest(as.factor(unlist(train_data[ncol(train_data)]) %>% as.numeric()) ~ .,
                                   data = train_data[,1:ncol(train_data) - 1], ntree = set_ntree[j],
                                   mtry = set_mtry[i])
      pred = predict(rf_find_best, newdata = test_data[,1:ncol(test_data) - 1])
      acc <- confusionMatrix(pred, factor(test_data[,ncol(test_data)], levels = lv), mode = 'prec_recall')$overall[1]
      xxx <- c(xxx, acc)
      mtry_record = c(mtry_record, rf_find_best$mtry)
      ntree_record = c(ntree_record, rf_find_best$ntree)
      which(xxx == max(xxx))
    }
  }
  paste('The optimal RF ntree is :', ntree_record[which(xxx == max(xxx))]) %>% print()
  paste('The optimal RF mtry is :', mtry_record[which(xxx == max(xxx))]) %>% print()

  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[i]),] %>% as.data.frame()
    test_data = data[unlist(order[i]),] %>% as.data.frame()
    modd = randomForest(as.factor(unlist(train_data[ncol(train_data)])  %>% as.numeric()) ~ ., data = train_data[,1:ncol(train_data) - 1],
                        ntree = ntree_record[which(xxx == max(xxx))][1], mtry =mtry_record[which(xxx == max(xxx))][1], importance = TRUE)
    varImpPlot(modd)
    pred = predict(modd, newdata = test_data[,1:ncol(test_data) - 1])
    true = unlist(test_data[ncol(test_data)]) %>% as.numeric()

    confusion_matrix <- confusionMatrix(factor(pred, levels = lv), factor(true, levels = lv))
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('Random Forest Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Random Forest p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Random Forest Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}

######################################################

# change the class to zero
#' My function description
#'
#' @export

xgboost123 <- function(data, order){
  acc = c()
  xxx = c()
  depthh = c()
  eeta = c()
  lambdaa = c()
  xgb_data <- data %>% as.matrix()
  set.seed(123)
  label = data[,ncol(data)]
  for(i in 1:length(unique(label))){
    a = i - 1
    b = label == sort(unique(label))[i]
    data[,ncol(data)][which(b == TRUE)] <- a
  }
  oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)
  train_data = data[oooorder,] %>% as.matrix()
  test_data = data[-oooorder,] %>% as.matrix()
  lv = levels(as.factor(unlist(data[ncol(data)])))
  for(i in seq(3,11,2)){
    for(j in c(0.05, 0.1, 0.15)){
      for(k in c(1,2)){
        mod_xg <-
          xgboost(
            data = train_data[, 1: ncol(train_data)-1],
            label = train_data[, ncol(train_data)] %>% as.numeric(),
            nrounds = 10000,
            objective = "multi:softprob",
            num_class = length(unique(label)),
            early_stopping_rounds = 2,
            max_depth = i,
            lambda = k,
            eta = j,
            subsample = 0.9
          )
        pred = predict(mod_xg, newdata = test_data[,1:ncol(test_data) - 1], reshape = TRUE)
        pred = as.data.frame(pred)
        colnames(pred) = sort(unique(train_data[, ncol(train_data)]))
        pred = apply(pred,1,function(x) colnames(pred)[which.max(x)])

        acc <- confusionMatrix(factor(as.numeric(test_data[, ncol(test_data)]), levels = lv), factor(pred, levels = lv), mode = 'prec_recall')$overall[1]
        xxx <- c(xxx, acc)
        eeta <- c(eeta, j) # eta
        lambdaa <- c(lambdaa, k) # lambda
        depthh <- c(depthh, i) # max_depth
      }
    }
  }

  paste('The optimal xgboost lambda is :', lambdaa[which(xxx == max(xxx))]) %>% print()
  paste('The optimal xgboost eta is :', eeta[which(xxx == max(xxx))]) %>% print()
  paste('The optimal xgboost max_depth is :', depthh[which(xxx == max(xxx))]) %>% print()

  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()

  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[i]),] %>% as.matrix()
    test_data = data[unlist(order[i]),] %>% as.matrix()
    mod_xg <-
      xgboost(
        data = train_data[, 1: ncol(train_data)-1],
        label = train_data[, ncol(train_data)] %>% as.numeric(),
        nrounds = 10000,
        objective = "multi:softprob",
        num_class = 6,
        early_stopping_rounds = 2,
        max_depth = lambdaa[which(xxx == max(xxx))],
        lambda = lambdaa[which(xxx == max(xxx))],
        eta = eeta[which(xxx == max(xxx))],
        subsample = 0.9
      )
    pred = predict(mod_xg, newdata = test_data[,1:ncol(test_data) - 1], reshape = TRUE)
    pred = as.data.frame(pred)
    colnames(pred) = sort(unique(train_data[, ncol(train_data)]))
    pred = apply(pred,1,function(x) colnames(pred)[which.max(x)])
    confusion_matrix <- confusionMatrix(factor(as.numeric(test_data[, ncol(test_data)]), levels = lv), factor(pred, levels = lv), mode = 'prec_recall')
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('Random Forest Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Random Forest p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Random Forest Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}

################################################################################################
# class adaboost

#' My function description
#'
#' @export
class_adabst <- function(data, order){
  set.seed(123)
  oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)
  train_data = data[oooorder,] %>% as.data.frame()
  test_data = data[-oooorder,] %>% as.data.frame()
  lv = levels(as.factor(unlist(data[ncol(data)])))
  coeflearn <- c('Breiman', 'Zhu', 'Freund')
  xxx <- c()
  for (i in coeflearn){
    train_data = cbind(train_data, train_data[, ncol(train_data)])
    train_data <- train_data %>% select(-(ncol(train_data) - 1))
    train_data[, ncol(train_data)] <- as.factor(train_data[, ncol(train_data)])
    model = boosting(`train_data[, ncol(train_data)]` ~., data = train_data, boos = TRUE, mfinal = 500, coeflearn = i)
    pred = predict(model, newdata = test_data[,1:ncol(test_data) - 1])
    acc <- confusionMatrix(factor(pred$class, levels = lv), factor(test_data[,ncol(test_data)], levels = lv), mode = 'prec_recall')$overall[1]
    xxx <- c(xxx, acc)
  }
  paste('The Optimal Coeflearn is :', coeflearn[which(xxx == max(xxx))]) %>% print()

  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[5]),] %>% as.data.frame()
    test_data = data[unlist(order[5]),] %>% as.data.frame()
    train_data = cbind(train_data, train_data[, ncol(train_data)])
    train_data <- train_data %>% select(-(ncol(train_data) - 1))
    train_data[, ncol(train_data)] <- as.factor(train_data[, ncol(train_data)])

    modd = boosting(`train_data[, ncol(train_data)]` ~., data = train_data, boos = TRUE, mfinal = 500, coeflearn = coeflearn[which(xxx == max(xxx))][1])
    pred = predict(modd, newdata = test_data[,1:ncol(test_data) - 1])
    confusion_matrix <- confusionMatrix(factor(pred$class, levels = lv), factor(test_data[,ncol(test_data)], levels = lv), mode = 'prec_recall')
    acc <- confusion_matrix$overall[1]
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('Adaboost Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Adaboost p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Adaboost Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}

###############################################################
#tree_stat
#' My function description
#'
#' @export
tree_stat2 <- function(data){
  r_tree_stat <- rpart(unlist(data[ncol(data)])  %>% as.numeric() ~ ., data = data[,1:ncol(data) - 1], method = 'class')
  r_tree_stat %>% printcp()
  r_tree_stat %>% summary()
  rpart.plot::rpart.plot(r_tree_stat)
}
# tree
##########################################################

#' My function description
#'
#' @export
keras_class <- function(data, order){
  lv = levels(as.factor(unlist(data[ncol(data)])))
  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[i]),] %>% as.data.frame()
    test_data = data[unlist(order[i]),] %>% as.data.frame()
    pred_name = colnames(train_data)[ncol(train_data)]

    train_file <- paste0(tempdir(), "/train.csv")
    write.csv(train_data, train_file, row.names = FALSE)

    test_file_to_eval <- paste0(tempdir(), "/eval.csv")
    write.csv(test_data, test_file_to_eval, row.names = FALSE)

    class <- model_structured_data_classifier(max_trials = 20) %>%
      fit(
        train_file,
        pred_name,
        validation_data = list(test_file_to_eval, pred_name)
      )
    pred <- class %>% predict(test_file_to_eval) %>% as.factor()
    true = unlist(test_data[ncol(test_data)]) %>% as.numeric() %>% as.factor()

    confusion_matrix <- confusionMatrix(factor(pred, levels = lv), factor(true, levels = lv))
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('Keras_Class Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Keras_Class p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Keras_Class Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}



#######################

#' My function description
#'
#' @export
print_result <- function(KNN_result, naive_result, svm_result, rf_result, xgb_result, adab_result, keras_result ){
  # build a empty data frame to save the result
  acc_and_p <- matrix(ncol = 2, nrow = 0)
  colnames(acc_and_p) <- c('Accuracy', 'p-value')
  acc_and_p <- rbind(acc_and_p, unlist(KNN_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(naive_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(svm_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(rf_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(xgb_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(adab_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(keras_result[4]))

  colnames(acc_and_p) <- c('Precision', 'p-value')
  rownames(acc_and_p) <- c('KNN', 'Naive_Bayes',
                           'SVM', 'Random Forest',
                           'XGBoost', 'Adaboost', 'AutoML')
  print(acc_and_p)



  # build f1
  f1_value <- matrix(ncol = length(unlist(KNN_result[3])), nrow = 0)

  f1_value <- rbind(f1_value, unlist(KNN_result[3]))
  f1_value <- rbind(f1_value, unlist(naive_result[3]))
  f1_value <- rbind(f1_value, unlist(svm_result[3]))
  f1_value <- rbind(f1_value, unlist(rf_result[3]))
  f1_value <- rbind(f1_value, unlist(xgb_result[3]))
  f1_value <- rbind(f1_value, unlist(adab_result[3]))
  f1_value <- rbind(f1_value, unlist(keras_result[3]))
  rownames(f1_value) <- c('KNN', 'Naive_Bayes',
                          'SVM', 'Random Forest',
                          'XGBoost', 'Adaboost', 'AutoML')

  print(f1_value)


  # build precision
  precision_value <- matrix(ncol = length(unlist(KNN_result[1])), nrow = 0)

  precision_value <- rbind(precision_value, unlist(KNN_result[1]))
  precision_value <- rbind(precision_value, unlist(naive_result[1]))
  precision_value <- rbind(precision_value, unlist(svm_result[1]))
  precision_value <- rbind(precision_value, unlist(rf_result[1]))
  precision_value <- rbind(precision_value, unlist(xgb_result[1]))
  precision_value <- rbind(precision_value, unlist(adab_result[1]))
  precision_value <- rbind(precision_value, unlist(keras_result[1]))
  rownames(precision_value) <- c('KNN', 'Naive_Bayes',
                                 'SVM', 'Random Forest',
                                 'XGBoost', 'Adaboost', 'AutoML')
  print(precision_value)


  # build recall
  recall_value <- matrix(ncol = length(unlist(KNN_result[2])), nrow = 0)

  recall_value <- rbind(recall_value, unlist(KNN_result[2]))
  recall_value <- rbind(recall_value, unlist(naive_result[2]))
  recall_value <- rbind(recall_value, unlist(svm_result[2]))
  recall_value <- rbind(recall_value, unlist(rf_result[2]))
  recall_value <- rbind(recall_value, unlist(xgb_result[2]))
  recall_value <- rbind(recall_value, unlist(adab_result[2]))
  recall_value <- rbind(recall_value, unlist(keras_result[2]))
  rownames(recall_value) <- c('KNN', 'Naive_Bayes',
                              'SVM', 'Random Forest',
                              'XGBoost', 'Adaboost', 'AutoML')
  print(recall_value)

  scx <- list(acc_and_p, f1_value, precision_value, recall_value)
  names(scx) <- c("Accuracy & p-value", "F1-Value", "Precision", "Recall")
  return(scx)
}

#' My function description
#'
#' @export
all_class_together <- function(data, pos_X, pos_y, method){
  data <- data_selection(img, 2:8, 1, 'omit')

  order = data_into_five(data)
  KNN_result <- class_KNN(data, order)
  naive_result <- class_naive(data, order)
  svm_result <- class_svm(data, order)
  tree_stat2(data)
  rf_result <- class_rf(data, order)
  xgb_result <- xgboost123(data, order)
  adab_result <- class_adabst(data, order)
  keras_result <- keras_class(data, order)

  result <- print_result(KNN_result, naive_result, svm_result, rf_result, xgb_result, adab_result, keras_result)
  kme_pca(data) %>% print()
  return(result)
}



####################################################################

#' My function description
#'
#' @export
class_KNN_scale <- function(data, order, trans_method){
  lv = levels(as.factor(unlist(data[ncol(data)])))
  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()

  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data)
    test_data = predict(trans_model, test_data)

    pred = knn(train_data, test_data,
               cl = train_label, k = length(unique(train_label)))
    true = unlist(test_label) %>% as.numeric()

    pred = knn(train_data[,-ncol(train_data)], test_data[,-ncol(test_data)],
               cl = train_label, k = length(unique(train_label)))
    true = unlist(test_label) %>% as.numeric()

    confusion_matrix <- confusionMatrix(factor(pred, levels = lv), factor(true, levels = lv))
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    confusion_matrix$overall

    paste('KNN Accuracy:', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('KNN p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('KNN Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}


##############################################

#' My function description
#'
#' @export
class_naive_scale <- function(data, order, trans_method){
  lv = levels(as.factor(unlist(data[ncol(data)])))
  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data)
    test_data = predict(trans_model, test_data)

    modd <- naiveBayes(train_label ~ ., data = train_data)
    pred = predict(modd, newdata = test_data)
    true = unlist(test_label) %>% as.numeric()

    confusion_matrix <- confusionMatrix(factor(pred, levels = lv), factor(true, levels = lv))
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('Naive Bayes Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Naive Bayes p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Naive Bayes Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}


##############################################

#' My function description
#'
#' @export
class_svm_scale <- function(data, order, trans_method){
  lv = levels(as.factor(unlist(data[ncol(data)])))
  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){

    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data)
    test_data = predict(trans_model, test_data)


    modd = svm(unlist(train_label)  %>% as.numeric() ~ ., data = train_data,
               type = 'C-classification', kernel = 'sigmoid')
    pred = predict(modd, newdata = test_data)
    true = unlist(test_label) %>% as.numeric()

    confusion_matrix <- confusionMatrix(factor(pred, levels = lv), factor(true, levels = lv))
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('SVM Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('SVM p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('SVM Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}


#########################################

#' My function description
#'
#' @export
class_rf_scale <- function(data, order,trans_method){
  set.seed(123)
  oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)

  train_data = data[oooorder,]
  train_label = train_data[,ncol(train_data)]
  train_data = train_data[,-ncol(train_data)]
  train_data = train_data
  trans_model = preProcess(train_data, method = trans_method)
  train_data = predict(trans_model, train_data)  %>% as.matrix()
  test_data = data[-oooorder,]
  test_label <- test_data[,ncol(test_data)]
  test_data <- test_data[,-ncol(test_data)]
  test_data = predict(trans_model, test_data) %>% as.matrix()

  lv = levels(as.factor(unlist(data[ncol(data)])))
  xxx <- c()
  mtry_record = c()
  ntree_record = c()
  set_mtry = c(as.numeric(round(log(ncol(train_data)-1))), as.numeric(round(log(ncol(train_data)-1))) + 1, as.numeric(round(log(ncol(train_data)-1))) + 2)
  set_ntree = seq(400, 1000, 100)
  for(i in 1: length(set_mtry)){
    for (j in 1: length(set_ntree)){
      rf_find_best <- randomForest(as.factor(unlist(train_label) %>% as.numeric()) ~ .,
                                   data = train_data, ntree = set_ntree[j],
                                   mtry = set_mtry[i])
      pred = predict(rf_find_best, newdata = test_data)
      acc <- confusionMatrix(pred, factor(test_label, levels = lv), mode = 'prec_recall')$overall[1]
      xxx <- c(xxx, acc)
      mtry_record = c(mtry_record, rf_find_best$mtry)
      ntree_record = c(ntree_record, rf_find_best$ntree)
      which(xxx == max(xxx))
    }
  }
  paste('The optimal RF ntree is :', ntree_record[which(xxx == max(xxx))]) %>% print()
  paste('The optimal RF mtry is :', mtry_record[which(xxx == max(xxx))]) %>% print()

  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){
    modd = randomForest(as.factor(train_label)  ~ ., data = train_data,
                        ntree = ntree_record[which(xxx == max(xxx))][1], mtry =mtry_record[which(xxx == max(xxx))][1], importance = TRUE)
    varImpPlot(modd)
    pred = predict(modd, newdata = test_data)
    true = unlist(test_label) %>% as.numeric()

    confusion_matrix <- confusionMatrix(factor(pred, levels = lv), factor(true, levels = lv))
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('Random Forest Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Random Forest p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Random Forest Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}


#################################################


#' My function description
#'
#' @export
tree_stat <- function(data){
  r_tree_stat <- rpart(unlist(data[ncol(data)])  %>% as.numeric() ~ ., data = data[,1:ncol(data) - 1], method = 'class')
  r_tree_stat %>% printcp()
  r_tree_stat %>% summary()
  rpart.plot::rpart.plot(r_tree_stat)
}
######################################################



#' My function description
#'
#' @export
# change the class to zero
xgboost_scale <- function(data, order,trans_method){
  acc = c()
  xxx = c()
  depthh = c()
  eeta = c()
  lambdaa = c()
  xgb_data <- data %>% as.matrix()
  set.seed(123)
  label = data[,ncol(data)]
  for(i in 1:length(unique(label))){
    a = i - 1
    b = label == sort(unique(label))[i]
    data[,ncol(data)][which(b == TRUE)] <- a
  }
  oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)
  train_data = data[oooorder,]
  train_label = train_data[,ncol(train_data)]
  train_data = train_data[,-ncol(train_data)]
  train_data = train_data
  trans_model = preProcess(train_data, method = trans_method)
  train_data = predict(trans_model, train_data)  %>% as.matrix()
  test_data = data[-oooorder,]
  test_label <- test_data[,ncol(test_data)]
  test_data <- test_data[,-ncol(test_data)]
  test_data = predict(trans_model, test_data) %>% as.matrix()

  lv = levels(as.factor(unlist(data[ncol(data)])))
  for(i in seq(3,11,2)){
    for(j in c(0.05, 0.1, 0.15)){
      for(k in c(1,2)){
        mod_xg <-
          xgboost(
            data = train_data %>% as.matrix(),
            label = train_label %>% as.numeric(),
            nrounds = 10000,
            objective = "multi:softprob",
            num_class = length(unique(data[,ncol(data)])),
            early_stopping_rounds = 2,
            max_depth = i,
            lambda = k,
            eta = j,
            subsample = 0.9
          )
        pred = predict(mod_xg, newdata = test_data, reshape = TRUE)
        pred = as.data.frame(pred)
        colnames(pred) = sort(unique(train_label))
        pred = apply(pred,1,function(x) colnames(pred)[which.max(x)])
        acc <- confusionMatrix(factor(as.numeric(test_label), levels = lv), factor(pred, levels = lv), mode = 'prec_recall')$overall[1]
        xxx <- c(xxx, acc)
        eeta <- c(eeta, j) # eta
        lambdaa <- c(lambdaa, k) # lambda
        depthh <- c(depthh, i) # max_depth
      }
    }
  }

  paste('The optimal xgboost lambda is :', lambdaa[which(xxx == max(xxx))]) %>% print()
  paste('The optimal xgboost eta is :', eeta[which(xxx == max(xxx))]) %>% print()
  paste('The optimal xgboost max_depth is :', depthh[which(xxx == max(xxx))]) %>% print()

  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()

  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)] %>% as.matrix()
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)] %>% as.matrix()

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data) %>% as.matrix()
    test_data = predict(trans_model, test_data) %>% as.matrix()



    mod_xg <-
      xgboost(
        data = train_data,
        label = train_label %>% as.numeric(),
        nrounds = 10000,
        objective = "multi:softprob",
        num_class = 6,
        early_stopping_rounds = 2,
        max_depth = lambdaa[which(xxx == max(xxx))],
        lambda = lambdaa[which(xxx == max(xxx))],
        eta = eeta[which(xxx == max(xxx))],
        subsample = 0.9
      )
    pred = predict(mod_xg, newdata = test_data, reshape = TRUE)
    pred = as.data.frame(pred)
    colnames(pred) = sort(unique(train_label))
    pred = apply(pred,1,function(x) colnames(pred)[which.max(x)])
    confusion_matrix <- confusionMatrix(factor(as.numeric(test_label), levels = lv), factor(pred, levels = lv), mode = 'prec_recall')
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('Random Forest Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Random Forest p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Random Forest Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}

################################################################################################

#' My function description
#'
#' @export
# class adaboost
class_adabst_scale <- function(data, order, trans_method){
  set.seed(123)
  oooorder = sample(1:nrow(data), floor(0.75 * nrow(data)), replace = FALSE)

  train_data = data[oooorder,]
  train_data = train_data %>% select(-(ncol(train_data) - 1))
  train_label = as.factor(train_data[,ncol(train_data)])
  train_data = train_data[,-ncol(train_data)]
  train_data = train_data
  trans_model = preProcess(train_data, method = trans_method)
  train_data = predict(trans_model, train_data)
  test_data = data[-oooorder,]
  test_label <- test_data[,ncol(test_data)]
  test_data <- test_data[,-ncol(test_data)]
  test_data = predict(trans_model, test_data)


  lv = levels(as.factor(unlist(data[ncol(data)])))
  coeflearn <- c('Breiman', 'Zhu', 'Freund')
  xxx <- c()
  train_data = cbind(train_data, train_label)
  for (i in coeflearn){
    colnames(train_data)[ncol(train_data)] <- 'train_data[, ncol(train_data)]'
    model = boosting(`train_data[, ncol(train_data)]` ~ ., data = train_data, boos = TRUE, mfinal = 500, coeflearn = i)
    pred = predict(model, newdata = test_data)
    acc <- confusionMatrix(factor(pred$class, levels = lv), factor(test_label, levels = lv), mode = 'prec_recall')$overall[1]
    xxx <- c(xxx, acc)
  }
  paste('The Optimal Coeflearn is :', coeflearn[which(xxx == max(xxx))]) %>% print()

  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){
    print(i)
    train_data = data[-unlist(order[i]),]
    train_label = as.factor(train_data[,ncol(train_data)])
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]

    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data)
    test_data = predict(trans_model, test_data)
    train_data = cbind(train_data, train_label)
    colnames(train_data)[ncol(train_data)] <- 'train_data[, ncol(train_data)]'

    modd = boosting(`train_data[, ncol(train_data)]` ~., data = train_data, boos = TRUE, mfinal = 500, coeflearn = coeflearn[which(xxx == max(xxx))][1])
    pred = predict(modd, newdata = test_data)
    confusion_matrix <- confusionMatrix(factor(pred$class, levels = lv), factor(test_label, levels = lv), mode = 'prec_recall')
    acc <- confusion_matrix$overall[1]
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('Adaboost Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Adaboost p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Adaboost Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}

############################################################################

#' My function description
#'
#' @export

keras_class_scale <- function(data, order, trans_method){
  lv = levels(as.factor(unlist(data[ncol(data)])))
  acc = c()
  p_value = c()
  precision = c()
  recall = c()
  f1 = c()
  for(i in 1:5){
    train_data = data[-unlist(order[i]),]
    train_label = train_data[,ncol(train_data)]
    train_data = train_data[,-ncol(train_data)]
    test_data = data[unlist(order[i]),]
    test_label = test_data[,ncol(test_data)]
    test_data = test_data[,-ncol(test_data)]
    trans_model = preProcess(train_data, method = trans_method)
    train_data = predict(trans_model, train_data)
    test_data = predict(trans_model, test_data)
    train_data <- cbind(train_data, train_label) %>% as.data.frame()
    test_data <- cbind(test_data, test_label) %>% as.data.frame()
    colnames(train_data)[ncol(train_data)] <- colnames(data)[ncol(data)]
    colnames(test_data)[ncol(test_data)] <- colnames(data)[ncol(data)]


    pred_name = colnames(train_data)[ncol(train_data)]

    train_file <- paste0(tempdir(), "/train.csv")
    write.csv(train_data, train_file, row.names = FALSE)

    test_file_to_eval <- paste0(tempdir(), "/eval.csv")
    write.csv(test_data, test_file_to_eval, row.names = FALSE)

    class <- model_structured_data_classifier(max_trials = 20) %>%
      fit(
        train_file,
        pred_name,
        validation_data = list(test_file_to_eval, pred_name)
      )
    pred <- class %>% predict(test_file_to_eval) %>% as.factor()
    true = unlist(test_data[ncol(test_data)]) %>% as.numeric() %>% as.factor()

    confusion_matrix <- confusionMatrix(factor(pred, levels = lv), factor(true, levels = lv))
    acc <- c(acc, confusion_matrix$overall[1]) %>% as.numeric()
    p_value <- c(p_value, confusion_matrix$overall[6]) %>% as.numeric()
    result_matrix <- confusion_matrix$byClass %>% as.data.frame()
    precision <- rbind(precision, result_matrix$Precision)
    recall <- rbind(recall, result_matrix$Recall)
    f1 <- rbind(f1, result_matrix$F1)

    paste('Keras_Class Accuracy :', confusion_matrix$overall[1]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Keras_Class p_value :', confusion_matrix$overall[6]) %>% print()
    print("      ")
    print(" __________________________ ")
    print("      ")
    paste('Keras_Class Class_Result : ') %>% print()
    result_matrix %>% print()
  }
  mean_f1 <- apply(f1, 2, mean) %>% matrix(nrow = 1)
  mean_recall <- apply(recall, 2, mean) %>% matrix(nrow = 1)
  mean_precition <- apply(precision, 2, mean) %>% matrix(nrow = 1)
  mean_acc_p <- c(mean(acc), mean(p_value)) %>% matrix(nrow = 1)

  overall_sum = list(mean_precition, mean_recall, mean_f1, mean_acc_p)
  return(overall_sum)
}


#######################

#' My function description
#'
#' @export
print_result <- function(KNN_result, naive_result, svm_result, rf_result, xgb_result, adab_result, keras_result){
  # build a empty data frame to save the result
  acc_and_p <- matrix(ncol = 2, nrow = 0)
  colnames(acc_and_p) <- c('Accuracy', 'p-value')
  acc_and_p <- rbind(acc_and_p, unlist(KNN_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(naive_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(svm_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(rf_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(xgb_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(adab_result[4]))
  acc_and_p <- rbind(acc_and_p, unlist(keras_result[4]))

  rownames(acc_and_p) <- c('KNN', 'Naive_Bayes',
                           'SVM', 'Random Forest',
                           'XGBoost', 'Adaboost', 'AutoML')
  print(acc_and_p)



  # build f1
  f1_value <- matrix(ncol = length(unlist(KNN_result[3])), nrow = 0)

  f1_value <- rbind(f1_value, unlist(KNN_result[3]))
  f1_value <- rbind(f1_value, unlist(naive_result[3]))
  f1_value <- rbind(f1_value, unlist(svm_result[3]))
  f1_value <- rbind(f1_value, unlist(rf_result[3]))
  f1_value <- rbind(f1_value, unlist(xgb_result[3]))
  f1_value <- rbind(f1_value, unlist(adab_result[3]))
  f1_value <- rbind(f1_value, unlist(keras_result[3]))
  rownames(f1_value) <- c('KNN', 'Naive_Bayes',
                          'SVM', 'Random Forest',
                          'XGBoost', 'Adaboost', 'AutoML')

  print(f1_value)


  # build precision
  precision_value <- matrix(ncol = length(unlist(KNN_result[1])), nrow = 0)

  precision_value <- rbind(precision_value, unlist(KNN_result[1]))
  precision_value <- rbind(precision_value, unlist(naive_result[1]))
  precision_value <- rbind(precision_value, unlist(svm_result[1]))
  precision_value <- rbind(precision_value, unlist(rf_result[1]))
  precision_value <- rbind(precision_value, unlist(xgb_result[1]))
  precision_value <- rbind(precision_value, unlist(adab_result[1]))
  precision_value <- rbind(precision_value, unlist(keras_result[1]))
  rownames(precision_value) <- c('KNN', 'Naive_Bayes',
                                 'SVM', 'Random Forest',
                                 'XGBoost', 'Adaboost', 'AutoML')
  print(precision_value)


  # build recall
  recall_value <- matrix(ncol = length(unlist(KNN_result[2])), nrow = 0)

  recall_value <- rbind(recall_value, unlist(KNN_result[2]))
  recall_value <- rbind(recall_value, unlist(naive_result[2]))
  recall_value <- rbind(recall_value, unlist(svm_result[2]))
  recall_value <- rbind(recall_value, unlist(rf_result[2]))
  recall_value <- rbind(recall_value, unlist(xgb_result[2]))
  recall_value <- rbind(recall_value, unlist(adab_result[2]))
  recall_value <- rbind(recall_value, unlist(keras_result[2]))
  rownames(recall_value) <- c('KNN', 'Naive_Bayes',
                              'SVM', 'Random Forest',
                              'XGBoost', 'Adaboost', 'AutoML')
  print(recall_value)

  scx <- list(acc_and_p, f1_value, precision_value, recall_value)
  names(scx) <- c("Accuracy & p-value", "F1-Value", "Precision", "Recall")
  return(scx)
}


#' My function description
#'
#' @export

all_class_together_scaled <- function(data, pos_X, pos_y, method, scale_method){
  data <- data_selection(data, pos_X, pos_y, method)
  order = data_into_five(data)
  KNN_result <- class_KNN_scale(data, order,scale_method)
  naive_result <- class_naive_scale(data, order,scale_method)
  svm_result <- class_svm_scale(data, order,scale_method)
  rf_result <- class_rf_scale(data, order,scale_method)
  xgb_result <- xgboost_scale(data, order,scale_method)
  adab_result <- class_adabst_scale(data, order,scale_method)
  keras_result <- keras_class_scale(data, order,scale_method)
  result <- print_result(KNN_result, naive_result, svm_result, rf_result, xgb_result, adab_result, keras_result)
  kme_pca(data) %>% print()
  return(result)
}



#' My function description
#'
#' @export


stat_ml <- function(data, pos_X, pos_y, objective, na_method, scale_method = NULL){
  method = na_method
  trans_method = scale_method
  start_time = Sys.time()
  if(is.null(scale_method)){
    if(objective == 'reg'){results = all_reg_together(data, pos_X, pos_y, na_method)}
    else{results = all_class_together(data, pos_X, pos_y, na_method)}
  }else{
    if(objective == 'reg'){results = all_reg_together_scaled(data, pos_X, pos_y, na_method, trans_method)}
    else{results = all_class_together_scaled(data, pos_X, pos_y, na_method, scale_method)}
  }
  end_time = Sys.time()
  warning <- "!!! ALL THE RESULTS HAVE BEEN AVERAGED BASED ON CROSS VALIDATION !!!"
  time_diff = end_time - start_time
  print(warning)
  return(results)
  print(time_diff)
}
















