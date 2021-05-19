##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(dslabs)
library(caret)
library(dplyr)
library(tidyverse)
library(lubridate)
library(recosystem)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                          title = as.character(title),
#                                         genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################################################################################################################
###################################################################################################################
###################################################################################################################

options(digits=5)

###################################################################################################################
###################################################################################################################
#DATA ANALYSIS#####################################################################################################
###################################################################################################################

#summary
str(edx)
summary(edx)

#ratings-histogram
edx %>% ggplot(aes(rating)) + geom_histogram(bins = 10, col="black") + geom_vline(aes(xintercept=mean(rating)), size=2, color="red") + theme_minimal()

#movies - most favoured and boxplot of random sample
edx %>% group_by(movieId) %>% filter(n()>150) %>% summarize(title,rating=mean(rating)) %>% unique() %>% 
  arrange(desc(rating)) %>% head(10)

edx %>%  group_by(movieId) %>% filter(n()>150) %>% ungroup() %>% filter(movieId %in% sample(movieId, 30, replace=FALSE)) %>% 
  ggplot(aes(as.factor(movieId), rating)) + geom_boxplot() + theme_minimal() + theme(axis.text.x=element_text(angle=90)) + xlab("movieId")

#users - boxplot of random sample
edx %>%  group_by(userId) %>% filter(n()>150) %>% ungroup() %>% filter(userId %in% sample(userId, 30, replace=FALSE)) %>% 
  ggplot(aes(as.factor(userId), rating)) + geom_boxplot() + theme_minimal() + theme(axis.text.x=element_text(angle=90)) + xlab("userId")

#time
edx_time <- edx %>% mutate(date = round_date(as_datetime(timestamp), unit="week")) %>% mutate(date_1= as.numeric(year(as_datetime(timestamp))), appearance=str_extract(title, "\\([0-9]{4}\\)")) %>% 
  mutate(appearance=as.numeric(str_extract(appearance, "[0-9]+"))) %>% mutate(lapse=date_1-appearance)

#lapse between appearance and rating
edx_time %>% group_by(lapse) %>% summarize(rating=mean(rating), n=n()) %>% ggplot(aes(lapse, rating)) +
  geom_point(aes(size=n)) + geom_smooth() +theme_minimal()

#appearance versus rating
edx_time %>% group_by(appearance) %>% summarize(rating=mean(rating), n=n()) %>% ggplot(aes(appearance, rating)) +
  geom_point(aes(size=n)) + geom_smooth() +theme_minimal()

#time effect of rating
edx_time %>% group_by(date) %>% summarize(rating=mean(rating), n=n()) %>% ggplot(aes(date, rating)) +
  geom_point(aes(size=n)) + geom_smooth() +theme_minimal()

#boxplots
edx_time %>% ggplot(aes(as.factor(lapse), rating)) + geom_boxplot() + xlab("years between appearance and rating") + theme(axis.text.x = element_text(angle = 90)) + scale_x_discrete(breaks=seq(0,95,5)) +theme_minimal()

edx_time %>% mutate(year=year(date)) %>% ggplot(aes(as.factor(year), rating)) + geom_boxplot() + xlab("year of rating") +theme_minimal()

#genres

edx_genres <- edx %>% group_by(userId) %>%  filter(n()>=150) %>% ungroup() %>% ## To avoid memory overflow create a subset of the data, keeping only users that have rated more than 150 movies.
  separate(genres, into=c("a","b","c","d","e","f","g","h"), sep="\\|", fill="right") %>%
  gather(key="bla", value="genre", 6:13, na.rm=TRUE) %>% select(-bla)

edx_genres %>% group_by(genre) %>% summarize(n=n()) %>% arrange(desc(n))

edx_genres %>% ggplot(aes(as.factor(genre), rating)) + geom_boxplot() + theme_minimal() + theme(axis.text.x = element_text(angle = 90))

edx_genres %>% filter(userId %in% sample(edx_genres$userId,4, replace=FALSE)) %>% 
  ggplot(aes(as.factor(genre), rating)) + geom_boxplot() + facet_grid(. ~ userId) + theme_minimal() + 
  theme(axis.text.x =element_text(angle = 90)) + xlab(" ")

###################################################################################################################
###################################################################################################################
#MODEL BUILDING####################################################################################################
###################################################################################################################


#loss function used in these analyses
RMSE <- function(test,pred){sqrt(mean((test-pred)^2))} 

#Make a data partition for model optimization
ind<-createDataPartition(edx$rating, times=1, p=0.2, list=FALSE)
train <- edx[-ind,]
test <- edx[ind,] %>% semi_join(train, by="movieId") %>% semi_join(train, by="userId")

##Random ratings#################################################################################################

pred <- sample(seq(0.5,5,0.5), nrow(test), replace=TRUE)

rmse <- RMSE(test$rating, pred)

results <- data.frame(what="random rating", rmse=rmse)
#results

##Simply mean####################################################################################################

mu <- mean(test$rating) #rating mean

rmse <- RMSE(test$rating, mu)

results<- bind_rows(results, data.frame(what="mu", rmse=rmse))

## Movie effect as in course Y_hat = mu + bi + e #################################################################

movie_effects <- train %>% group_by(movieId) %>% summarize(bi = mean(rating - mu)) 

pred <- test %>% inner_join(movie_effects, by="movieId") %>% mutate(pred=mu+bi) %>% .$pred

rmse <- RMSE(test$rating, pred)

results<- bind_rows(results, data.frame(what="mu+bi", rmse=rmse))

## Adding user effect as in course Y_hat = mu + bi + bu + e ######################################################

user_effects <- train %>% left_join(movie_effects, by="movieId") %>% group_by(userId) %>% summarize(bu=mean(rating-mu-bi))

pred <- test %>% left_join(movie_effects, by="movieId") %>% left_join(user_effects, by="userId") %>% mutate(pred=mu+bu+bi) %>% .$pred

rmse <- RMSE(test$rating,pred)

results <- bind_rows(results, data.frame(what="mu+bi+bu", rmse=rmse))

## REGULARIZATION################################################################################################

# finding best lambda_bi

lambda_bi<-seq(0,10,0.25)

rmses<- sapply(lambda_bi, function(lam){
  reg_movie_effects <- train %>% group_by(movieId) %>% summarize(bi = sum(rating - mu)/(n()+lam)) 
  pred <- test %>% inner_join(reg_movie_effects, by="movieId") %>% mutate(pred=mu+bi) %>% .$pred
  return(RMSE(test$rating, pred))})

plot(lambda_bi,rmses)
lambda_bi<- lambda_bi[which.min(rmses)]


# applying best lambda_bi       

reg_movie_effects <- train %>% group_by(movieId) %>% summarize(bi_reg = sum(rating-mu)/(n()+lambda_bi))
pred <- test %>% inner_join(reg_movie_effects, by="movieId") %>% mutate(pred=mu+bi_reg) %>% .$pred
rmse <- RMSE(test$rating,pred)       
results<- bind_rows(results, data.frame(what="mu+bi_reg", rmse=rmse))

# finding best lambda_bu

lambda_bu<-seq(0,10,0.25)

rmses<- sapply(lambda_bu, function(lam){
  reg_user_effects <- train %>% left_join(reg_movie_effects, by="movieId") %>% group_by(userId) %>% summarize(bu_reg = sum(rating - mu - bi_reg)/(n()+lam)) 
  pred <- test %>% left_join(reg_user_effects, by="userId") %>% left_join(reg_movie_effects, by="movieId") %>% mutate(pred=mu+bi_reg+bu_reg) %>% .$pred
  return(RMSE(test$rating, pred))})

plot(lambda_bu,rmses)
lambda_bu<- lambda_bu[which.min(rmses)]



# applying best lambda_bu       

reg_user_effects <- train %>% left_join(reg_movie_effects, by="movieId") %>% group_by(userId) %>% summarize(bu_reg = sum(rating - mu - bi_reg)/(n()+lambda_bu)) 
pred <- test %>% left_join(reg_user_effects, by="userId") %>% left_join(reg_movie_effects, by="movieId") %>% mutate(pred=mu+bi_reg+bu_reg) %>% .$pred

rmse <- RMSE(test$rating,pred)       
results<- bind_rows(results, data.frame(what="mu+bi_reg+bu_reg", rmse=rmse))
#results
rm(movie_effects, user_effects, pred, rmse, rmses)

##TIME DEPENDANCE##############################################################################################################

train <- mutate(train, date = round_date(as_datetime(timestamp), unit="week"))
test <- mutate(test, date = round_date(as_datetime(timestamp), unit="week"))

train_1<- train %>% left_join(reg_movie_effects, by="movieId") %>% left_join(reg_user_effects, by="userId") %>% mutate(diff=rating-mu-bi_reg-bu_reg) 

train_1 %>% group_by(date) %>% summarize(diff = mean(diff)) %>% ggplot(aes(date, diff)) + geom_point() +  geom_smooth()

fit_date <- lm(diff~date, data=train_1)

pred<-test %>% left_join(reg_user_effects, by="userId") %>% left_join(reg_movie_effects, by="movieId") %>% 
  mutate(date=round_date(as_datetime(timestamp), unit="week")) %>% mutate(time_dep=predict(fit_date, test)) %>%
  mutate(pred=mu+bi_reg+bu_reg+time_dep) %>% .$pred

rmse <- RMSE(test$rating,pred)
results<-bind_rows(results,data.frame(what="mu+bi_reg+bu_reg+time_dep", rmse=rmse))
#results

rm(pred)


##1980-Effect######################################################################################################
train_1980 <- train %>% left_join(reg_movie_effects, by="movieId") %>% left_join(reg_user_effects, by="userId") %>%
  mutate(diff=rating-mu-bi_reg-bu_reg) %>%  mutate(date_1= as.numeric(year(as_datetime(timestamp))), appearance=str_extract(title, "\\([0-9]{4}\\)")) %>% 
  mutate(appearance=as.numeric(str_extract(appearance, "[0-9]+")))

effect_1980 <- train_1980 %>% mutate(is1980=ifelse(appearance>1980,0,1)) %>% group_by(is1980) %>% 
  summarize(effect_1980=mean(diff))

pred <- test %>% left_join(reg_movie_effects, by="movieId") %>% left_join(reg_user_effects, by="userId") %>%
  mutate(date_1= as.numeric(year(as_datetime(timestamp))), appearance=str_extract(title, "\\([0-9]{4}\\)")) %>% 
  mutate(appearance=as.numeric(str_extract(appearance, "[0-9]+"))) %>% mutate(is1980=ifelse(appearance>1980,0,1))%>%
  left_join(effect_1980, by="is1980") %>% mutate(pred=mu+bi_reg+bu_reg+effect_1980) %>% .$pred

rmse <- RMSE(test$rating,pred)
results<-bind_rows(results,data.frame(what="mu+bi_reg+bu_reg+effect_1980", rmse=rmse))
results


##GENRE DEPENDANCE################################################################################################

# adding the effect_1980 to dataset
train_1 <- train_1980 %>% mutate(is1980=ifelse(appearance>1980,0,1)) %>% left_join(effect_1980, by="is1980") %>%
  mutate(diff=rating-mu-bi_reg-bu_reg-effect_1980) %>%
  select(userId, movieId, rating, timestamp, title, genres, bi_reg, bu_reg, effect_1980, diff)

## To avoid memory overflow create a subset of the data, keeping only users that have rated more than 150 movies.
train_subset <- train_1 %>% group_by(userId) %>%  filter(n()>=150) %>% ungroup()

# expanding individual genres
train_expanded <- train_subset %>% 
  separate(genres, into=c("a","b","c","d","e","f","g","h"), sep="\\|", fill="right") %>%
  gather(key="bla", value="genre", 6:13, na.rm=TRUE) %>% select(-bla)

user_genre_rating<-train_expanded %>% group_by(userId, genre) %>% summarize(avg_diff=mean(diff))

rm(train_expanded)

#same for the test set
test_expanded <- test %>% left_join(reg_user_effects, by="userId") %>% left_join(reg_movie_effects, by="movieId") %>%
  mutate(appearance=str_extract(title, "\\([0-9]{4}\\)")) %>% 
  mutate(appearance=as.numeric(str_extract(appearance, "[0-9]+"))) %>%
  mutate(is1980=ifelse(appearance>1980,0,1)) %>% left_join(effect_1980, by="is1980") %>% select(-is1980) %>%
  separate(genres, into=c("a","b","c","d","e","f","g","h"), sep="\\|", fill="right") %>% 
  gather(key="bla", value="genre", 6:13, na.rm=TRUE) %>% select(-bla) 

#make predictions based on above
pred <-test_expanded %>% left_join(user_genre_rating, by=c("userId","genre"))  %>% 
  mutate_if(is.numeric, funs(ifelse(is.na(.), 0, .))) %>% group_by(userId, movieId) %>% mutate(ugr=mean(avg_diff)) %>%
  summarize(bu_reg,bi_reg,effect_1980,ugr) %>% unique()  %>% mutate(pred=mu+bi_reg+bu_reg+effect_1980+ugr) %>% 
  .$pred

rmse1 <- RMSE(test$rating,pred)
results<-bind_rows(results,data.frame(what="mu+bi_reg+bu_reg+effect_1980+user_genre_dependance", rmse=rmse1))
# results

###################################################################################################################
#MatrixFactorization ##############################################################################################
## https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html

###########ITERATIONS##############################################################
# Is there a continuous improvement with more iterations or does the improvement fade?


# how many iterations

set.seed(130484)

train_2 <- data_memory(train$userId, train$movieId, rating = train$rating)
test_2 <- data_memory(test$userId, test$movieId)

iter <- seq(5,145,20)

doiter<-function(n){
  
  r<-Reco()
  
  r$train(train_2, opt=list(niter=c(n)))
  
  pred<- r$predict(test_2,out_memory()) 
  
  rmse <- RMSE(test$rating,pred)       
  
  rmse
}

iterationtable <- data.frame(iterations=iter, rmse=sapply(iter,doiter))

iterationtable

iterationtable %>% ggplot(aes(iterations, rmse)) + geom_point()

######SENSITIVITY OF PARAMETERS################################################
# look at every parameter individually and test 5 parameters between the lowest and 150% of the highest default
# parameter

m<-5 # how many data points to generate

dim <- seq(5,30,length.out=m)
dimfunc <- function(n){
  r<-Reco()
  r$train(train_2,opt=list(dim=c(n)))
  pred <- r$predict(test_2,out_memory()) 
  
  rmse <- RMSE(test$rating,pred)       
  
  rmse
}

costp_l1 <- seq(0,0.15,length.out=m)
costp_l1func <- function(n){
  r<-Reco()
  r$train(train_2,opt=list(costp_l1=c(n)))
  pred <- r$predict(test_2,out_memory()) 
  
  rmse <- RMSE(test$rating,pred)       
  
  rmse
}

costp_l2 <- seq(0.01,0.15,length.out=m)
costp_l2func <- function(n){
  r<-Reco()
  r$train(train_2,opt=list(costp_l1=c(n)))
  pred <- r$predict(test_2,out_memory()) 
  
  rmse <- RMSE(test$rating,pred)       
  
  rmse
} 

costq_l1 <- seq(0,0.15,length.out=m)
costq_l1func <- function(n){
  r<-Reco()
  r$train(train_2,opt=list(costp_l1=c(n)))
  pred <- r$predict(test_2,out_memory()) 
  
  rmse <- RMSE(test$rating,pred)       
  
  rmse
}

costq_l2 <- seq(0.01,0.15,length.out=m)
costq_l2func <- function(n){
  r<-Reco()
  r$train(train_2,opt=list(costp_l1=c(n)))
  pred <- r$predict(test_2,out_memory()) 
  
  rmse <- RMSE(test$rating,pred)       
  
  rmse
} 

lrate <- seq(0.01,0.15, length.out=m)
lratefunc <- function(n){
  r<-Reco()
  r$train(train_2,opt=list(costp_l1=c(n)))
  pred <- r$predict(test_2,out_memory()) 
  
  rmse <- RMSE(test$rating,pred)       
  
  rmse
} 

sensitivity <- data.frame(n=seq(1,m), dim=sapply(dim, dimfunc), costp_l1=sapply(costp_l1, costp_l1func), 
                          costp_l2=sapply(costp_l2, costp_l2func), costq_l1=sapply(costq_l1, costq_l1func),
                          costq_l2=sapply(costq_l2, costq_l2func), lrate=sapply(lrate,lratefunc))

sensitivity %>% gather(key="parameter", value="rmse",2:7) %>% ggplot(aes(n,rmse, col=parameter)) + geom_point()
rm(r)

#Matrix Factorization (using 75 iterations as a compromise between cpu time and improvement and tuning the dim parameter)
set.seed(130484)

train_2 <- data_memory(train_1$userId, train_1$movieId, rating = train_1$rating)
test_2 <- data_memory(test$userId, test$movieId)

r<-Reco()

params <- r$tune(train_2, opt=list(dim=seq(5,25,length.out=4), costp_l1=c(0), 
                                   costp_l2=c(0.01), costq_l1=c(0), costq_l2=c(0.01), lrate=c(0.01)))

r$train(train_2, opt=list(params$min, niter=c(75)))

pred<- r$predict(test_2,out_memory()) 

rmse <- RMSE(test$rating,pred)       
results<- bind_rows(results, data.frame(what="matrix_factorization_only", rmse=rmse))


rm(test_2, train_2, r)

#MatrixFactorization on differentials#############################################################################
## https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html

set.seed(130484)

train_2<- data_memory(train_1$userId, train_1$movieId, rating = train_1$diff)
test_2 <- data_memory(test$userId, test$movieId)

r<-Reco()

params <- r$tune(train_2, opt=list(dim=seq(5,25,length.out=4), costp_l1=c(0), 
                                   costp_l2=c(0.01), costq_l1=c(0), costq_l2=c(0.01), lrate=c(0.01)))

r$train(train_2, opt=list(params$min, niter=c(75)))

pred<- r$predict(test_2,out_memory()) 

pred<- test %>% left_join(reg_user_effects, by="userId") %>% 
  left_join(reg_movie_effects, by="movieId") %>% 
  mutate(appearance=str_extract(title, "\\([0-9]{4}\\)")) %>% 
  mutate(appearance=as.numeric(str_extract(appearance, "[0-9]+"))) %>% 
  mutate(is1980=ifelse(appearance>1980,0,1))%>%
  left_join(effect_1980, by="is1980") %>% 
  mutate(pred_diff=pred) %>% mutate(pred=mu+bi_reg+bu_reg+effect_1980+pred_diff) %>% 
  .$pred

rmse_1 <- RMSE(test$rating,pred)       
results<- bind_rows(results, data.frame(what="mu+bi_reg+bu_reg+effect_1980+matFact(diffs)", rmse=rmse_1))

rm(r)

###################################################################################################################
###################################################################################################################
###################################################################################################################
#Finally training model on edx dataset and test with validation
###################################################################################################################

mu <- mean(edx$rating) #rating mean

#regularized movie and user effects
reg_movie_effects <- edx %>% group_by(movieId) %>% summarize(bi_reg = sum(rating-mu)/(n()+lambda_bi)) # movie effect

edx <- edx %>% left_join(reg_movie_effects, by="movieId")

reg_user_effects <- edx %>% group_by(userId) %>% summarize(bu_reg = sum(rating - mu - bi_reg)/(n()+lambda_bu)) # user effect

edx <- edx  %>% left_join(reg_user_effects, by="userId")

#1980 effect
edx <- edx %>% mutate(appearance=str_extract(title, "\\([0-9]{4}\\)")) %>% 
  mutate(appearance=as.numeric(str_extract(appearance, "[0-9]+")), diff_1=rating-mu-bi_reg-bu_reg)

effect_1980 <- edx %>% mutate(is1980=ifelse(appearance>1980,0,1)) %>% 
  group_by(is1980) %>% summarize(effect_1980=mean(diff_1))

edx <- edx %>% mutate(is1980=ifelse(appearance>1980,0,1)) %>% 
  left_join(effect_1980, by="is1980") %>% 
  mutate(diff=rating-mu-bi_reg-bu_reg-effect_1980) %>% select(-is1980)
  

#matrix factorization on differentials
set.seed(130484)

edx_2<- data_memory(edx$userId, edx$movieId, rating = edx$diff)

r<-Reco()

params <- r$tune(edx_2, opt=list(dim=seq(5,25,length.out=4), costp_l1=c(0), 
                                 costp_l2=c(0.01), costq_l1=c(0), costq_l2=c(0.01), lrate=c(0.01)))

r$train(edx_2, opt=list(params$min, niter=c(75)))

#final predictions on validation data
validation <- validation %>% left_join(reg_user_effects, by="userId") %>% left_join(reg_movie_effects, by="movieId")
validation <- validation %>% mutate(appearance=str_extract(title, "\\([0-9]{4}\\)")) %>% 
  mutate(appearance=as.numeric(str_extract(appearance, "[0-9]+"))) %>%
  mutate(is1980=ifelse(appearance>1980,0,1)) %>% 
  left_join(effect_1980, by="is1980") %>%  mutate(diff=rating-mu-bi_reg-bu_reg-effect_1980) %>% 
    select(-is1980)
validation_1 <- data_memory(validation$userId, validation$movieId, rating = validation$diff)
validation <- validation %>% mutate(mat_fact_diff = r$predict(validation_1,out_memory()))

pred <- validation %>% mutate(pred=mu+bi_reg+bu_reg+effect_1980+mat_fact_diff) %>% .$pred

#final RMSE
final_rmse <- RMSE(validation$rating, pred)

#final Result
final_rmse
