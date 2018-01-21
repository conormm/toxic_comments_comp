## approach to finding sythetic twins 
# used to find neighbours in random forest tree / node space
# inspired by these tweets:

#The idea is very simple. You want to compare two groups' outcomes on some metric y. But the two groups are quite different in their covariates X. One approach is to predict y given X using a random forest, saving the proximity matrix. This i,jth entry in this matrix is the
#proportion of terminal nodes shared by observations i and j. That is, "how similar" they are in some random forest space. Now, for each person in group B, select the corresponding person in group A that is nearest in this this space. And compute your differences on this subset
#There's your synthetic control group. The neat thing about this 'distance', unlike both propensity score/Mahalanobis distance matching, is that it's a) the distance in the X space, and b) _supervised_--it's the similarity in terms of the Xs that matter to the prediction of y.

# good start - the part you haven't done is to ensure that closest match to
# i_a is from j_b not j_a

library(randomForest)
library(dplyr)
library(janitor)
library(purrr)

set.seed(123)

df <- clean_names(as_data_frame(iris))[, 1:4]
group <- sample(c("A", "B"), 1, size = nrow(df))
group_ix_a <- group == 'A' 
group_ix_b <- group == 'B'
fit <- randomForest(sepal_length ~ ., data = df, proximity=TRUE)

prox_mat <- fit$proximity
diag(prox_mat) <- 0

apply(prox_mat, 1, function(x) max(x, na.rm = TRUE))

ifelse(prox_mat == group_ix_a, )

a = data_frame(a_close = apply(prox_mat, 1, function(x) which(x == max(x[group_ix_a], na.rm = TRUE)))) 
b = data_frame(b_close = apply(prox_mat, 1, function(x) which(x == max(x[group_ix_b], na.rm = TRUE)))) 
a_close[2][[1]][[1]]
group_ix_b


invoke_rows(prox_mat, function(x) which(x == max(x[group_ix_a], na.rm = TRUE)))


closest_ob <- data.frame(
  ix_i = apply(prox_mat, 1, function(x) which(x == max(x, na.rm = TRUE)))) %>% 
  mutate(
    closest_ix_j = row.names(.), 
    group = group, 
    y_i = df$sepal_length, 
    y_j = df$sepal_length[.$ix_i]
  )

closest_ob
   
  

  
closest_ob



df[c(117, 150), ]
View(prox_mat)
prox_mat[, 150]
data.frame(prox_mat[150, ])
