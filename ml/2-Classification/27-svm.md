데이터는 <a href="20-data.md">여기</a>를 참조하거나 다음 명령 실행.

```R
rm(list=ls(all=TRUE))
load(url("https://github.com/chan079/loebook/raw/main/ml/2-Classification/data.RData"))
data(Hmda, package="Ecdat")
```

# Support Vector Machines

앞서 살펴본 [Support Vector Regression](../1-Regression/17-svr.md)은 [Support Vector
Machine][SVM] (SVM)을 회귀 문제에 적용한 것이다. 이제 분류 문제에서
SVM을 살펴보자. 데이터로는 `TrainSet` 대신 random oversample한
데이터(`Over`)를 만들어 사용한다.

```R
overSample <- function(DF) {
  idx1 <- which(DF$deny=='no')
  idx2 <- with(DF, sample(which(deny=='yes'), sum(deny=='no'), replace = TRUE))
  DF[c(idx1,idx2), ]
}

set.seed(1)
Over <- overSample(TrainSet)
```

`e1071` 패키지의 `svm()` 명령을 사용한다.
커널(kernel)로는 선형(linear) 커널을 사용한다.

```R
svmfit0 <- svm(deny~., data=Over, kernel="linear")
summary(svmfit0)
# Call:
# svm(formula = deny ~ ., data = Over, kernel = "linear")
# 
# 
# Parameters:
#    SVM-Type:  C-classification 
#  SVM-Kernel:  linear 
#        cost:  1 
# 
# Number of Support Vectors:  2181
# 
#  ( 1090 1091 )
# 
# 
# Number of Classes:  2 
# 
# Levels: 
#  no yes
```

이 결과를 원래 학습데이터인 `Over`에 적용한 결과는 다음과 같다.

```R
SummPred(Over$deny, predict(svmfit0, Over))
# $ConfusionMatrix
#       pred
# actual   no  yes
#    no  1610  338
#    yes  618 1330
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.6827515   0.8264887   0.7973621   0.7546201 
```

별로 인상적이지 않다. `TestSet`에 적용한 예측 성능은 다음과 같다.

```R
SummPred(TestSet$deny, predict(svmfit0, TestSet))
# $ConfusionMatrix
#       pred
# actual  no yes
#    no  124  23
#    yes   8  12
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.6000000   0.8435374   0.3428571   0.8143713 
```

Lasso보다 약간 낫다.

앞에서 `cost`는 1로 설정되었다. 다양한 cost (0.01, 0.1, 1, 10, 100)에
대해서 10-fold CV를 해 보자.

```R
set.seed(10)
tune.out <- tune(svm, deny~., data=Over, kernel='linear', ranges = list(cost = 10^seq(-2,2)))
summary(tune.out)
# Parameter tuning of ‘svm’:
# 
# - sampling method: 10-fold cross validation 
# 
# - best parameters:
#  cost
#   100
# 
# - best performance: 0.2461499 
# 
# - Detailed performance results:
#     cost     error dispersion
# 1   0.01 0.2546220 0.02108922
# 2   0.10 0.2471795 0.02014827
# 3   1.00 0.2469198 0.02123332
# 4  10.00 0.2464063 0.02123804
# 5 100.00 0.2461499 0.02125690
```

Cost = 100이 최선의 매개변수라고 나왔다. `cost`를 1000까지 올리면
"reaching max number of iterations"라는 경고 메시지가 나오고 제대로
CV가 이루어지지 않았다. `cost`를 100으로 설정한 경우 Test Set에서의
성능은 다음과 같다.

```R
SummPred(TestSet$deny, predict(tune.out$best.model, TestSet))
# $ConfusionMatrix
#       pred
# actual  no yes
#    no  124  23
#    yes   8  12
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.6000000   0.8435374   0.3428571   0.8143713 
```

Cost를 1에서 100으로 증가시켰는데 결과에는 아무런 차이도 없다.

### 수동 CV

수동으로 5-fold CV를 해 보자. 단, 이 경우 random oversampling한
데이터셋이므로 `yes` 클래스로부터 동일한 관측치들이 반복적으로
추출되었으므로 CV가 의미가 있는지 모르겠어서, 우선 (over-sampling하지
않은) Train Set을 `yes`와 `no` 각각에 대해 5개로 분할한 후 통합하고,
각 fold를 over-sampling하고자 한다. 먼저 Train Set을 5개로 분할하자.

```R
## Manual 5-fold CV
summary(TrainSet$deny)
#   no  yes 
# 1948  265
TS.y <- subset(TrainSet, deny=='yes')  # TS: TrainSet
TS.n <- subset(TrainSet, deny=='no')

folds <- 5

set.seed(1)
TS.y$index <- sample(1:folds, nrow(TS.y), replace = TRUE)
TS.n$index <- sample(1:folds, nrow(TS.n), replace = TRUE)
table(TS.y$index)
#  1  2  3  4  5 
# 61 52 43 56 53 
table(TS.n$index)
#   1   2   3   4   5 
# 384 409 362 385 408 
TrainSet2 <- rbind(TS.y, TS.n)
table(TrainSet2$index)
#   1   2   3   4   5 
# 445 461 405 441 461 
```

이제 `TrainSet2`에는 `index`라는 변수가 있어서 1−5 중 하나의 값을
갖는다. 다음으로 다양한 `cost` 값들을 설정하자.

```R
costs <- c(0.01, 0.1, 1, 10, 100, 200)
```

다음으로 각 fold마다 CV error를 구하는 함수를 정의하자. `yes`를
over-sample한 표본에서 틀린 예측 개수를 error로 정의한다.

```R
myfun <- function(DF, k) {
  ans <- rep(NA, length(costs))
  for (j in seq_along(costs)) {
    set.seed(1)
    DF.kth <- overSample(subset(DF, index == k))
    DF.nok <- overSample(subset(DF, index != k))
    svmfit <- svm(deny~.-index, DF.nok, kernel = 'linear', cost = costs[j])
    pred <- predict(svmfit, DF.kth)
    ans[j] <- sum(DF.kth$deny != predict(svmfit, DF.kth)) # error (count)
  }
  ans
}
```

다음으로 병렬처리하여 CV error들을 구하자.

```R
library(foreach)
library(doParallel)

## Prepare for parallel processing
cores <- detectCores() # number of cores
cl <- makeCluster(cores[1]-1, outfile="") # do not redirect to null
registerDoParallel(cl)

## Parallel processing
cverrs <- foreach(fold = 1:folds, .combine = rbind, .packages = c('e1071')) %dopar% {
  cat('Work for fold', fold, '\n') # quiet
  myfun(TrainSet2, fold)
}
# Work for fold 1 
# Work for fold 2 
# Work for fold 4 
# Work for fold 3 
# Work for fold 5 

## Clean up
stopCluster(cl) # Don't forget this

## Results (CV errors)
cverrs
#          [,1] [,2] [,3] [,4] [,5] [,6]
# result.1  196  196  192  193  194  194
# result.2  222  226  221  218  218  218
# result.3  146  148  152  151  151  151
# result.4  196  175  182  181  183  183
# result.5  257  255  254  255  255  255
```

위 마지막 결과에서 각 열은 서로 다른 `cost` 값에 해당하고, 각 행은
fold에 해당한다. 각 열별로 error들의 합을 구하여 비교하면 다음과 같다.

```R
colSums(cverrs)
# [1] 1017 1000 1001  998 1001 1001
```

이 중 가장 작은 값은 4번째이다. 4번째 cost 값은 다음에 의하면 10이다.

```R
costs
# [1]   0.01   0.10   1.00  10.00 100.00 200.00
costs[which.min(colSums(cverrs))]
# [1] 10
```

Cost 값을 10으로 하고 다시 `Over`를 이용하여 SVM 학습을 하고
그 결과를 이용하여 Test Set에서 예측을 하면 결과는 다음과 같다.

```R
svmfit1 <- svm(deny~., Over, kernel = 'linear', cost = 10)
SummPred(TestSet$deny, predict(svmfit1, TestSet))
# $ConfusionMatrix
#       pred
# actual  no yes
#    no  124  23
#    yes   8  12
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.6000000   0.8435374   0.3428571   0.8143713 
```

성능에는 아무런 차이도 없다.

[SVM]: https://en.wikipedia.org/wiki/Support-vector_machine