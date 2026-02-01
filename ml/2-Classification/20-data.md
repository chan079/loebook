# 분류: 데이터 준비

`Ecdat` 패키지를 아직 설치하지 않았으면 설치한다. 이하 전체 코드는 <a href="#allcodes">여기</a>로 이동.

```r
install.packages("Ecdat")
```

`Ecdat` 패키지의 `Hmda` 데이터를 이용하고자 한다. 이 데이터셋은
[Boston HMDA 데이터 셋][hmda]이다(`?Hmda` 참조). 주택담보대출 승인
여부(`deny` 변수)를 예측하는 것이 목적이다. 데이터셋 취지와 맥락 등에
대해서는 [계량경제학강의](../) 17.11절에도 설명하였다.

```R
data(Hmda, package="Ecdat")
dim(Hmda)
# [1] 2381   13
summary(Hmda$deny)
#   no  yes 
# 2096  285 
```

전체 관측치 수는 2,381개이다. 목표변수 `deny`는 2,096개 관측치에서
`no`이고 285개 관측치에서 `yes`이다. 12개 예측변수 후보가 있다.

`NA`가 있는지 확인하자.

```R
which(is.na(Hmda), arr.ind = TRUE)
#       row col
# 2381 2381   6
# 2381 2381   8
```

2,381행의 6열과 8열에 `NA`가 있다. 이 관측치를 제외하자.

```R
Hmda1 <- na.omit(Hmda)
dim(Hmda1)
# [1] 2380   13
table(Hmda1$deny)
#   no  yes 
# 2095  285 
```

남은 관측치 수는 2,380개이다. 그 중 2,095개의 `deny`가 `"no"`이고
285개의 `deny`가 `"yes"`이다. 관측치들의 일부를 테스트셋으로 따로 떼어
두자. 거절된 경우가 승인된 경우의 13.6%이므로 테스트셋에서도 그 정도가
되게끔 거절된(`"yes"`) 관측치 20개, 승인된(`"no"`) 관측치 147개를 준비하겠다.


```R
set.seed(1)
test.index <- c(sample(which(Hmda1$deny=='yes'), 20), sample(which(Hmda1$deny=='no'), 147))
train.index <- setdiff(seq_len(nrow(Hmda1)), test.index)
TrainSet <- Hmda1[train.index, ]
summary(TrainSet$deny)
#   no  yes 
# 1948  265
TestSet <- Hmda1[test.index, ]
summary(TestSet$deny)
#  no yes 
# 147  20 
```

이제 `TrainSet`은 학습용 데이터셋, `TestSet`은 테스트용 데이터셋이다.

앞으로 실제와 예측 결과를 비교한 테이블([confusion matrix]라 함;
`table(actual,pred)` 방식)을 만들고
[sensitivity][evalbin]([recall][evalbin]이라고도 함)와
[specificity][evalbin]를 구할 것이다. Positive (`yes`)로 예측한 것 중
실제로 positive인 경우의 비율인 [precision][evalbin] (정밀도)과, 전체
중 정확히 맞춘 경우의 비율을 나타내는 [accuracy][evalbin]도 살펴볼
것이다. 자주 사용할 것이므로 아예 함수를 만들어 놓겠다.

```R
SummPred <- function(actual, pred) {
  x <- table(actual, pred)
  se <- x['yes','yes']/sum(x['yes',])
  sp <- x['no','no']/sum(x['no',])
  pr <- x['yes','yes']/sum(x[,'yes'])
  ac <- (x['yes','yes']+x['no','no'])/sum(x)
  ans <- c(Sensitivity = se, Specificity = sp, Precision = pr, Accuracy = ac)
  list(ConfusionMatrix = x, Summary = ans)
}
# Example: SummPred(TestSet$deny, yhat)
```

위의 `SummPred` 함수를 사용하려면 우선 목표변수 예측치를 구하여야
한다. 이 과정을 모두 다음 `Performance()` 함수로 구현한다. 이 함수가
복잡한 것은 R에서 `glm`, `glmnet`, `lda`와 `qda`가 확률 또는
목표변수를 예측하는 방식이 모두 다르기 때문이다.

```R
Performance <- function(object, DataSet, cutoff = 0.5, fm = deny~., ...) {
  y <- model.frame(fm, data = DataSet)[, 1]
  if (inherits(object, 'glm')) {
    phat <- predict(object, DataSet, type = 'r', ...)
  } else if (inherits(object, 'glmnet') || inherits(object, 'cv.glmnet')) {
    X <- model.matrix(fm, data = DataSet)[, -1]
    phat <- predict(object, X, type = 'r', ...)
  } else if (inherits(object, 'lda') || inherits(object, 'qda')) { # LDA and QDA
    phat <- predict(object, DataSet)$posterior[,'yes']
  } else if (inherits(object, 'tree')) {
    phat <- predict(object, DataSet)[, 'yes']
  } else if (inherits(object, 'randomForest')) {
    phat <- predict(object, DataSet, type = 'prob')[, 'yes']
  } else if (inherits(object, 'boosting')) {
    phat <- predict(object, DataSet)$prob[,2]
  } else {
    stop('Not implemented for ', class(object))
  }
  yhat <- levels(y)[1 + (phat >= cutoff)]
  SummPred(y, yhat)
}
```

<a name="allcodes">전체 코드는 다음과 같다.</a>

```R
data(Hmda, package="Ecdat")
Hmda1 <- na.omit(Hmda)
set.seed(1)
test.index <- c(sample(which(Hmda1$deny=='yes'), 20), sample(which(Hmda1$deny=='no'), 147))
train.index <- setdiff(seq_len(nrow(Hmda1)), test.index)
TrainSet <- Hmda1[train.index, ]
TestSet <- Hmda1[test.index, ]

SummPred <- function(actual, pred) {
  x <- table(actual, pred)
  se <- x['yes','yes']/sum(x['yes',])
  sp <- x['no','no']/sum(x['no',])
  pr <- x['yes','yes']/sum(x[,'yes'])
  ac <- (x['yes','yes']+x['no','no'])/sum(x)
  ans <- c(Sensitivity = se, Specificity = sp, Precision = pr, Accuracy = ac)
  list(ConfusionMatrix = x, Summary = ans)
}

Performance <- function(object, DataSet, cutoff = 0.5, fm = deny~., ...) {
  y <- model.frame(fm, data = DataSet)[, 1]
  if (inherits(object, 'glm')) {
    phat <- predict(object, DataSet, type = 'r', ...)
  } else if (inherits(object, 'glmnet') || inherits(object, 'cv.glmnet')) {
    X <- model.matrix(fm, data = DataSet)[, -1]
    phat <- predict(object, X, type = 'r', ...)
  } else if (inherits(object, 'lda') || inherits(object, 'qda')) { # LDA and QDA
    phat <- predict(object, DataSet)$posterior[,'yes']
  } else if (inherits(object, 'tree')) {
    phat <- predict(object, DataSet)[, 'yes']
  } else if (inherits(object, 'randomForest')) {
    phat <- predict(object, DataSet, type = 'prob')[, 'yes']
  } else if (inherits(object, 'boosting')) {
    phat <- predict(object, DataSet)$prob[,2]
  } else {
    stop('Not implemented for ', class(object))
  }
  yhat <- levels(y)[1 + (phat >= cutoff)]
  SummPred(y, yhat)
}
```

[hmda]: https://www.bostonfed.org/home/publications/research-department-working-paper/1992/mortgage-lending-in-boston-interpreting-hmda-data.aspx
[confusion matrix]: https://en.wikipedia.org/wiki/Confusion_matrix
[evalbin]: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
