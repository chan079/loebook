# Super Learner

Breiman (1996)의 [Stacked Regressions][Breiman96], Kennedy (2017)의
[Guide to SuperLearner][Kennedy17] (R [SuperLearner][SuperLearner-pkg]
패키지 사용설명서), Polley and van der Laan (2010)의 [Super Learner in
Prediction][PolleyLaan10], [H2O.ai document][h2o.stack], Naimi and
Balzer (2018)의 [Stacked Generalization: An Introduction to Super
Learning][NaimiBalzer18]을 참고하였다.

[Breiman96]: https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf

[Kennedy17]: https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html

[PolleyLaan10]: https://biostats.bepress.com/ucbbiostat/paper266/

[h2o.stack]: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html

[NaimiBalzer18]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6089257/

## 개요

Super Learning은 여러 방법을 종합하는 앙상블(ensemble)로서 일종의
“메타러닝”(학습한 결과를 다시 학습하는 것)이다. 이 방법은 여러
알고리즘으로부터의 예측치들을 결합하여 최적을 만드는 방법이다. Breiman
(1996)은 이 방법을 “stacked regression”이라 하였다. [H2O.ai
document][h2o.stack]는 이 앙상블을 Stacking 또는 Super Learning이라
칭한다. Stacking / Super Learning의 목적은 여러 알고리즘으로부터의
예측치들을 선형결합하여 최종 예측치를 만드는 것이다. 문제는 여러
방법들이 결국은 동일한 값을 예측하는 것이기 때문에 아주 강하게
상관되어 있다는 것인데(Breiman, 1996), Breiman (1996)은 이 문제를
해결하기 위하여 선형결합할 계수가 음수가 되지 않도록(non-negativity)
하는 방법을 제안하였다(Breiman 논문 2절의 제목은 “Why Non-negativity
Constraints Work”이다). Breiman (1996)의 stacked regression이 나중에
Super Learning으로 일반화되었다. 선형결합을 하지 않고 여러 알고리즘 중
가장 좋은 것 하나를 선택하는 것을 Discrete Super Learner라
한다. 그보다는 보통 Breiman (1996)의 최적 가중평균(합이 1이고
비음(&geq;0)인 가중치 사용)에 따라 선형결합을 한다. 이 ‘상위 수준
학습’에 사용하는 데이터는 여러 알고리즘으로부터의 예측치이며, 이를
Breiman (1996)은 “level one” 데이터라고 하였다.

Polley and van der Laan (2010)과 Naimi and Balzer (2018)에 설명에
기초하여 구체적인 절차를 정리해 보면 다음과 같다.[^note1]

[^note1]: Breiman (1996)의 설명은 명료하다. Polley and van der Laan (2010)의 설명은 중간에 불분명한 곳이 있어 이해할 수 없었고, Naimi and Balzer (2018)의 설명을 더 쉽게 따라갈 수 있었다.

1. V-fold CV를 위한 데이터셋을 만든다(예를 들어 V=5).
2. 각 fold와 알고리즘에 대하여 훈련용 데이터(5-fold라면 80% 데이터)로
   학습한 결과를 검증용 데이터(5-fold라면 나머지 20% 데이터)에
   적용하여 예측치와 예측오차의 정도(risk, 보통 MSE)를 구한다.
3. 각 알고리즘에 대하여 이들 CV 예측오차(MSE) 평균을 구한다. 참고로,
   이 중 CV 예측오차 평균이 가장 작은 알고리즘이 'Discrete Super
   Learner'이다.
4. 상위수준(“level-one”) 학습을 통하여 각 알고리즘에 가중치를
   부여한다. 이 학습 과정이 Super Learning이다. 이에 대해서는 나중에
   자세히 설명한다.
5. 가중치를 얻으면, 각각의 알고리즘으로 전체 학습용 데이터셋에 대하여
   학습한 예측치들을 가중치에 따라 가중평균내어 최종 예측치를 구한다.

### 데이터 읽기

시작하기 전에, (아직 실행하지 않았으면) [데이터 준비](10-data.md) 페이지 마지막의 코드를 한꺼번에 실행해서 데이터를 준비하라.

## Discrete Super Learner

앞에서 고려한 OLS, best subset selection, ridge, lasso, PCR, PLS, SVM,
Random Forest 등 8개 방법들을 10-fold CV로 비교해 보자(신경망은 시간이
많이 걸리므로 이 실습에서 제외). 튜닝 매개변수가 있는 경우 해당
항목들에서 CV로 찾은 최적 튜닝 매개변수값을 사용하고자 한다. 다음을
실행해 두자.

```R
library(leaps)
library(glmnet)
library(pls)
library(randomForest)
library(e1071)

## Convenience function; don't use it inside a function
predict.regsubsets <- function(x, newx, id) {
  bhat <- coef(x, id)
  if (is.data.frame(newx)) newx <- model.matrix(eval(x$call[[2]]), newx)
  as.numeric(newx[, names(bhat), drop = FALSE] %*% bhat)
}

## CV for ridge and lasso
set.seed(1)
cv.ridge2 <- cv.glmnet(X,Y,alpha=0, lambda.min.ratio=1e-6, nlambda=200)
set.seed(1)
cv.lasso <- cv.glmnet(X,Y, alpha=1)  # 0 = ridge, 1 = lasso
```

이제 10-fold CV를 할 것인데, 8개 방법 각각을 사용한 CV 예측치(9
folds를 이용하여 훈련한 결과를 나머지 1개 fold에 적용한 예측치)들을
구하고자 한다. 10-fold CV를 위하여 관측치(행)들을 10개로 분할하기 위한 준비를 하자.

```R
## Compare all methods by CV
set.seed(1)
group <- sample(rep(1:10, nrow(z14)), nrow(z14)) # same grouping as before
table(group)
# group
#  1  2  3  4  5  6  7  8  9 10 
# 26 21 17 22 24 23 20 23 24 23
```

각 fold마다, 8개 방법 각각에 대하여, 해당 fold를 제외한 나머지 자료를
이용하여 모형을 train하고 그 결과를 해당 fold에 적용하여 구한 예측치를
구하는 함수 `myfun`을 다음과 같이 정의하자.

```R
myfun <- function(DF.train, DF.pred) {
  fm <- ynext~.  # formula

  y.tr <- model.frame(fm, DF.train)[,1]
  x.tr <- model.matrix(fm, DF.train)[,-1]
  x.pr <- model.matrix(fm, DF.pred)[,-1]

  ## OLS
  reg <- lm(fm, data=DF.train)
  ans <- data.frame(ols = predict(reg, DF.pred))
  ## best subset selection
  regfull <- regsubsets(fm, data=DF.train, nvmax=15)
  bhat <- coef(regfull, 2) # Subset Selection 섹션 참조
  newx <- model.matrix(fm, DF.pred) # need (Intercept) too
  ans$bss <- as.numeric(newx[, names(bhat), drop=F] %*% bhat)
  ## ridge
  ridge <- glmnet(x.tr, y.tr, alpha = 0, lambda.min.ratio = 1e-6, nlambda=200)
  ans$ridge <- as.vector(predict(ridge, x.pr, s = cv.ridge2$lambda.min))
  ## lasso
  lasso <- glmnet(x.tr, y.tr, alpha = 1)
  ans$lasso <- as.vector(predict(lasso, x.pr, s = cv.lasso$lambda.min))
  ## pcr
  pcreg <- pcr(fm, data=DF.train, ncomp=15, scale=TRUE)
  ans$pcr <- as.vector(predict(pcreg, DF.pred, ncomp=15)) # PCR 섹션 참조
  ## pls
  plsreg <- plsr(fm, data=DF.train, scale = TRUE)
  ans$pls <- as.vector(predict(plsreg, DF.pred, ncomp = 18)) # PLS 섹션 참조
  ## svm
  svmfit <- svm(fm, data=DF.train, kernel='linear', cost=0.1) # SVM 섹션 참조
  ans$svm <- predict(svmfit, DF.pred, type='response')
  ## random forest
  set.seed(1)
  rf <- randomForest(fm, data=DF.train, mtry=12) # Tree Ensemble 섹션 참조
  ans$rf <- predict(rf, DF.pred, type='r')
  ## return
  ans
}
```

이 함수를 호출할 때에는, 예를 들어 `myfun(z14[group!=1,],
z14[group==1,])`이라고 하면 그룹1을 제외한 그룹2~10의
자료(`z14[group!=1,]`)를 이용하여 훈련한 결과를 그룹1
자료(`z14[group==1,]`)에 적용하여 구한 예측치들을 얻게 된다. 다음과
같이 하여 10개 그룹에 대하여 모두 순차적으로 시행하고 결과를 세로로
쌓는 것이 가능하다.

```R
## Sequential
cvpreds <- NULL
for (fold in 1:10) {
  cat('Work for fold', fold, '\n')
  cvpreds <- rbind(cvpreds, myfun(z14[group!=fold, ], z14[group==fold, ]))
}
```

다만, 이렇게 하고 나면 `cvpreds`의 행 순서가 원래 데이터 행 순서와
다르므로 다음 명령으로써 원래 행 순서를 복원해 준다(잊으면 안 됨).

```R
cvpreds <- cvpreds[rownames(z14), ] # important!
head(cvpreds, 3)
#        ols      bss    ridge    lasso      pcr      pls      svm       rf
# 1 612.0279 600.7213 612.8286 602.8878 614.6901 604.5204 612.0749 592.9033
# 2 581.5426 586.1364 583.4614 575.4188 608.6897 577.8710 575.3670 592.7779
# 3 515.7194 532.9504 525.3181 535.8068 543.0980 512.9284 538.0901 521.3503
```

8개 방법 각각의 CV RMSE는 다음과 같다.

```R
apply(cvpreds, 2, function(x) RMSE(x, z14$ynext)) # sequential
#      ols      bss    ridge    lasso      pcr      pls      svm       rf 
# 51.69413 49.86573 51.73498 50.61905 53.95436 51.61936 52.85023 55.30486 
```

결과 설명에 앞서, 앞의 순차적(sequential)인 시행 방법은 하나의 CPU
코어만을 이용하여 10개의 fold에 대하여 순차적으로 모형 학습과 예측을
시행한다. 코어가 여럿 있는 경우 여러 코어를 사용하여
**병렬(parallel)** 처리가 가능하다. 다음과 같이 하면 된다.

```R
library(foreach)
library(doParallel)

## Prepare for parallel processing
cores <- detectCores() # number of cores
cl <- makeCluster(cores[1]-1) # fast, so redirect output to null device
#cl <- makeCluster(cores[1]-1, outfile="") # do not redirect to null
registerDoParallel(cl)

## Parallel processing
cvpreds <- foreach(fold = 1:10, .combine = rbind, .packages = c('glmnet', 'leaps', 'pls', 'e1071', 'randomForest')) %dopar% {
  cat('Work for fold', fold, '\n') # quiet
  myfun(z14[group!=fold,], z14[group==fold,])
}

## Clean up
stopCluster(cl) # Don't forget this
```

순차 시행에 비하여 병렬 시행의 속도가 2배 정도 빨랐는데, 순차
시행에서도 4초 정도밖에 걸리지 않아 차이를 크게 느끼지는
못했다. 여기서도 행 순서가 뒤죽박죽 되므로 원래 데이터와 동일한 행
순서를 갖도록 재정렬해 주고 각 방법의 CV RMSE를 구하면 다음과 같다.

```R
cvpreds <- cvpreds[rownames(z14), ] # important!
apply(cvpreds, 2, function(x) RMSE(x, z14$ynext)) # CV pred error
#      ols      bss    ridge    lasso      pcr      pls      svm       rf 
# 51.69413 49.86573 51.73498 50.61905 53.95436 51.61936 52.85023 55.30486 
```

앞의 순차(sequential) 시행 시 결과와 동일한 것을 확인할 수 있다.

결과를 보면, [best subset selection](11-subset-selection.md)이 가장 좋으며,
이것이 **Discrete Super Learner**이다. 그 다음은
[lasso](13-ridge-lasso.md)이다.

## Super Learning

다음으로 Super Learning을 해 보자. Super Learner는 실제
목표값(`z14$ynext`)과 8개 방법들의 예측치들 간 차이를 가장 작게 해
주는 8개 예측치들의 선형결합인데, 절편은 없고 그 계수들은 합산하여 1이
되어야 하고 음수가 되면 안 된다. [Nonnegative least squares][nnls]
(NNLS)를 사용할 수 있다.

```R
## Super Learner
library(lsei)
super <- pnnls(as.matrix(cvpreds), z14$ynext, sum=1) # sum=1, nnls
```

추정된 가중치들은 `super$x`에 있으며 그 합은 1이다.

```R
sum(super$x)
# [1] 1
```

가중치들은 다음과 같다.

```R
wgt <- setNames(super$x, colnames(super$r))
wgt
#        ols        bss      ridge      lasso        pcr        pls        svm 
# 0.00000000 0.69281541 0.00000000 0.00000000 0.00000000 0.14967827 0.06447699 
#         rf 
# 0.09302933 
```

CV를 이용하여 구한 Super Learner는 0.693×(Best subset selection) +
0.150×(PLS) + 0.064×(SVM) + 0.093×(Random Forest)이다.

이 Super Learner를 train set CV 예측치에 적용하여 구한 예측치의
RMSE는 다음과 같다.

```R
RMSE(as.matrix(cvpreds) %*% wgt, z14$ynext)
# [1] 49.59206
```

이 RMSE는 앞에서 구한 8개 방법들로부터의 CV 예측치 RMSE들 중 가장 작은
49.86573 (bss)보다 작다. 이는 Super Learner가 RMSE를 최소화하는 convex
weighting이기 때문에 당연한 결과이다.

지금까지 Super Learner 가중치(`wgt`)를 구하였다.  이제 Super Learner를
**test set**에 적용해 보자. 우선 전체 train set (`z14`)을 이용하여
각각의 방법에 대하여 추정을 하고, 이를 test set (`z15`)에 적용하여
예측치를 구한 후, 위 `wgt` 값에 따라 가중평균을 구하면
된다. 가중치(`wgt`)에 의하면 BSS, PLS, SVM, RF만 가중치가 0이 아니므로
이 4개에 대해서만 train을 하면 되겠지만, 시간이 별로 걸리지 않(고
무엇이 0이고 무엇이 0이 아닌지 확인하기도 귀찮)으므로 8개 전부에
대하여 train하고 예측을 한다. 참고로, 위에서는 각 fold별로 train set
(전체의 약 90%)으로 추정을 하고 validation set (전체의 약 10%)에
대하여 예측을 하였는데, 이제는 전체 train set을 이용하여 예측하고 그
결과를 test set에 적용하여 예측을 한다. 앞의 `myfun` 함수를 그대로
사용할 수 있다.

```R
testp <- myfun(z14, z15)
head(testp)
#          ols      bss    ridge    lasso      pcr      pls      svm       rf
# 269 571.8592 591.7046 572.9674 585.3675 581.6550 577.6366 578.7972 589.7740
# 270 597.8058 600.8355 598.3856 590.4644 614.2575 596.4543 608.0084 596.1170
# 271 522.9258 538.2746 531.8178 542.3065 547.7498 526.3882 538.8058 506.8312
# 272 434.6552 461.8369 437.2974 460.2935 455.8359 434.2788 446.4369 448.1672
# 273 421.6089 435.1991 422.1376 436.0459 420.6601 422.0230 424.2278 426.9244
# 274 530.7625 550.6181 529.6336 551.4492 543.1088 535.6742 536.0981 516.9408
```

각각의 test set RMSE는 다음과 같다.

```R
apply(testp, 2, function(x) RMSE(x, z15$ynext))
#      ols      bss    ridge    lasso      pcr      pls      svm       rf 
# 51.38724 48.98381 49.72394 47.75449 47.95138 51.50172 48.99140 50.64245 
```

앞에서 구한 Discrete Super Learner (best subset selection)의 test set
RMSE는 48.98381이다. Super Learner 예측치와 RMSE는 다음과 같다.

```R
superpred <- as.numeric(as.matrix(testp) %*% wgt)
RMSE(superpred, z15$ynext)
# [1] 48.54089
```

앞에서 train set의 CV 예측치의 경우와 달리 test set에서는 개별
방법보다 Super Learner가 반드시 더 좋은 성과를 보여야 할 이유가 없다.
실제로 lasso의 test set RMSE가 47.75449로서 Super Learner의 RMSE
48.54089보다 오히려 더 작다. 이는 이 test set에 대해 해 보니 그렇다는
것이며, 현실에서 실제 적용할 때에는 test란 없으므로 어느 편이 더
나을지 알 수 없다.  CV 예측치의 경우 Super Learner가 더 나았으니까
다른 데이터셋에서도 Super Learner가 더 나을 것이라고 믿을 뿐이다.

[nnls]: https://en.wikipedia.org/wiki/Non-negative_least_squares
