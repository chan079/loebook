# 스플라인(Splines)

실습용 데이터 준비는 [여기](10-data.md) 참조.

## Natural Cubic Spline

[Best subset selection](11-subset-selection.md)을 이용하면 [BIC]를 사용하든
[CV]를 사용하든 똑같이 2개의 변수(`deathrate`와 `aged`)가
선택되었다. 이 두 변수만을 사용하여 회귀를 하는데(머신러닝이 아니라
휴먼러닝이 많이 가미되었다), 이제는 3차 natural [spline] (NS)을
활용하여 비선형성을 허용해 보자.

3차 NS를 사용하여 학습할 때 어느 정도 성과를 얻는지 보자. 먼저 두
변수의 `df`를 모두 3 (자의적인 선택) 정도로 하고 살펴보자(`df=3`은
가운데에 2개의 구분점이 있는 NS를 말한다).

```R
library(splines)
reg.ns <- lm(ynext~ns(deathrate,3)+ns(aged,3), data=z14)
RMSE(z15$ynext, predict(reg.ns, z15))
# [1] 46.85395
rmspe.rw
# [1] 53.24273
```

[Best subset selection](11-subset-selection.md)에 의하여 선택된 2개의 변수를
선형으로 사용한 경우(RMSE = 48.98381)에 비하여 시험 데이터의 예측
정확성에 개선이 이루어졌다. 이제 두 변수의 `df`를 Adjusted R제곱, [AIC],
[BIC]를 이용하여 선택해 보자. 먼저 두 변수의 df를 각각 1~4로 설정하는
16개의 셋팅을 만들어 `dfset`이라고 하자. `df1`과 `df2`는 각각
`deathrate`와 `aged`용 df에 해당한다.

```R
dfset <- expand.grid(df1 = 1:4, df2 = 1:4)
dfset
#    df1 df2
# 1    1   1
# 2    2   1
# 3    3   1
# 4    4   1
# 5    1   2
# 6    2   2
# 7    3   2
# 8    4   2
# 9    1   3
# 10   2   3
# 11   3   3
# 12   4   3
# 13   1   4
# 14   2   4
# 15   3   4
# 16   4   4
```

이제 `for` 반복문을 이용하여 각각의 셋팅에 해당하는 NS 회귀를 하고
Adjusted R-squared, AIC, BIC를 기록한다.

```R
dfset$bic <- dfset$aic <- dfset$adj.r2 <- NA
for (i in 1:nrow(dfset)) {
  reg1 <- lm(ynext~ns(deathrate,dfset$df1[i])+ns(aged,dfset$df2[i]), data=z14)
  dfset$adj.r2[i] <- summary(reg1)$adj.r.squared
  dfset$aic[i] <- AIC(reg1)
  dfset$bic[i] <- BIC(reg1)
}
```

어느 셋팅에서 `adj.r2`, `aic`, `bic`가 각각 최적이 되는지 보자.

```R
dfset[which.max(dfset$adj.r2),]
#   df1 df2    adj.r2      aic      bic
# 6   2   2 0.9788745 2375.844 2396.287
dfset[which.min(dfset$aic),]
#   df1 df2    adj.r2      aic      bic
# 6   2   2 0.9788745 2375.844 2396.287
dfset[which.min(dfset$bic),]
#   df1 df2    adj.r2      aic      bic
# 1   1   1 0.9784861 2377.944 2391.572
```

Adjusted R-squared와 AIC에 의할 때 두 변수의 `df`가 모두 2일 때
최적이다. BIC에 의하면 선형모형(두 변수 df가 모두 1)이 최적이다. BIC
이용 결과(선형회귀)는 이미 앞에서 보았으므로, 이번에는 AIC를 활용하여 df를 모두
2로 설정하자. 회귀 결과를 2015년 테스트 셋에 적용한 예측의 결과는
다음과 같다.

```R
reg.ns <- lm(ynext~ns(deathrate, 2) + ns(aged, 2), data=z14)
RMSE(z15$ynext, predict(reg.ns, z15))
# [1] 46.77627
rmspe.rw
# [1] 53.24273
```

`df=3` 경우(테스트셋 RMSE는 46.85395)에 비해 RMSE가 줄어든 정도가
미미하여 큰 의미는 없어 보이나, 아무튼 줄어들었다.

### Cross Validation

[CV]를 통하여 두 변수의 df를 결정하자. 10-fold CV를 위해 우선
관측치(행)들을 무작위로 1~10군으로 할당하자.

```R
set.seed(1)
group <- sample(1:10, nrow(z14), replace = TRUE)
```

위에서와 마찬가지로 `df1`(`deathrate` 용)과 `df2`(`aged` 용)을 1~4까지
할당한 `dfset`을 만들고 각 셋팅마다 해당 df들을 이용하여 구한 10-fold
CV 예측오차 제곱합의 합계를 저장한다.

```R
dfset <- expand.grid(df1 = 1:4, df2 = 1:4, cv.error = NA)
for (i in 1:nrow(dfset)) {
  cv.err <- 0
  df1 <- dfset$df1[i]
  df2 <- dfset$df2[i]
  for (k in 1:10) {
    reg <- lm(ynext~ns(deathrate, df1) + ns(aged, df2), data=z14, subset = group != k)
    zk <- z14[group==k, ]
    cv.err <- cv.err + sum((zk$ynext - predict(reg, zk))^2)
  }
  dfset$cv.error[i] <- cv.err
}
```

CV error가 가장 작은 셋팅은 다음과 같다.

```R
dfset[which.min(dfset$cv.error),]
#   df1 df2 cv.error
# 1   1   1 559499.5
```

위에서 BIC를 사용한 경우와 똑같은 결과를 얻었다. NS (natural
[spline])에서 df가 1이면 선형모형과 똑같으므로 [앞에서](11-subset-selection.md)
얻은 결과와 똑같겠지만, 편의상 (앞으로 돌아가서 확인하는 수고를 하고
싶지 않으므로) 재실행해 보면 결과는 다음과 같다.

```R
reg.ns2 <- lm(ynext~ns(deathrate,1) + ns(aged,1), data=z14) # final model fit
RMSE(z15$ynext, predict(reg.ns2, z15))
# [1] 48.98381
```

주의할 점이 있다. 지금 이 실습에서는 2016년 사망률이 있어서 예측의
성능을 평가해 볼 수 있지만, 실제 적용 시에는 성능을 확인할 방법이
없다. 그러므로 1차함수가 좋은지 knot가 1개씩 있는 NS가 좋은지 아니면
다른 것이 좋은지 바로 알 방법이 없다. 시간이 지나서 미래가 과거로
변하면 그때서야 판단할 수 있다.

또한 R의 `ns`(자연 3차 스플라인)와 `bs`(스플라인) 함수는 끝점과 중간
구분점을 변수값 분포에 따라 자동으로 설정한다. 학습용 데이터와
테스트용 데이터에서 변수들의 분포가 달라서, 학습용 데이터와 테스트용
데이터에서 구분점들의 개수는 동일할지라도 구분점의 위치들이
상이하다. 즉, 학습자료 `z14`에서 `ns(deathrate, 2)`라고 할 때와 시험용
자료 `z15`에서 `ns(deathrate, 2)`라고 할 때의 basis들이
상이하다. 이에, 구분점들을 학습용 데이터의 변수 분포에 따라 아예
확정해 버리는 방법도 고려할 만하다. (계량경제학 전공자의) 상식에
기대어 생각해 보면 미리 확정하는 것이 직관적이지만, 알고리즘은 잘
정의되어 있으므로 위와 같이 R로 하여금 자동으로 정하도록 하는 것도
문제될 것은 없다는 생각도 든다.

전체 코드는 다음과 같다.

```R
library(splines)
reg.ns <- lm(ynext~ns(deathrate,3)+ns(aged,3), data=z14)
RMSE(z15$ynext, predict(reg.ns, z15))
rmspe.rw
dfset <- expand.grid(df1 = 1:4, df2 = 1:4)
dfset
dfset$bic <- dfset$aic <- dfset$adj.r2 <- NA
for (i in 1:nrow(dfset)) {
  reg1 <- lm(ynext~ns(deathrate,dfset$df1[i])+ns(aged,dfset$df2[i]), data=z14)
  dfset$adj.r2[i] <- summary(reg1)$adj.r.squared
  dfset$aic[i] <- AIC(reg1)
  dfset$bic[i] <- BIC(reg1)
}
dfset[which.max(dfset$adj.r2),]
dfset[which.min(dfset$aic),]
dfset[which.min(dfset$bic),]
reg.ns <- lm(ynext~ns(deathrate, 2) + ns(aged, 2), data=z14)
RMSE(z15$ynext, predict(reg.ns, z15))
rmspe.rw
set.seed(1)
group <- sample(1:10, nrow(z14), replace = TRUE)
dfset <- expand.grid(df1 = 1:4, df2 = 1:4, cv.error = NA)
for (i in 1:nrow(dfset)) {
  cv.err <- 0
  df1 <- dfset$df1[i]
  df2 <- dfset$df2[i]
  for (k in 1:10) {
    reg <- lm(ynext~ns(deathrate, df1) + ns(aged, df2), data=z14, subset = group != k)
    zk <- z14[group==k, ]
    cv.err <- cv.err + sum((zk$ynext - predict(reg, zk))^2)
  }
  dfset$cv.error[i] <- cv.err
}
dfset[which.min(dfset$cv.error),]
reg.ns2 <- lm(ynext~ns(deathrate,1) + ns(aged,1), data=z14) # final model fit
RMSE(z15$ynext, predict(reg.ns2, z15))
```
