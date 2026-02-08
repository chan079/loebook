데이터는 <a href="10-data.md">여기</a>를 참조하거나 다음 명령 실행.

```R
rm(list=ls(all=TRUE))
load(url("https://github.com/chan079/loebook/raw/main/ml/1-Regression/data.RData"))
```

# Support Vector Regression

Support Vector Regression (SVR)은 원래 분류 문제 해결을 위해 개발된
[SVM]을 회귀 문제에 응용한 것이다. SVM은 [나중](../2-Classification/27-svm.md)에
살펴본다. SVR도 SVM과 마찬가지로 `e1071` 라이브러리의 `svm` 함수를
사용하여 구현할 수 있다. 커널(kernel)로는 linear, polynomial, radial
basis, sigmoid를 사용할 수 있다. 구체적인 함수 형태는 linear는 `u'v`,
polynomial은 `(gamma*u'v+coef0)^degree`, radial basis는
`exp(-gamma*|u-v|^2)`, sigmoid는 `tanh(gamma*u'v+coef0)`이다. 디폴트는
`"radial"`이다. Polynomial 커널의 경우 `gamma` (디폴트는 1/dim),
`coef0` (디폴트는 0), `degree` (디폴트는 3)가 별도 매개변수이다.

Linear 커널을 사용한 SVR 결과는 다음과 같다(cost = 1).

```R
library(e1071)
svmfit0 <- svm(ynext~., data=z14, kernel = 'linear')
summary(svmfit0)

# Call:
# svm(formula = ynext ~ ., data = z14, kernel = "linear")
# 
# 
# Parameters:
#    SVM-Type:  eps-regression
#  SVM-Kernel:  linear
#        cost:  1
#       gamma:  0.05263158
#     epsilon:  0.1
# 
# 
# Number of Support Vectors:  88
```

Train Set의 설명 정도는 다음과 같다.

```R
RMSE(z14$ynext, predict(svmfit0, z14, type='response'))
# [1] 47.35984
```

Test Set 예측 성능은 다음과 같다.

```R
RMSE(z15$ynext, predict(svmfit0, z15, type='response'))
# [1] 51.52811
```

그리 인상적이지 않다.

커널이 주어질 때, [SVM]의 주요 튜닝 매개변수는 `cost`이고, 디폴트 값은
1이다. 이제 여러 cost 값을 고려하고 10-fold [CV]를 이용하여 최적
`cost` 매개변수를 정하고자 한다. Linear 커널을 사용한다.

```R
library(e1071)
set.seed(10)
tune.out <- tune(svm, ynext~., data=z14, kernel = 'linear', ranges = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
options(scipen=10)
summary(tune.out)

# Parameter tuning of ‘svm’:
# 
# - sampling method: 10-fold cross validation
# 
# - best parameters:
#  cost
#     1
# 
# - best performance: 2649.135
# 
# - Detailed performance results:
#      cost     error dispersion
# 1   0.001 20238.847  9812.8138
# 2   0.010  3783.161  1462.0512
# 3   0.100  2898.436  1089.8265
# 4   1.000  2649.135  1025.5508
# 5   5.000  2682.268  1004.7396
# 6  10.000  2682.982   999.6104
# 7 100.000  2684.648  1001.7048
```

위 10-fold CV 결과에 의하면 최적의 `cost` 매개변수는 1이다. 이를 이용하여 Test Set의 목표변수를 예측하여 성과 지표를 구하면 다음과 같다.

```R
RMSE(z15$ynext, predict(tune.out$best.model, z15, type='response'))
# [1] 51.52811
```

[CV]: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
[SVM]: https://en.wikipedia.org/wiki/Support-vector_machine
