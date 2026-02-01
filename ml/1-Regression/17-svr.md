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
# Number of Support Vectors:  86
```

Train Set의 설명 정도는 다음과 같다.

```R
RMSE(z14$ynext, predict(svmfit0, z14, type='response'))
# [1] 47.33744
```

Test Set 예측 성능은 다음과 같다.

```R
RMSE(z15$ynext, predict(svmfit0, z15, type='response'))
# [1] 55.02433
```

전혀 인상적이지 않다.

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
#   0.1
# 
# - best performance: 2710.426 
# 
# - Detailed performance results:
#      cost     error dispersion
# 1   0.001 25951.266 10290.9325
# 2   0.010  3961.079  1217.0956
# 3   0.100  2710.426   980.4872
# 4   1.000  2743.499  1006.4024
# 5   5.000  2742.039  1131.7863
# 6  10.000  2791.526  1209.1412
# 7 100.000  2806.500  1248.2878
```

위 10-fold CV 결과에 의하면 최적의 `cost` 매개변수는 0.1이다(random
seed를 1로 하면 5를 선택하고, 또 다른 random seed에서는 결과가 또
다르다). 이를 이용하여 Test Set의 목표변수를 예측하여 성과 지표를
구하면 다음과 같다.

```R
RMSE(z15$ynext, predict(tune.out$best.model, z15, type='response'))
# [1] 48.9914
```

[SVM]: https://en.wikipedia.org/wiki/Support-vector_machine
[CV]: https://en.wikipedia.org/wiki/Cross-validation_(statistics)