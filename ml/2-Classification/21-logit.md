# 로지스틱 회귀

이항반응모형에서 [로지스틱 회귀][logit]가 대표적으로 사용된다.

## <a name="logit-full">Logit: 전체 예측변수 이용</a>

`TrainSet` 데이터셋에서 목표변수(`deny`)를 제외한 나머지 전체 변수들을
예측변수로 사용하여 로지스틱 회귀를 한 결과는 다음과 같다.

```R
full <- glm(deny~., data=TrainSet, family=binomial)
summary(full)
# Call:
# glm(formula = deny ~ ., family = binomial, data = TrainSet)
# 
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.6580  -0.4182  -0.3067  -0.2214   3.0105  
# 
# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -7.16606    0.57680 -12.424  < 2e-16 ***
# dir          4.56694    1.07889   4.233 2.31e-05 ***
# hir         -0.07120    1.29376  -0.055 0.956109    
# lvr          1.81944    0.51468   3.535 0.000408 ***
# ccs          0.30188    0.04108   7.348 2.02e-13 ***
# mcs          0.23081    0.14776   1.562 0.118268    
# pbcryes      1.24497    0.21237   5.862 4.57e-09 ***
# dmiyes       4.45624    0.55904   7.971 1.57e-15 ***
# selfyes      0.58139    0.22057   2.636 0.008394 ** 
# singleyes    0.42718    0.16191   2.638 0.008330 ** 
# uria         0.07359    0.03463   2.125 0.033571 *  
# condominium -0.10736    0.17705  -0.606 0.544277    
# blackyes     0.71092    0.18717   3.798 0.000146 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
#     Null deviance: 1621.8  on 2212  degrees of freedom
# Residual deviance: 1178.3  on 2200  degrees of freedom
# AIC: 1204.3
# 
# Number of Fisher Scoring iterations: 6
```

위 결과에서 “Residual deviance” `1178.3`은 -2 곱하기 최대화된
우도함수값(-2logL)이다. 이 값이 작을수록 train set이 잘 맞춰지는
것이라 생각하면 된다. [AIC] `1204.3`은 -2logL + (2*모수 개수)이다.
다음과 같이 계산해도 똑같은 값을 얻는다.

```R
full$deviance + 2*13
# [1] 1204.26
```

Train set에서 학습이 얼마나 잘 이루어졌는지 살펴보자. 목표변수인
`deny` 변수는 `"no"` 또는 `"yes"`의 `factor` 변수이다.

```R
str(Hmda$deny)
#  Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 2 1 ...
```

첫 번째 나오는 `"no"`가 기본값이고 두 번째 나오는 `"yes"`가
‘설정되었음’을 의미한다. 이에 기초하여 확률 예측값이 0.5보다
크면('작으면'이 아니라) `"yes"`로 예측하도록 목표변수의 이진적
예측값을 구하자. 다음의 첫 번째 줄은 확률을 예측하고 두 번째 줄은
확률예측값이 [0.5]{#half-train}보다 크면 `yes`로 예측한다.

```R
train.phat.full <- predict(full, TrainSet, type='r')
train.denyhat.full <- ifelse(train.phat.full > 0.5, "yes", "no")
```

`deny`의 실제값과 예측값 조합의 빈도를 나타내는 2×2 행렬 [confusion
matrix]를 구하면 다음과 같다.

```R
SummPred(TrainSet$deny, train.denyhat.full)
# $ConfusionMatrix
#       pred
# actual   no  yes
#    no  1921   27
#    yes  182   83
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.3132075   0.9861396   0.7545455   0.9055581 
```

이상의 절차는 '[데이터 준비](20-data.md)' 단원에 만들어 놓은
`Performance()` 함수를 사용해서 간편하게 처리할 수 있다.

```R
Performance(full, TrainSet)
# $ConfusionMatrix
#       pred
# actual   no  yes
#    no  1921   27
#    yes  182   83
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.3132075   0.9861396   0.7545455   0.9055581 
```

결과를 설명하면, 학습 데이터셋에서 실제로 거절되지 않은(truth가 `no`,
negative) 1,948건(=1,921+27) 중 1,921건은 거절되지 않은 것으로 제대로
예측되었고 27건은 거절된 것으로 잘못 예측되었다. 그러므로 false
positive rate ([FPR][evalbin], 1 - specificity, 실제 거절되지 않은 것
중 거절되었다고 잘못 예측된 비율)은 27/1948 = 0.0139이고
특정도([specificity], 실제 거절되지 않은 것 중 올바르게 예측된 비율,
1-FPR)는 1-0.0139=0.9861이다.  거절된(`yes`, positive) 265건 중
올바르게 예측한 것은 83건에 불과하고 나머지 182건은 잘못
맞추었다. True positive rate ([TPR][evalbin], 민감도 [sensitivity],
거절된 것 중 올바르게 예측한 비율)은 83/265 = 0.3132에 지나지
않는다. 또, 총 2,213건 중 제대로 맞춘 것(대각 원소)은 1921 + 83 =
2,004건이고, 제대로 맞추지 못한 것은 209건이다.  오차율(1-accuracy,
전체 중 잘못 예측한 비율)은 209/2213 = 0.0944, 적중률([accuracy], 전체
중 올바르게 예측한 비율)은 0.9056이다.  정밀도([precision], 거절이라고
예측된 것 중 실제로 거절된 건의 비율)는 83/(27+83) = 0.7545이다.

<pre>
<b>학습 데이터셋 예측 성과지표</b>
Sensitivity = 83/(182+83) = 0.3132 = True Positive Rate = Recall
Specificity = 1921/(1921+27) = 0.9861 = 1 - False Positive Rate
Precision   = 83/(27+83)
Accuracy = (1921+83)/(1921+27+182+83) = 0.9056 = 1 - Error rate
Error rate  = (27+182)/(1921+27+182+83) = 0.0944 = 1 - Accuracy
</pre>

오차율(1-Accuracy) 자체만 보면 괜찮아 보이나 [sensitivity]가 너무
낮다.  이는 전반적으로 `yes` 예측을 하지 못함을 의미한다. 그 이유는
전체 데이터셋에 `yes`의 비율이 낮아(약 11.2% =
(182+83)/(1921+27+182+83)), 로짓 알고리즘에서 `yes`를 맞추는 것이
등한시되기 때문이다. 이 문제는 나중에 [ROC 곡선](22-roc.md) 부분과
[Class Imbalance](23-imbalance.md) 부분에서 자세히 다루기로 한다.

이처럼 이 예측모델은 train set에 대해서도 positive를 positive로 제대로
예측해 주지 못하니([sensitivity]가 너무 낮음) test set에 대해서도
`yes`를 `yes`로 예측할 빈도가 낮을 것으로 예상할 수 있다. 정말 그런지
한번 해 보자. [테스트셋]{#half-test}에 대해 ‘설정’될(`yes`가 될)
확률을 우선 예측한 후, 이 확률이 0.5보다 크면 `yes`로, 0.5보다 작거나
같으면 `no`로 예측한다. 그리고 나서 [confusion matrix]를 구하면 다음과
같다.

```R
Performance(full, TestSet)
# $ConfusionMatrix
#       pred
# actual  no yes
#    no  146   1
#    yes  15   5
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.2500000   0.9931973   0.8333333   0.9041916 
```

예상대로 “no”를 “no”로 제대로 예측한 비율인 [specificity]는 146 /
(146 + 1) = 0.9932로 매우 높지만, “yes”를 “yes”로 제대로 예측한 비율인
[sensitivity] ([true positive rate][evalbin])는 5/(15+5) = 0.25로 매우
낮다. 학습한 모델이 대체로 거절이 이루어지지 않는다(`no`)고 예측하기
때문이다. 표본에 negative의 비율이 높고(약 88%가 `no`) negative를
negative라고 잘 예측하기 때문에 전반적인 예측 정확도([accuracy])는
(146+5)/(146+1+15+5) = 0.9042로 상당히 높다. 하지만 이는 당초부터
테스트셋에서 `no`인 경우가 대부분이기 때문이지 모델의 변별력이 특별히
좋기 때문은 아니다. 그냥 항상 `no`라고 예측을 하여도 88%를 맞춘다.

[Confusion matrix]와 관련된 통계들은 `caret` 패키지의 <a
href="https://www.rdocumentation.org/packages/caret/topics/confusionMatrix"
target="_blank">`confusionMatrix()`</a> 함수로도 간편하게 구할 수
있다.  단, 앞에서는 ‘실제’가 세로, ‘예측’이 가로임에 반하여, 이
`caret` 패키지 함수에서는 ‘실제’가 가로, ‘예측’이 세로로 열거되어
있음에 유의하라. [ROC] 곡선(yes와 no 예측 시 확률 경계값을 0에서 1까지
바꾸면서 sensitivity와 1-specificity를 표시한 그림) 등에 대해서는
[다음 소절](22-roc.md)을 참조하라.

## Logit: Backward Stepwise Selection

[AIC] 기준 최적 모형의 backward stepwise selection을 해 보자.
[A. Kassambara의 포스트][1]와 [토론토 대학 Jerry Brunner 교수가
작성하였다고 생각되는 pdf 문서][Brunner]에서 도움을 얻었다.

[1]: http://www.sthda.com/english/articles/36-classification-methods-essentials/150-stepwise-logistic-regression-essentials-in-r/
[Brunner]: http://www.utstat.toronto.edu/~brunner/oldclass/appliedf11/handouts/2101f11StepwiseLogisticR.pdf

```R
backward <- step(full, direction = 'backward', trace = 0)
backward
# Call:  glm(formula = deny ~ dir + lvr + ccs + mcs + pbcr + dmi + self + 
#     single + uria + black, family = binomial, data = TrainSet)
# 
# Coefficients:
# (Intercept)          dir          lvr          ccs          mcs      pbcryes  
#    -7.19496      4.54731      1.81017      0.30115      0.22937      1.24642  
#      dmiyes      selfyes    singleyes         uria     blackyes  
#     4.46472      0.58426      0.40419      0.07597      0.69080  
# 
# Degrees of Freedom: 2212 Total (i.e. Null);  2202 Residual
# Null Deviance:	    1622 
# Residual Deviance: 1179 	AIC: 1201
```

이 기준에 의하여 최종 선택된 모형은 10개 변수로 이루어진 모형이다.
Full model로부터 제외된 2개의 변수는 `hir` (housing expenses to income
ratio)과 `condominium` (is unit a condominium?)이다. 선택된 모형의
residual deviance는 `1179`로서 full model의 `1178.3`보다 약간 크나(즉,
train set 설명력은 약간 떨어지나), AIC는 `1201`로서 full model의
`1204.3`보다 약간 더 작다(AIC는 작을수록 예측 성능이 좋은 것이라는
믿음이 있다).

테스트셋에 적용하여 [confusion matrix]를 만들자.

```R
Performance(backward, TestSet)
# $ConfusionMatrix
#       pred
# actual  no yes
#    no  146   1
#    yes  15   5
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.2500000   0.9931973   0.8333333   0.9041916 
```

결과는 full logit의 경우와 동일하다(다른 데이터에서는 그렇지 않을 수도
있다).

## Logit: Forward Stepwise Selection

[AIC]에 기반한 forward stepwise selection을 하자([토론토 대학 Jerry
Brunner 교수가 작성하였다고 생각되는 pdf 문서][Brunner]에서 도움을
얻었음). 이를 위해서는 가장 작은 모형과 가장 큰 모형을 지정해 주어야
한다. 가장 작은 모형은 null model (우변에 상수항만 있는
모형)이다. Forward stepwise selection을 위해 우선 null model을
추정하자(`step` 명령이 이걸 요구하니까).

```R
null <- glm(deny~1, data=TrainSet, family=binomial)
```

이제 forward stepwise selection을 하자.

```R
step(null, scope = list(formula(null), upper = formula(full)), direction = 'forward', trace = 0)

# Call:  glm(formula = deny ~ dmi + ccs + dir + pbcr + black + lvr + self + 
#     single + uria + mcs, family = binomial, data = TrainSet)
# 
# Coefficients:
# (Intercept)       dmiyes          ccs          dir      pbcryes     blackyes  
#    -7.19496      4.46472      0.30115      4.54731      1.24642      0.69080  
#         lvr      selfyes    singleyes         uria          mcs  
#     1.81017      0.58426      0.40419      0.07597      0.22937  
# 
# Degrees of Freedom: 2212 Total (i.e. Null);  2202 Residual
# Null Deviance:	    1622 
# Residual Deviance: 1179 	AIC: 1201
```

Backward stepwise selection과 비교할 때 변수들의 순서만 다르고 선택된
변수들의 집합은 서로 동일하다.

## Logit: Backward와 forward 결합

Backward stepwise selection과 forward stepwise selection을 결합할 수도
있다.

```R
step(null, scope = list(lower = formula(null), upper = formula(full)), direction = 'both', trace = 0)

# Call:  glm(formula = deny ~ dmi + ccs + dir + pbcr + black + lvr + self + 
#     single + uria + mcs, family = binomial, data = TrainSet)
# 
# Coefficients:
# (Intercept)       dmiyes          ccs          dir      pbcryes     blackyes  
#    -7.19496      4.46472      0.30115      4.54731      1.24642      0.69080  
#         lvr      selfyes    singleyes         uria          mcs  
#     1.81017      0.58426      0.40419      0.07597      0.22937  
# 
# Degrees of Freedom: 2212 Total (i.e. Null);  2202 Residual
# Null Deviance:	    1622 
# Residual Deviance: 1179 	AIC: 1201
```

결과는 forward stepwise selection 및 backward stepwise selection과 동일하다.

# 판별분석(Discriminant Analysis)

판별분석(Discriminant analysis) 중 [Linear discriminant analysis][LDA]
(LDA)와 [Quadratic discriminant analysis][QDA] (QDA)를
살펴보자. [James et al. (2013)][book]의 4장을 참고하였다. 참고로,
LDA와 QDA는 예측변수(X)들이 [정규분포][normaldist]를 갖는다는
가정하에서 도출되는데, 이 예제에서는 더미변수들이 X변수로 포함되어
있고 더미변수가 정규분포를 가질 리는 만무하므로 정규분포 가정은
만족되지 않는다. 하지만 그렇다고 하여 LDA와 QDA를 사용해서는 안 된다는
법도 없으므로 실습해 보겠다. 참고로 Stack Exchange에 [이런
글타래][SEdum]도 있다.  이하에서 LDA와 QDA의 이론에 대하여는 길게
설명하지 않겠다.

[normaldist]: https://en.wikipedia.org/wiki/Normal_distribution {target="_blank"}
[SEdum]: https://stats.stackexchange.com/questions/158772/can-we-use-categorical-independent-variable-in-discriminant-analysis {target="_blank"}

## [Linear Discriminant Analysis]{#LDA}

LDA는 피예측변수(`deny`)가 `yes`인 경우와 `no`인 경우에 X변수들의
평균이 서로 다르고 분산공분산행렬은 동일한 정규분포를 갖는다고
가정한다. 이 가정에서 도출되는 바에 의하면 `yes`와 `no`의 경계면이
예측변수(X변수)들에 대하여 선형이다. `MASS` 라이브러리의 `lda()`
함수를 이용한다.

```R
library(MASS)
lda.fit <- lda(deny~., data=TrainSet)
lda.fit
# Call:
# lda(deny ~ ., data = TrainSet)
# 
# Prior probabilities of groups:
#        no       yes 
# 0.8802531 0.1197469 
# 
# Group means:
#           dir       hir       lvr      ccs      mcs    pbcryes      dmiyes
# no  0.3231376 0.2507864 0.7272820 1.963039 1.700719 0.04671458 0.002053388
# yes 0.3909430 0.2928115 0.8176361 3.324528 1.883019 0.27169811 0.154716981
#       selfyes singleyes     uria condominium  blackyes
# no  0.1119097 0.3814168 3.736910   0.2864476 0.1160164
# yes 0.1622642 0.4943396 4.046793   0.3245283 0.3320755
# 
# Coefficients of linear discriminants:
#                     LD1
# dir          3.13808213
# hir         -0.32409794
# lvr          0.67157884
# ccs          0.21987557
# mcs          0.11490580
# pbcryes      1.43952374
# dmiyes       4.91088057
# selfyes      0.36114631
# singleyes    0.25441415
# uria         0.04145506
# condominium -0.06456575
# blackyes     0.59438727
```

`table(TrainSet$deny)`를 보면 알 수 있듯이 `TrainSet`의 `deny` 변수는
1,948개 관측치에서 `no`, 265개 관측치에서 `yes`이다. 즉, 전체의
88.0%에서 `no`, 12.0%에서 `yes`이다.  이것이 위 결과의 “Prior
probabilities of groups”으로 제시되어 있다. 그 아래에는 추정 결과들이
적당한 선에서 열거되어 있다.

`TestSet`에 대하여 예측하고 confusion matrix를 만들자.

```R
Performance(lda.fit, TestSet)
# $ConfusionMatrix
#       pred
# actual  no yes
#    no  145   2
#    yes  14   6
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.3000000   0.9863946   0.7500000   0.9041916 
```

위 ‘[Logit: 전체 예측변수 이용](#logit-full)’ 소절로부터의 결과와
비교하자면, full logistic regression에서 sensitivity가 0.25
(=5/20)였다. 여기 LDA에서는 근소하게 상승하나(6/(14+6)=0.3) 낮은 것은
여전히 마찬가지이다.

## [Quadratic Discriminant Analysis]{#QDA}

Quadratic discriminant analysis (QDA)에서는 X의 평균뿐 아니라
분산공분산행렬도 피예측변수가 `no`인 집단과 `yes`인 집단 간에 서로
다를 수 있다고 가정한다. 이 가정하에서 경계면이 feature들의 2차함수가
된다(그래서 “QDA”). `MASS` 라이브러리의 `qda()` 함수를 이용한다.

```R
library(MASS)
qda.fit <- qda(deny~., data=TrainSet)
qda.fit
# Call:
# qda(deny ~ ., data = TrainSet)
# 
# Prior probabilities of groups:
#        no       yes 
# 0.8802531 0.1197469 
# 
# Group means:
#           dir       hir       lvr      ccs      mcs    pbcryes      dmiyes
# no  0.3231376 0.2507864 0.7272820 1.963039 1.700719 0.04671458 0.002053388
# yes 0.3909430 0.2928115 0.8176361 3.324528 1.883019 0.27169811 0.154716981
#       selfyes singleyes     uria condominium  blackyes
# no  0.1119097 0.3814168 3.736910   0.2864476 0.1160164
# yes 0.1622642 0.4943396 4.046793   0.3245283 0.3320755
```

결과를 테스트셋에 적용하면 다음과 같다.

```R
Performance(qda.fit, TestSet)
# $ConfusionMatrix
#       pred
# actual  no yes
#    no  138   9
#    yes  13   7
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.3500000   0.9387755   0.4375000   0.8682635 
```

Sensitivity는 full logit = 5/20 &lt; LDA = 6/20 &lt; QDA =
7/20이다. 하지만 0.35는 여전히 낮은 값이다.

[logit]: https://en.wikipedia.org/wiki/Logistic_regression
[AIC]: https://en.wikipedia.org/wiki/Akaike_information_criterion
[confusion matrix]: https://en.wikipedia.org/wiki/Confusion_matrix
[evalbin]: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
[sensitivity]: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
[specificity]: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
[accuracy]: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
[precision]: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
