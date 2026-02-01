데이터는 <a href="20-data.md">여기</a> 참조.

# Class Imbalance 문제

Class imbalance란 목표변수 클래스 크기가 서로간에 크게 다른 것을
일컫는다.  이 단원 주제에 관해서는 [Wikipedia 항목][ovun], [Karthe
(2016)][Karthe2016], [Jason Brownlee (2021)][7], R bloggers에
[finnstats (2021)][8] 등을 참고하였다.

[Karthe2016]: https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/ "Karthe (2016)"
[7]: https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
[8]: https://www.r-bloggers.com/2021/05/class-imbalance-handling-imbalanced-data-in-r/

Train set에서 `yes`와 `no`의 비율이 크게 다르다. `no`가 88%이다.

```R
mean(TrainSet$deny=='no')
# [1] 0.8802531
```

이 경우 `no`를 잘 맞추면 전체적으로 잘 맞추게 되기 때문에, logit
추정에서는 결과적으로 `no` 클래스의 올바른 예측이 중요하다.
전체적으로 잘 맞추는 것(오차율을 최소화하는 것)이 목적이면 이 방법은
좋은 방법이다.  하지만 때에 따라 sensitivity와 specificity를 골고루
좋게 만들고 싶은 경우도 있다. 이 문제는 [앞 단원에서 Youden Index를
최대화하는 cutoff 값을 사용함으로써](22-roc.md) 부분적으로 해결할 수
있었다. 하지만 이는 모형을 추정하고 나서 확률 cutoff 값을 조정하는
것으로서, 모형 추정 시에는 class imbalance 문제를 고려하지 않는다.  이
단원에서는 추정 시에 class imbalance 문제를 명시적으로 고려하는 방법을
다룬다.

한 가지 방법은 더 작은 클래스에 높은 가중치를 부여하여
훈련(추정)방법으로 하여금 두 클래스의 오류를 유사한 정도로 심각하게
고려하도록 해 주는 것이다(가중 회귀). 로지스틱 회귀의 경우에 이것이
가능하다. 이와 유사하지만 더 확장성 있는 방법은 아예 표본에서 클래스
간 균형을 맞추는 것이다.  이 방법은 logistic regression,
[LDA](21-logit.md#LDA), [QDA](21-logit.md#QDA), lasso logit, neural
network 등등 모든 모형에 똑같이 적용할 수 있다. 데이터 생성법만 알면
되므로, 편의상 설명은 [전체 변수 활용 logistic
regression](21-logit.md#logit-full)에 국한하고자 한다.

## Weighted regression

문제는 `yes` 클래스 관측치 수가 `no` 클래스 관측치 수에 비하여 너무
적다는 것이다(전체의 12%). 그러다 보니 결과적으로 `no` 클래스만 잘
맞추어도 전체적으로 오차율이 작아진다.  로지스틱 회귀의 경우, 이에
대한 해결책으로 간단하게 생각해볼 수 있는 방법은 `yes` 클래스에 높은
가중치를 주는 것이다.  Train set에서 `yes`의 비율이 전체의 12%밖에
되지 않으므로 표본에서 `no` 대 `yes`의 비율은 약 88:12의
비율이다. 그러므로 `no` 관측치 하나에 1의 가중치를 준다면 `yes` 관측치
하나에는 88/12의 가중치를 주어 전체적으로 `no` 클래스와 `yes` 클래스가
동일한 비중을 갖도록 만든다.  그런데 R의 로짓 회귀를 위한 `glm` 함수는
가중치가 자연수가 아니면 경고 메시지를 준다.  이 경고 메시지가 보기
싫어서 자연수 가중치를 주고자 하는데, 1:(88/12)는 3:22이므로 `no`
관측치들에 3의 가중치를 주고 `yes` 관측치들에 22의 가중치를 주는
가중치 벡터 `wgt`를 만들어 사용하자.

```R
table(TrainSet$deny)
#   no  yes 
# 1948  265 
wgt <- ifelse(TrainSet$deny=='no', 3, 22)  # 265:1948 ~= 3:22
```

이 `wgt`를 가중치로 사용하여 로짓 회귀를 하고 이를 이용하여 train
set의 예측 성능을 살펴보면 결과는 다음과 같다.

```R
wlogit <- glm(deny~., family = binomial, data=TrainSet, weights = wgt)
Performance(wlogit, TrainSet)
# $ConfusionMatrix
#       pred
# actual   no  yes
#    no  1586  362
#    yes   79  186
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.7018868   0.8141684   0.3394161   0.8007230
```

`yes`의 비중이 커졌으므로 `yes`를 잘 맞추려는 유인이 생겼고, sensitivity
(`yes`를 `yes`로 제대로 예측한 비율)와 specificity (`no`를 `no`로
제대로 예측한 비율)가 더 균형잡혀 있다.  Sensitivity는 당초 31.3%에서
70.2%로 상승했다.  그 대가로 specificity는 당초 98.6%에서 81.4%로
하락했다.  자연스러운 결과이다.  비용 없이 얻기만 하는 것은 없다.

위 추정 결과를 test set에 적용한 결과는 다음과 같다.

```R
Performance(wlogit, TestSet)
# $ConfusionMatrix
#       pred
# actual  no yes
#    no  125  22
#    yes   8  12
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.6000000   0.8503401   0.3529412   0.8203593 
```

Sensitivity (`yes` 중 `yes`로 예측된 비율)는 0.6으로 상승했고(원래는
0.25) specificity (`no` 중 `no`로 예측된 비율)는 0.85로
하락했다(원래는 0.99).

### Weighted regression과 관측치 중복의 동일성

참고로, 자연수 가중치를 주는 방법은 관측치들을 중복 사용하는 것과
동일하다. `no` 대 `yes`에 3:22의 가중치를 주는 것은 `no` 관측치를 3번
반복하고 `yes` 관측치를 22번 반복하는 것과 같다.  이렇게 데이터를
변경하여 로짓 추정을 해 보자.

```R
## 관측치 중복 사용하여 클래스 균형화
rn <- rownames(TrainSet)
dup.idx <- c(rep(rn[TrainSet$deny=='no'], 3), rep(rn[TrainSet$deny=='yes'], 22))
dup.data <- TrainSet[dup.idx, ]
dup.logit <- glm(deny~., family = binomial, data=dup.data)
cbind(wlogit$coef, dup.logit$coef)
#                    [,1]        [,2]
# (Intercept) -4.91146023 -4.91146023
# dir          4.71383980  4.71383980
# hir         -0.75016224 -0.75016224
# lvr          1.61341082  1.61341082
# ccs          0.31082364  0.31082364
# mcs          0.18359830  0.18359830
# pbcryes      1.20448993  1.20448993
# dmiyes       4.54125726  4.54125726
# selfyes      0.65767243  0.65767243
# singleyes    0.42254303  0.42254303
# uria         0.08365357  0.08365357
# condominium -0.07670152 -0.07670152
# blackyes     0.80821088  0.80821088
```

위 계수 비교 결과에서 첫 번째 열은 앞의 가중 로짓 회귀(`wlogit`)
결과이고 두 번째 열은 데이터를 중복 사용한 후 로짓 회귀(`dup.logit`)를
한 결과인데, 둘이 완전히 일치함을 확인할 수 있다.

지금까지 가중치를 주는 방법을 고려하였고, 이 방법이 데이터를 중복
사용하는 것과 동일한 것임을 보았다. 이하에서는 무작위 추출을 통하여
데이터셋에서 클래스를 균형화하는 방법을 살펴본다.

## Random oversampling

수가 적은 클래스인 `yes` 관측치들로부터 무작위 복원추출을 하여
클래스를 균형화하는 것([random oversampling][ovun])도 가능하다. Train
set에서 `yes`와 `no` 개수를 보자.

```R
summary(TrainSet$deny)
#   no  yes 
# 1948  265 
```

`no` 클래스 관측치 수가 1,948개이므로 `yes` 클래스 265개로부터
1,948개를 복원추출하자.  이는 `yes` 클래스 관측치 1개가 평균 1,948/265
= 7.35번 중복되어 사용되는 것과 같은 효과를 가지며, 이는 `yes`
클래스에 7.35의 가중치를 주는 것과 비슷하다(완전히 똑같지는 않는데, 그
이유는 7.35가 정수가 아니고 무작위 추출이기 때문이다). 다음 코드는
이렇게 `yes` 클래스 관측치들을 무작위 복원추출하여 균형화된 데이터셋을
만든다.

```R
set.seed(1)
idx1 <- which(TrainSet$deny=='no')  # "no"
idx2 <- with(TrainSet, sample(which(deny=='yes'), sum(deny=='no'), replace=TRUE)) # to duplicate "yes"
Over <- TrainSet[c(idx1,idx2),]
summary(Over$deny)
#   no  yes 
# 1948 1948 
```

이제 `yes`와 `no` 클래스의 표본크기가 같다.

참고로, [Lunardon, Menardi, and Torelli (2014)][11]의 [ROSE] (Random
Over-Sampling Examples) 패키지의 `ovun.sample` 명령을 사용해서 똑같이
구할 수도 있다. 패키지에 관한 자세한 설명은 [논문][12] (The R Journal,
2014, 6:1, 79-89)을 참조하라. 사용자 측면에서 `ROSE` 패키지는 사용하기
매우 간편하다(반면 아래 [SMOTE](#SMOTE) 용 [smotefamily] 패키지는
사용하기 매우 번거롭다).

[11]: https://journal.r-project.org/archive/2014/RJ-2014-008/index.php
[12]: https://journal.r-project.org/archive/2014/RJ-2014-008/RJ-2014-008.pdf

```R
library(ROSE)
n.max <- max(table(TrainSet$deny))
Over2 <- ovun.sample(deny~., TrainSet, method="over", N=n.max*2, seed=1)$data
summary(Over2$deny)
#   no  yes 
# 1948 1948
mapply(identical, Over, Over2)
#         dir         hir         lvr         ccs         mcs        pbcr 
#        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
#         dmi        self      single        uria condominium       black 
#        TRUE        TRUE        TRUE        TRUE        TRUE        TRUE 
#        deny 
#        TRUE 
```

마지막 명령 결과를 보면 앞에 직접 무작위 복원추출하여 균형화한
결과(`Over`)와 `ROSE` 패키지를 사용하여 over-sample한 결과(`Over2`)의
변수들이 동일함을 알 수 있다. 단, random seed가 다르면 다른 데이터셋이
됨에 유의하라.

이 균형화된 데이터를 분석하자. 로짓 회귀를 한다.

```R
logit.over <- glm(deny~., family = binomial, data = Over)
```

Train set에서 성능을 보자(학습에 사용한 `Over`가 아님).

```R
Performance(logit.over, TrainSet)
# $ConfusionMatrix
#       pred
# actual   no  yes
#    no  1592  356
#    yes   79  186
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.7018868   0.8172485   0.3431734   0.8034343 
```

[Sensitivity]와 [specificity]가 더 균형잡혀
있다. [Precision]\(positive라고 예측한 것 중 실제로 positive인 것의
비율)은 낮다.  이 결과를 test set에 적용하면 결과는 다음과 같다.

```R
Performance(logit.over, TestSet)
# $ConfusionMatrix
#       pred
# actual  no yes
#    no  123  24
#    yes   8  12
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.6000000   0.8367347   0.3333333   0.8083832 
```

Sensitivity는 0.6, specificity는 0.8367이다. [Precision]은 0.333밖에 되지 않는다.

[Youden Index]를 최대화시킬 수도 있지만 데이터셋이 이미
균형화되었으므로 필요는 없다.  그렇다고 해서 하면 절대 안 된다는 것도
아니니 그냥 시험삼아 해 보자.

```R
DF <- data.frame(truth = TrainSet$deny, pred = predict(logit.over, TrainSet, type = 'r'))
library(OptimalCutpoints)
cp <- optimal.cutpoints(pred~truth, data=DF, tag.healthy="no", method="Youden")
(cutoff <- cp$Youden$Global$optimal.cutoff$cutoff)
# [1] 0.496526
```

거의 0.5이므로(이것이 클래스 균형화의 효과이다) 별 차이 없을
것이다. 그래도 정 해 보고 싶으면 `Performance(logit.over, TestSet,
cutoff = cutoff)`라고 하면 된다.

## ROSE (Random Over-Sampling Examples)

[ROSE] package를 사용해서 큰 클래스로부터는 적당히 random하게
under-sampling하고 작은 클래스로부터는 적당히 over-sampling하여 대충
균형을 맞춰보자. 샘플링 방법에 대해 `ROSE` [도움말][ROSE-help]에
이렇게 설명되어 있다: &quot;Operationally, the new examples are drawn
from a conditional kernel density estimate of the two class, as
described in Menardi and Torelli (2013).&quot; 도움말의
&quot;Details&quot; 항목에 더 자세한 설명이 있다.

```R
library(ROSE)
Rose <- ROSE(deny~., data=TrainSet, seed=1)$data
table(Rose$deny)
#   no  yes 
# 1148 1065
```

`TrainSet` 데이터에 `no`가 1,948개, `yes`가 265개 있었는데 `ROSE`에
의해 생성된 데이터는 1,148개와 1,065개이다. Over-sampling과
under-sampling이 결합된 방법으로 보인다. 새로 생성된 데이터를 사용한
결과는 (클래스 균형화 이전의 원래) train set의 경우 다음과 같다.

```R
logit.rose <- glm(deny~., data=Rose, family=binomial)

Performance(logit.rose, TrainSet)
# $ConfusionMatrix
#       pred
# actual   no  yes
#    no  1594  354
#    yes   85  180
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.6792453   0.8182752   0.3370787   0.8016268 
```

대동소이하다. `TestSet`에 적용하면 결과는 다음과 같다.

```R
Performance(logit.rose, TestSet)
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

앞의 완전한 over-sampling에 비하면 `no` 행이 (122,25)에서 (124,23)으로
개선되었다. 그렇지만 그냥 그렇다는 것이지 이로부터 ‘실제 예측에
적용하면 `ROSE` 명령을 사용한 방법이 over-sampling한 방법보다
예측성과가 더 좋을 것’이라고 일반화시키지는 말기 바란다.

## <a name="#SMOTE">Synthetic minority oversampling technique (SMOTE)</a>

[Karthe (2016)][Karthe2016]에 [Chawla et al. (2002)의 SMOTE][SMOTE]가
설명되어 있다. 원래 `DMwR` 패키지에 `SMOTE` 명령이 있다고들 하는데
2022.2월 현재 `DMwR` 패키지는 CRAN으로부터 제거되어 있다. [이
링크][21]에 `DMwR` 패키지 설치 방법이 제시되어 있는데 소스 컴파일이
필요할 것이다. 그 대신 R의 [smotefamily] 패키지에 `SMOTE` 명령을
사용하겠다. 단, 이 `SMOTE` 명령은 사용하기 매우 번거롭다(`DMwR`
패키지의 `SMOTE` 명령도 그만큼 사용하기 번거로웠던 것 같다).

[21]: https://community.rstudio.com/t/downloading-a-package-that-has-been-removed-from-cran/107479

```R
fm <- deny~.
X <- model.matrix(fm, TrainSet)[,-1]
Y <- model.frame(fm, TrainSet)[,1]
set.seed(1)
Smote <- smotefamily::SMOTE(as.data.frame(X),Y)$data # install smotefamily
```

생성된 `Smote` 데이터에 목표변수명은 `class`이고(`deny`가 아님)
`character` 형이다. `factor` 형으로 바꾸자. `yes`와 `no`의 개수를 세어
보면 상당한 balancing이 이루어졌음을 알 수 있다.

```R
Smote$class <- as.factor(Smote$class)
table(Smote$class)
#   no  yes 
# 1948 1855 
```

이상에서 SMOTE를 사용하여 `yes` 클래스를 oversample한 데이터를
생성했다. 이제 이 `Smote` 데이터를 이용하여 학습을 해 보자.

```R
logit.smote <- glm(class~., data=Smote, family=binomial)
```

Train set (`TrainSet`)에서 예측한 결과를 보자.

```R
phat.smote <- predict(logit.smote, as.data.frame(X), type='r')
yhat.smote <- ifelse(phat.smote >= 0.5, "yes", "no")
SummPred(TrainSet$deny, yhat.smote)
# $ConfusionMatrix
#       pred
# actual   no  yes
#    no  1611  337
#    yes   80  185
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.6981132   0.8270021   0.3544061   0.8115680 
```

앞에 random oversampling (ROSE)한 결과는 다음과 같았다.

```R
# Random oversampling
# Sensitivity Specificity   Precision    Accuracy 
#   0.7018868   0.8172485   0.3431734   0.8034343 
```

SMOTE 표본을 사용할 때 [sensitivity]가 아주 약간 악화되고
[specificity]가 아주 약간 개선되었다. 대단한 일은 일어나지 않았다.

Test set에 적용한 결과는 다음과 같다.

```R
TestX <- model.matrix(fm, TestSet)[,-1]
phat <- predict(logit.smote, as.data.frame(TestX), type='r')
yhat <- ifelse(phat>0.5, "yes", "no")
SummPred(TestSet$deny, yhat)
# $ConfusionMatrix
#       pred
# actual  no yes
#    no  126  21
#    yes   8  12
# 
# $Summary
# Sensitivity Specificity   Precision    Accuracy 
#   0.6000000   0.8503401   0.3529412   0.8203593 
```

코딩이 복잡하다.

## ROC 곡선 비교

원래의 `TrainSet`, random oversampling 결과인 `Over`, SMOTE 표본인
`Smote`, ROSE 표본인 `Rose`를 사용할 때의 [ROC] 곡선을 비교하자.
`ROCit` 패키지의 `rocit` 함수를 사용하려면 확률예측치가 필요하므로
귀찮지만 다음과 같이 확률예측치를 구하자. 비교를 위해서 클래스 균형화
없이 그냥 `TrainSet`에 대하여 full logit한 결과와도 비교하자.  우선
train set에서 확률예측치를 구하자.

```R
logit.orig <- glm(deny~., data = TrainSet, family = binomial)

phat.orig <- predict(logit.orig, TrainSet, type = 'r')
phat.wlogit <- predict(wlogit, TrainSet, type = 'r')
phat.over <- predict(logit.over, TrainSet, type = 'r')
phat.rose <- predict(logit.rose, TrainSet, type = 'r')
## SMOTE - complicated
X <- model.matrix(fm, TrainSet)[,-1]
phat.smote <- predict(logit.smote, as.data.frame(X), type='r')
```

다음으로 ROC를 구하자.

```R
library(ROCit)
roc.orig <- rocit(phat.orig, TrainSet$deny)
roc.wlogit <- rocit(phat.wlogit, TrainSet$deny)
roc.over <- rocit(phat.over, TrainSet$deny)
roc.rose <- rocit(phat.rose, TrainSet$deny)
roc.smote <- rocit(phat.smote, TrainSet$deny)
plot(roc.orig, YIndex = FALSE)
with(roc.over, lines(FPR, TPR, col=3))
with(roc.rose, lines(FPR, TPR, col=4))
with(roc.smote, lines(FPR, TPR, col=5))
with(roc.wlogit, lines(FPR, TPR, col=2))
legend('right', c('Original train set', 'Weighted logit', 'Random oversampling', 'ROSE', 'SMOTE'), lty=1, col=1:5, bty='n', cex=.75)
```

![원래 train set, 가중 로짓, random하게 oversample된 데이터, ROSE 데이터를
이용한 분석에서 ROC 곡선](imgs/roc_comp4.svg)

거의 차이가 없으므로 cutoff 값만 잘 맞추면 성능은 모두 비슷할
것이다. [AUC][ROC] (ROC 곡선 아래 넓이, area under the curve)도 거의
차이 나지 않는다.

```R
c(roc.orig$AUC, roc.wlogit$AUC, roc.over$AUC, roc.rose$AUC, roc.smote$AUC)
# [1] 0.8310953 0.8317558 0.8306633 0.8260683 0.8307446
```

참고로, 이 데이터의 경우 클래스 균형을 맞추는 것은 (cutoff 값을
조정하는 것에 비하여) 별 도움이 되지 않는 것으로 나타났으나, 다른
데이터에서는 다를 수 있다고 한다([Karthe, 2016][Karthe2016]).

[ovun]: https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis
[ROSE]: https://cran.r-project.org/package=ROSE
[smotefamily]: https://CRAN.R-project.org/package=smotefamily
[sensitivity]: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
[specificity]: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
[precision]: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
[Youden Index]: https://en.wikipedia.org/wiki/Youden%27s_J_statistic
[ROSE-help]: https://www.rdocumentation.org/packages/ROSE/versions/0.0-4/topics/ROSE
[SMOTE]: https://www.jair.org/index.php/jair/article/view/10302
[sensitivity]: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
[ROC]: https://en.wikipedia.org/wiki/Receiver_operating_characteristic