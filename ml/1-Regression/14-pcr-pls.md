데이터는 <a href="10-data.md">여기</a>를 참조하거나 다음 명령 실행.

```R
rm(list=ls(all=TRUE))
load(url("https://github.com/chan079/loebook/raw/main/ml/1-Regression/data.RData"))
```

# PCR과 PLS

## Principal Components Regression (PCR)

10-fold [CV]로 [PCR]에서 최적 주성분 개수를 결정해 보자.

```R
library(pls)
set.seed(1)
pcreg <- pcr(ynext~., data=z14, scale = TRUE, validation = "CV") # segments=10 (default)
RMSEP(pcreg)
#        (Intercept)  1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
# CV           338.1    142.9    124.7    125.2    123.2    118.6    117.3
# adjCV        338.1    142.1    124.3    124.8    122.8    117.9    116.7
#        7 comps  8 comps  9 comps  10 comps  11 comps  12 comps  13 comps
# CV       116.6    102.9    101.8     101.1    100.44     95.82     90.63
# adjCV    116.6    102.1    101.1     100.5     99.87     95.31     90.20
#        14 comps  15 comps  16 comps  17 comps  18 comps  19 comps
# CV        92.11     53.85     53.96     54.53     51.13     50.97
# adjCV     91.83     53.60     53.72     54.35     50.89     50.73
validationplot(pcreg, type='o')
```

![PCR의 성분 개수별 CV 예측오차](imgs/pcr_cv.svg)

PCR에서 예측변수들의 주성분을 구하는 과정은 비지도학습([unsupervised
learning])이므로 피예측변수에 대한 설명력이 신속히 개선되지는 않을
수도 있음에 유의하라. CV 에러가 가장 작은 것은 19개 성분을 모두
사용하는 경우(`19 comps`)로서, 이는 [전체 변수를 사용한
OLS](11-subset-selection.md)와 동일하다.

그런데 CV 기준을 보면 `ncomp=15`에서 예측오차 정도가 크게 감소한 후
미미하게만 감소한다. `ncomp=15`를 선택하면 test set에서 예측 성과는
다음과 같다(아래 코드에서 `X15`는 [데이터 준비](10-data.md)에서
생성되었으며 2015년 test set 우변변수들의 행렬임).

```R
RMSE(z15$ynext, predict(pcreg, X15, ncomp=15))
# 47.95138
```

## Partial Least Squares (PLS)

[PLS]는 특성변수들의 선형결합 시 목표변수에 대한 설명력을 고려하므로
CV 오차가 PCR의 경우보다 훨씬 빨리 하락한다.

```R
set.seed(1)
plsreg <- plsr(ynext~., data=z14, scale = TRUE, validation = 'CV')
RMSEP(plsreg)
#        (Intercept)  1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
# CV           338.1    116.9    93.88    73.64    61.03    56.00    54.53
# adjCV        338.1    116.4    93.72    73.15    60.48    55.28    54.20
#        7 comps  8 comps  9 comps  10 comps  11 comps  12 comps  13 comps
# CV       53.53    53.12    52.45     51.80     51.49     51.14     51.12
# adjCV    53.29    52.88    52.22     51.52     51.20     50.88     50.87
#        14 comps  15 comps  16 comps  17 comps  18 comps  19 comps
# CV        51.10     51.05     51.08     51.07     50.96     50.97
# adjCV     50.86     50.81     50.85     50.76     50.71     50.73
validationplot(plsreg, type='o')
```

![PLS의 성분 개수별 CV 예측오차](imgs/pls_cv.svg)

그림을 보면 PCR에 비하여 PLS의 CV 오차가 확실히 빨리 감소하는 것을
확인할 수 있다. 이는 PCR에서 주성분들을 구하는 방법이 목표변수와 아무
상관 없는 ‘비지도학습’인 반면 PLS는 목표변수를 고려하는 ‘지도학습’이기
때문이다.

위 PLS에서 CV error 기준으로 가장 작은 예측오차를 보이는 것은
`ncomp=18`이지만, 그림을 볼 때 5 이상이면 충분히 최적화된 것으로
보이고, 또 15보다 16이 더 오차가 커지므로, test set에서 18, 15, 5를
비교하면 다음과 같다.

```R
RMSE(z15$ynext, predict(plsreg, X15, ncomp=18))
# [1] 51.50172
RMSE(z15$ynext, predict(plsreg, X15, ncomp=15))
# [1] 51.20954
RMSE(z15$ynext, predict(plsreg, X15, ncomp=5))
# [1] 51.44507
```

어느 경우에나 예측성능은 거의 동일하다. 이상에서 test set에서의
예측성능을 확인해 보았는데, 이는 test set에서 목표변수 실제값을 알고
있기 때문에 확인할 수 있는 것이다. (테스트해 보는 것이 아니라) 실제
문제에 적용할 때에는 예측 성과가 어떠할지 미리 알 수 없다. 학습용
데이터만을 사용해서 CV를 한 결과로는 `ncomp=18`이 가장 좋은 것이라는
판단이 구해졌다. 단, 이 결과는 난수의 시드를 바꾸거나 CV를 위한 관측치
분류 방법을 바꾸면 바뀔 수 있다.

[CV]: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
[PCR]: https://en.wikipedia.org/wiki/Principal_component_regression
[PLS]: https://en.wikipedia.org/wiki/Partial_least_squares_regression