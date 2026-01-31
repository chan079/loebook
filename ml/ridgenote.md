## glmnet에 의한 ridge 회귀의 복원

[Ridge 계산에 관한 주석](13-ridge-lasso.md#fnref1)에 따라
`glmnet`에 의한 ridge 회귀 결과를 직접 계산하여 확인해 본다. 데이터는
[여기](10-data.md) 참고.

주석 내용을 반복하면 다음과 같다.

<blockquote>
`glmnet`에 `standardize = TRUE`
옵션(디폴트)을 주고 계산하는 ridge 추정값은 다음과 같이 구할 수 있다.
① $\tilde{x}_{ij}^* = (x_{ij}-\bar{x}_j)/\hat\sigma_j$로 표준화한다.
여기서 $\bar{x}_j$는 $x_{ij}$의 표본평균, $\hat\sigma_j$는 $x_{ij}$의
표본평균편차(표본분산 계산 시 $n$으로 나눔). 표본평균을 빼는 것은
절편을 처리하기 위함이며, $\hat\sigma_j$로 나누는 것은 변수 표준화. ②
$\hat\beta^* = (n^{-1} \tilde{X}^{*\prime} \tilde{X}^* +
\lambda^*I)^{-1} n^{-1} \tilde{X}^{*\prime}y$를 구한다($\tilde{X}^*$는
$\tilde{x}_{ij}^*$의 행렬, $\lambda^* = \lambda/\hat\sigma_y$). 이것은
‘베타계수’ $\beta_j \hat\sigma_j$들의 추정값 벡터이다. ③ 표준화를
되돌리기 위해 $\hat\beta_j = \hat\beta_j^* / \hat\sigma_j$를 계산하면
이것이 최종적인 기울기 계수의 ridge 추정값. ④ 절편 추정값은 앞과
동일하게 $\hat\alpha = \bar{y} - \hat\beta_1 \bar{x}_1 - \cdots -
\hat\beta_p \bar{x}_p$로 계산.
</blockquote>

## `glmnet`에 의한 ridge 회귀

다음 `glmnet`에 의한 결과를 보라.

```R
ridge <- glmnet(X, Y, alpha = 0)
(lamb <- ridge$lambda[50])  # an arbitrary choice
# [1] 3485.563
coef(ridge, s = lamb)
# 20 x 1 sparse Matrix of class "dgCMatrix"
#                         1
# (Intercept) 1349.96474879
# grdp          -1.46733791
# regpop       -70.45896296
# popgrowth     -2.60572536
# eq5d          -2.24328091
# deaths        -0.01593032
# drink         -2.97176050
# hdrink         1.03296288
# smoke          1.90460780
# aged           2.90703405
# divorce       -8.98595269
# medrate        0.43610941
# gcomp         -0.90484486
# vehipc        97.75098402
# accpv         -1.13114311
# dumppc        -4.58445915
# stratio       -3.16084850
# deathrate      0.07065207
# pctmale       -3.94300918
# accpc          1.88459406
```

이 주어진 $\lambda$ (`lamb`)을 이용한 ridge 추정값을 행렬연산에 의하여
똑같이 구해 보는 것이 목적이다.

## 행렬 연산에 의한 ridge 구현

먼저, `glmnet`에서 말하는 “표준편차”들은 $n-1$이 아닌 $n$으로 나누어
구한다. 이를 위해 `sd0`라는 함수를 만든다.

```R
sd0 <- function(x) sqrt(mean((x-mean(x))^2))

## X, Y defined in data preparation page
sig.Y <- sd0(Y)
sig.X <- apply(X, 2, sd0)
```

#### 준비: $\lambda$의 변환

```R
(lamb.star <- lamb/sig.Y)
# [1] 10.35528
```

`glmnet`에서 사용한 $\lambda$ 값 `3485.563`은 행렬 연산을 통한 수동
연산에서 `10.35528`에 해당한다.

#### ① X변수들의 표준화

```R
Xs <- apply(X, 2, function(x) (x-mean(x))/sd0(x))
```

#### ② $\hat\beta^*$의 계산

```R
n <- nrow(Xs)
bhat.star <- solve(1/n*crossprod(Xs) + lamb.star*diag(ncol(Xs)), 1/n*crossprod(Xs, Y))
```

#### ③ $\hat\beta$의 복원

```R
(bhat <- bhat.star / sig.X)
#                   [,1]
# grdp       -1.46731962
# regpop    -70.45811386
# popgrowth  -2.60571609
# eq5d       -2.24326913
# deaths     -0.01593022
# drink      -2.97175109
# hdrink      1.03295937
# smoke       1.90460738
# aged        2.90703561
# divorce    -8.98595524
# medrate     0.43611098
# gcomp      -0.90484667
# vehipc     97.75111247
# accpv      -1.13114176
# dumppc     -4.58445394
# stratio    -3.16085391
# deathrate   0.07065214
# pctmale    -3.94300686
# accpc       1.88459496
```

이 결과는 앞에서 `glmnet`으로 구한 결과와 동일함을 확인할 수 있다.

#### ④ 절편 추정값 확인

```R
mean(Y) - sum(colMeans(X)*bhat)
# [1] 1349.963
```

#### 대안적인 계산 방법

주석에 다음과 같이 쓰여 있기도 하다. 

<blockquote>
참고로, 위 ①∼③의 $\hat\beta$은 $D =
diag(\hat\sigma_1^2, \ldots, \hat\sigma_p^2)$라 할 때
$(n^{-1}\tilde{X}'\tilde{X} + \lambda^* D)^{-1} n^{-1}
\tilde{X}'y$으로 계산한 것과 동일하다.
</blockquote>

이를 확인해 보자.

```R
Xd <- apply(X, 2, function(x) x - mean(x))
solve(1/n*crossprod(Xd) + lamb.star*diag(sig.X^2), 1/n*crossprod(Xd,Y))
#                   [,1]
# grdp       -1.46731962
# regpop    -70.45811386
# popgrowth  -2.60571609
# eq5d       -2.24326913
# deaths     -0.01593022
# drink      -2.97175109
# hdrink      1.03295937
# smoke       1.90460738
# aged        2.90703561
# divorce    -8.98595524
# medrate     0.43611098
# gcomp      -0.90484667
# vehipc     97.75111247
# accpv      -1.13114176
# dumppc     -4.58445394
# stratio    -3.16085391
# deathrate   0.07065214
# pctmale    -3.94300686
# accpc       1.88459496
```

결과는 `glmnet`과 동일하다.

[[Ridge 계산에 관한 주석으로 돌아가기](13-ridge-lasso.md#fnref1)]
