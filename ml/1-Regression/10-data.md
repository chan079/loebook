# 데이터 준비

`loedata` 패키지(『계량경제학 강의』 데이터 패키지)를 아직 설치하지 않았으면 설치한다.

```R
install.packages("loedata")
```

패키지 설치 후 아래와 같이 데이터를 정리한다. 이하 전체 코드는 <a href="#allcodes">여기</a>로 이동.

```R
data(RegkoPanel, package='loedata')
z <- RegkoPanel
z$lngrdppc <- log(z$grdp/z$regpop)
z$lnpop <- log(z$regpop)
for (v in c("grdp", "regpop", "nbirth", "deaths", "gcomp")) z[[v]] <- NULL
z$eq5d <- z$eq5d*100

z14 <- subset(z, year==2014)
z15 <- subset(z, year==2015)
z16 <- subset(z, year==2016)

z14$ynext <- z15[match(z14$id, z15$id), 'deathrate']
z15$ynext <- z16[match(z15$id, z16$id), 'deathrate']

z14 <- na.omit(z14[,-(1:5)])
z15 <- na.omit(z15[,-(1:5)])
z.test <- z15

dim(z14)
# [1] 223  20
names(z14)
#  [1] "popgrowth" "eq5d"      "drink"     "hdrink"    "smoke"     "aged"
#  [7] "divorce"   "medrate"   "gcomp"     "vehipc"    "accpv"     "dumppc"
# [13] "stratio"   "deathrate" "cbrate"    "tfrate"    "pctmale"   "accpc"
# [19] "lngrdppc"  "lnpop"     "ynext"
```

`z14` 데이터에서 `ynext` 변수는 2015년 `deathrate`이고, z15 데이터에서 `ynext` 변수는 2016년 `deathrate`이다. `z14` 데이터를 이용하여 머신러닝을 실습해 본다(train set). 목표변수는 이듬해 사망률(`ynext`), 예측변수들은 당해 연도의 모든 변수들이다. `z15` (`z.test`) 데이터는 시험용으로 사용한다(test set).

경우에 따라 data frame이 아니라 특성변수 행렬(matrix)과 목표변수 벡터가 필요하다. 이때 사용할 목적으로 `X`, `Y`, `X15`를 만든다.

```R
fm <- ynext~.
Y <- model.frame(fm, data=z14)[,1]
X <- model.matrix(fm, data=z14)[,-1]
X15 <- model.matrix(fm, data=z.test)[,-1]
```

`RMSE(x,y)`라는 함수를 만들어 사용하자. 그리고 `rmspe.rw`는 단순 임의보행(random walk)을 가정하여 예측값을 구하는 경우(직전 연도와 동일한 값으로 예측)의 RMSE이다.

```R
RMSE <- function(x,y) sqrt(mean((x-y)^2))
rmspe.rw <- RMSE(z15$ynext, z15$deathrate)
rmspe.rw # random walk, defined in index11.php
# [1] 53.24273
```

### <a name="allcodes">전체 코드</a>

```R
rm(list=ls(all=TRUE))
data(RegkoPanel, package='loedata')
z <- RegkoPanel
z$lngrdppc <- log(z$grdp/z$regpop)
z$lnpop <- log(z$regpop)
for (v in c("grdp", "regpop", "nbirth", "deaths", "gcomp")) z[[v]] <- NULL
z$eq5d <- z$eq5d*100
z14 <- subset(z, year==2014)
z15 <- subset(z, year==2015)
z16 <- subset(z, year==2016)
z14$ynext <- z15[match(z14$id, z15$id), 'deathrate']
z15$ynext <- z16[match(z15$id, z16$id), 'deathrate']
z14 <- na.omit(z14[,-(1:5)])
z15 <- na.omit(z15[,-(1:5)])
z.test <- z15
fm <- ynext~.
Y <- model.frame(fm, data=z14)[,1]
X <- model.matrix(fm, data=z14)[,-1]
X15 <- model.matrix(fm, data=z.test)[,-1]
RMSE <- function(x,y) sqrt(mean((x-y)^2))
rmspe.rw <- RMSE(z15$ynext, z15$deathrate)

# save(z14, z15, z.test, X, Y, X15, RMSE, rmspe.rw, file = 'data.RData')
```
