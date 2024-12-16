# 계량경제학강의 제5판 오타 수정

## 407면 하단~408면 상단

실습결과 3행, 5행, 10행, 12행을 수정하여야 하며, 올바른 코드와 결과는 다음과 같습니다.

```r
stage1 <- lm(x2~x1+z2a+z2b,data=Ivdata)
Ivdata$x2hat <- stage1$fitted
tsls <- ivreg(y~x1+x2|x1+z2a+z2b,data=Ivdata)
aux1 <- lm(z2b~x1+x2hat,data=Ivdata)
Ivdata$w <- aux1$resid*tsls$residuals
Ivdata$one <- 1
aux2 <- lm(one~w-1,data=Ivdata)
stat <- nrow(Ivdata)*summary(aux2)$r.sq
stat
# [1] 0.3571221
1-pchisq(stat,1)
# [1] 0.5501089
```
