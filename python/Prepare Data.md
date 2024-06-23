# 준비

## 데이터

데이터를 CSV 형태로 준비하려면 우선 R에서 다음을 실행하여 필요한 패키지를 설치한다.

```r
## R에서 실행
pkgs <- c("loedata", "gamair", "Ecdat", "AER", "wooldridge", "sampleSelection")
for (pkg in setdiff(pkgs, installed.packages())) install.packages(pkg)
```

다음으로 R에서 데이터셋들을 CSV 형태로 변환하여 저장할 것이다. 다음 R
코드를 클립보드에 복사한 후 R 콘솔(R, 터미널, Rstudio 등)에 붙여 넣어
실행한다. 그러면 R 데이터가 CSV 형식으로 변환되어 현 디렉토리(R 코드가
실행되고 있는 디렉토리로서 getwd()로써 알 수 있음)의 csv 하위 폴더에
저장된다. 이렇게 csv 폴더가 만들어지면 이 csv 폴더를 작업하고자 하는
폴더에 복사하면 된다. 필자의 경우 홈 디렉토리 내 doc/thebook/python
폴더 내에 csv 폴더가 있고, doc/thebook/python 폴더에서 파이썬 코딩을
하고 있다.

```r
## R에서 실행
What <- list(
    loedata = c('Death', 'Klips', 'Pubserv', 'Ekc', 'Galtonpar', 'Hcons',
        'Klosa', 'Ksalary', 'Regko', 'Fastfood', 'Hies', 'Ivdata', 'Hmda'),
    gamair = c('hubble'),
    Ecdat = c('Cigar', 'Consumption', 'Crime', 'Housing', 'Wages1', 'Wages',
        'Doctor', 'Schooling', 'Tobacco'),
    AER = c('CigarettesB'),
    wooldridge = c('twoyear', 'smoke', 'wage2'),
    sampleSelection = c('Mroz87'))

topdir = 'csv'
if (!dir.exists(topdir)) dir.create(topdir)

for (pkg in names(What)) {
  cat(pkg, ':', sep='')
  outdir <- file.path(topdir, pkg)
  if (!dir.exists(outdir)) dir.create(outdir)
  for (dta in What[[pkg]]) {
    cat(' ', dta, sep='')
    outfile <- file.path(outdir, paste0(dta, '.csv'))
    eval(parse(text = sprintf('data(%s, package = "%s")', dta, pkg)))
    z <- get(dta)
    if (dta == 'CigarettesB' && pkg == 'AER') {
      z$state <- rownames(z)
      z <- z[c('state', setdiff(names(z), 'state'))]
    }
    write.csv(z, file = outfile, row.names = FALSE)
  }
  cat("\n")
}
```

위 코드의 실행이 끝나면 "csv" 폴더 아래에 패키지명에 해당하는 폴더들이
생기고 이들 각 폴더 안에 데이터 CSV 파일이 생성된다. "csv" 폴더를 열어
보면 무슨 말인지 알 수 있을 것이다. 마지막으로, "csv" 폴더를 적절한
곳으로 이동하면 된다.

## Python 패키지 준비

다음 python 패키지를 설치한다.

```sh
pip install statsmodels numpy pandas matplotlib scipy
```

`statsmodels`는 회귀분석, `numpy`는 log나 exp 등, `pandas`는 CSV 파일
등 읽기, `matplotlib`은 그림, `scipy`는 t분포 등의 CDF와 임계값 등에
사용된다.
