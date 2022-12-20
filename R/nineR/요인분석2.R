# 요인분석2
# 다음의 [예제 1]은 R 패키지 {psych}를 이용하여 FA를 수행한다.
# [예제 1] 분석에 사용될 자료는 300명의 대학생에 대해 6개 항목(과목에 대해 좋아하는 정
# 도)에 대한 설문을 실시한 결과(가상의 자료)이다. 각 항목은 1(아주 싫어함)부터 5(아주 좋
# 아함)의 값을 가진다. 6개의 항목은 서로 다른 영역의 과목에 대한 선호도를 학생들에게 묻
# 는 것으로 구성되었다. 6개 과목은 biology(BIO), geology(GEO), chemistry(CHEM),
# algebra(ALG), calculus(CALC), statistics(STAT)이다. 이 자료는 아래의 싸이트로부터
# .csv 파일로 다운로드 받을 수 있다(c:/subjects.csv로 저장).
subjects <- read.csv("subjects.csv", head=T)
head(subjects, 3)
tail(subjects)
#install.packages("psych")
library(psych)
options(digits=3)
(corMat <- cor(subjects))

#install.packages("GPArotation")
library("GPArotation") # 사교회전(“oblimin”옵션)의 수행에 필요함
EFA <- fa(r = corMat, nfactors = 2, rotate="oblimin", fm = "pa")
EFA
ls(EFA)
EFA$loadings
load<-EFA$loadings
plot(load, type="n")
text(load, labels=names(subjects), cex=.7)
install.packages("nFactors")
library(nFactors)
ev <- eigen(cor(subjects)) # get eigenvalues

ap <- parallel(subject=nrow(subjects),var=ncol(subjects), rep=100,cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS)
install.packages("c:\\emmeans_1.7.0.tar.gz",repos=NULL,type="source")
install.packages("FactoMineR")
install.packages("mvtnorm")
library(estimability)
library(emmeans)
library(devtools)
install_github("husson/FactoMineR")
library(FactoMineR)
install.packages("sem")
library(sem)
result <- PCA(subjects)
names(subjects) <-c("X1", "X2", "X3", "X4", "X5", "X6")
names(subjects)
mydata.cov <- cov(subjects)
model.mydata <- specifyModel()
F1 -> X1, lam1, NA
F1 -> X2, lam2, NA
F1 -> X3, lam3, NA
F2 -> X4, lam4, NA
F2 -> X5, lam5, NA
F2 -> X6, lam6, NA
X1 <-> X1, e1, NA
X2 <-> X2, e2, NA
X3 <-> X3, e3, NA
X4 <-> X4, e4, NA
X5 <-> X5, e5, NA
X6 <-> X6, e6, NA
F1 <-> F1, NA, 1
F2 <-> F2, NA, 1
F1 <-> F2, F1F2, NA

mydata.sem <- sem(model.mydata, mydata.cov, nrow(subjects))
#print(results) #(fit indices, paramters, hypothesis tests)
summary(mydata.sem)  
stdCoef(mydata.sem)
