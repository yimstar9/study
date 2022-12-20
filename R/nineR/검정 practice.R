N = 10000
X = 165.1
S = 2
low <- X -1.96 * S / sqrt(N)
high <- X + 1.96 * S / sqrt(N)
low; high
high - X
(low - X) * 100
(high - X) * 100


setwd("C:/nineR ")
data <- read.csv("Part3/one_sample.csv", header = TRUE)
head(data)
x <- data$survey
summary(x)
length(x)
table(x)

install.packages("prettyR")
library(prettyR)
freq(x)
##binom.test(x, n, p, alternative=c(“two.sided”, “less”, “greater”), conf.level = 0.95)

binom.test(14, 150, p = 0.2)
binom.test(14, 150, p = 0.2, alternative = "two.sided", conf.level = 0.95)
binom.test(c(14, 150), p = 0.2, 
           alternative = "greater", conf.level = 0.95)

binom.test(c(14, 150), p = 0.2, 
           alternative = "less", conf.level = 0.95)


mean(x, na.rm = T)
x1 <- na.omit(x)
mean(x1)
shapiro.test(x1)

par(mfrow = c(1, 2))
hist(x1)

t.test(x1, mu = 5.2)
qqnorm(x1)
qqline(x1, lty = 1, col = "blue")
t.test(x1, mu = 5.2, alter = "two.side", conf.level = 0.95)
t.test(x1, mu = 5.2, alter= "greater", conf.level = 0.95)
qt(0.05, 108, lower.tail=F)



data <- read.csv("Part3/two_sample.csv", header = TRUE)
head(data)
summary(data)
result <- subset(data, !is.na(score), c(method, score))
a <- subset(result, method == 1)
b <- subset(result, method == 2)
a1 <- a$score
b1 <- b$score
length(a1)
length(b1)
mean(a1)
mean(b1)
var.test(a1, b1)

t.test(a1, b1, altr = "two.sided", 
       conf.int = TRUE, conf.level = 0.95)
t.test(a1, b1, alter = "greater", 
       conf.int = TRUE, conf.level = 0.95)
t.test(a1, b1, alter = "less", 
       conf.int = TRUE, conf.level = 0.95)

data <- read.csv("Part3/paired_sample.csv", header = TRUE)

result <- subset(data, !is.na(after), c(before, after))
x <- result$before
y <- result$after
x; y

mx <- matrix(c(1,2,3,4,5,6,7,8,9),nrow=3,byrow=T)
mｘ
as.vector(mx)
a <- as.Date("08/13/2013","%M/%d/%Y")
a
mean(a)

# 두 집단 평균 검정 (독립 표본 T-검정)
data <- read.csv("part3/two_sample.csv", header = TRUE)
head(data)
summary(data)
result <- subset(data, !is.na(score), c(method, score))
a <- subset(result, method == 1);a
b <- subset(result, method == 2)
a1 <- a$score;a1
b1 <- b$score
var.test(a1, b1)

#카이제곱
# install.packages("gmodels")
library(gmodels)
# install.packages("ggplot2")
library(ggplot2)
data <- read.csv("part3/cleanDescriptive.csv", header = TRUE)
head(data)
x <- data$level2
y <- data$pass2
result <- data.frame(Level = x, Pass = y)
dim(result)
table(result)
CrossTable(x = diamonds$color, y = diamonds$cut)
x <- data$level2;x
y <- data$pass2;y
CrossTable(x, y)


chisq.test(c(4, 6, 17, 16, 8, 9))


data <- textConnection(
  "스포츠음료종류 관측도수
 1 41
 2 30
 3 51
 4 71
 5 61
 ")
x <- read.table(data, header = T)
x
chisq.test(x$관측도수)


data <- read.csv("part3/cleanDescriptive.csv", header = TRUE)
x <- data$level2
y <- data$pass2
CrossTable(x, y, chisq = TRUE)

data <- read.csv("part3/homogenity.csv")
head(data)
data <- subset(data, !is.na(survey), c(method, survey))

data$method2[data$method == 1] <- "방법1"
data$method2[data$method == 2] <- "방법2"
data$method2[data$method == 3] <- "방법3"
data$survey2[data$survey == 1] <- "1.매우만족"
data$survey2[data$survey == 2] <- "2.만족"
data$survey2[data$survey == 3] <- "3.보통"
data$survey2[data$survey == 4] <- "4.불만족"
data$survey2[data$survey == 5] <- "5.매우불만족"

table(data$method2, data$survey2)
# *교차분할표 작성 시 각 집단의 길이가 같아야 함
# 4단계: 동질성 검정 – 모든 특성치에 대한 추론 검정
chisq.test(data$method2, data$survey2)

letters<-textConnection(
"a b c d e f g h i j k l m n
o p q r s
 t u v w x y z");

letters
> readLines(x, 2)
[1] "a" "b"

> close(x)
