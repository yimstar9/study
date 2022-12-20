##임성구

# 1. reshape2패키지와 iris데이터 셋을 사용하여 다음을 실행하시오.
# library(reshape2)
# data(iris)
# 1) 꽃의 종류(Species)를 기준으로 ‘넓은 형식’을 ‘긴 형식’으로 변경하기
# (힌트. melt()함수 이용)
# 2) 꽃의 종별로 나머지 4가지 변수의 합계 구하기
# (힌트. dcast()함수 이용)

library(reshape2)
data(iris)

#1)
melt <- melt(iris,id="Species", na.rm=TRUE)
head(melt)

#2)
names(melt)
dcast <- dcast(melt, Species ~ variable, sum)
dcast

# 2. dplyr패키지와 iris데이터 셋을 이용하여 다음을 실행하시오
# 1) iris의 꽃잎의 길이(Petal.Length)컬럼을 대상으로 1.5이상의 값만 필터링하시오
# (힌트. 파이프 연산자(%>%)와 filter()함수 이용)
# 2) 1)번의 결과에서 1, 3, 5번 컬럼을 선택하시오
# (힌트.  파이프 연산자(%>%)와 select()함수 이용)
# 3) 2)번의 결과에서 1-3번 컬럼의 값을 뺀 diff 파생변수를 만들고, 앞부분 6개만 출력하
# 시오.
# (힌트. diff=1번 컬럼 – 3번 컬럼. 파이프 연산자(%>%)와 mutate()함수 이용)
# 4) 3)번의 결과에서 꽃의 종(Species)별로 그룹화하여 Sepal.Length와 Petal.Length변수의
# 평균을 계산하시오.
# (힌트. 파이프 연산자(%>%)와 group_by()와 summarise() 함수 이용)

# 1) iris의 꽃잎의 길이(Petal.Length)컬럼을 대상으로 1.5이상의 값만 필터링하시오
# (힌트. 파이프 연산자(%>%)와 filter()함수 이용)
library(dplyr)
data(iris)
a <- iris %>% filter(Petal.Length>=1.5);a

# 2) 1)번의 결과에서 1, 3, 5번 컬럼을 선택하시오
# (힌트. 파이프 연산자(%>%)와 select()함수 이용)
names(a)
b <- a%>% filter(Petal.Length >= 1.5) %>% select(Sepal.Length, Petal.Length, Species);b

#3) 2)번의 결과에서 1-3번 컬럼의 값을 뺀 diff 파생변수를 만들고, 앞부분 6개만 출력하
# 시오.
diff <- b%>%mutate(diff = Sepal.Length - Petal.Length) ;head(diff)

# 4) 3)번의 결과에서 꽃의 종(Species)별로 그룹화하여 Sepal.Length와 Petal.Length변수의
# 평균을 계산하시오.
# (힌트. 파이프 연산자(%>%)와 group_by()와 summarise() 함수 이용)
diff%>%group_by(Species)%>%summarise(Sepal_mean=mean(Sepal.Length), Petal_mean=mean(Petal.Length))

