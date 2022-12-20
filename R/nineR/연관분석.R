# .libPaths()
# install.packages("Matrix")
# library(Matrix)
 

#install.packages("hms")
#install.packages("gganimate")


#install.packages(c("igraph",'tidygraph','graphlayouts','gganimate'),dependencies=T)
# update.packages("arules")
# update.packages("arulesViz")
# install.packages("igraph", type="binary")
# install.packages("tidygraph", type="binary")
# install.packages("graphlayouts", type="binary")
# install.packages("arules")
# install.packages("arules",type="binary")
# install.packages("arulesViz",type="binary")
# install.packages("ggraph", type="binary")
library(arules)
library(arulesViz)
library(ggraph)

#library(arulesViz)
# arulesViz패키지

# 실습 (트랜잭션 객체를 대상으로 연관규칙 생성)
# 1단계: 연관분석을 위한 패키지 설치
# install.packages("arules")
# library(arules)
# arules 패키지
# 2단계: 트랜잭션(transaction) 객체 생성


tran <- read.transactions("dataset4/tran.txt", format = "basket", sep = ",")
tran
# read.transaction()함수: transaction 객체 생성
# https://www.rdocumentation.org/packages/arules/versions/1.6-7/topics/read.transactions
# 3단계: 트랜잭션 데이터 보기
inspect(tran)
# inspect()함수: transaction 객체 확인
# https://www.rdocumentation.org/packages/arules/versions/1.6-6/topics/inspect
# 4단계: 규칙(rule) 발견1
rule <- apriori(tran, parameter = list(supp = 0.3, conf = 0.1))
inspect(rule)
# arules패키지에서 제공하는 apriori()함수를 이용하여 트랜잭션 객체를 대상으로 규칙
# 발견
# 형식: apriori(트랜잭션 data, parameter=list(supp, conf))
# Where 
# Default: support: 0.1, confidence: 0.8
# https://www.rdocumentation.org/packages/arules/versions/1.6-7/topics/apriori
# 5단계: 규칙(rule) 발견2
rule <- apriori(tran, parameter = list(supp = 0.1, conf = 0.1))
inspect(rule)
# 
# 2.2 트랜잭션 객체 생성
# 거래 데이터를 대상으로 트랜잭션 객체를 생성하기 위해서 arules패키지에서 제공되는
# read.transaction()함수 사용
# read.transactions(file, format=c("basket", "single"), sep=NULL, cols=NULL,rm.duplicate=FALSE, encoding="unknown")
# where
# file: 트랜잭션 객체를 생성할 대상의 데이터 파일명
# format: 트랜잭션 데이터 셋의 형식 지정(basket 또는 single)
# - basket: 여러 개의 상품(item)으로 구성된 경우 (transaction ID없이 상품으로만 구성된
#                                    경우)
# - single: 트랜잭션 구분자(Transaction ID)에 의해서 상품(item)이 대응된 경우
# sep: 각 상품(item)을 구분하는 구분자 지정
# cols: single인 경우 읽을 컬럼 수 지정(basket은 생략)
# rm.duplicates: 중복 트랜잭션 상품(item) 제거
# encoding: 데이터 셋의 인코딩 방식 지정
# 실습 (single 트랜잭션 객체 생성)

stran <- read.transactions("dataset4/demo_single", format = "single", cols = c(1, 2))
inspect(stran)
# format=”single” 속성: 한 개의 트랜잭션 구분자에 의해서 상품(item)이 연결된 경우
# step 속성 제외: item은 공백으로 구분
# cols 속성: 처리할 컬럼 지정
stran2 <- read.transactions("dataset4/single_format.csv", format = "single",
                            sep = ",", cols = c(1, 2), rm.duplicates = T)
# 중복된 트랜잭션이 존재하는 경우 해당 트랜잭션을 제거하기 위해 rm.duplicates=T 속성
# 지정
# 2단계: 트랜잭션과 상품수 확인
stran2
# 3단계: 요약통계량 제공
summary(stran2)
# 실습 (규칙 발견(생성))
# 1단계: 규칙 생성하기
astran2 <- apriori(stran2)
# arules패키지에서 제공되는 apriori()함수는 연관규칙의 평가척도를 이용하여 규칙을 생성
# 2단계: 발견된 규칙 보기
inspect(astran2)
# 3단계: 상위 5개의 향상도를 내림차순으로 정렬하여 출력
inspect(head(sort(astran2, by = "lift")))

data(Adult)
Adult
data("AdultUCI")
str(AdultUCI)
# 실습 (Adult 데이터 셋의 요약통계량 보기)
# 1단계: data.frame형식으로 보기
adult <- as(Adult, "data.frame")
str(adult)
head(adult)
# 2단계: 요약통계량
summary(Adult)

ar <- apriori(Adult, parameter = list(supp = 0.1, conf = 0.8))
# 실습 (다양한 신뢰도와 지지도를 적용한 예)
# 1단계: 지지도를 20%로 높인 경우 1,306개 규칙 발견
ar1 <- apriori(Adult, parameter = list(supp = 0.2))
# 2단계: 지지도를 20%, 신뢰도 95%로 높인 경우 348개 규칙 발견
ar2 <- apriori(Adult, parameter = list(supp = 0.2, conf = 0.95))
# 3단계: 지지도를 30%, 신뢰도 95%로 높인 경우 124개 규칙 발견
ar3 <- apriori(Adult, parameter = list(supp = 0.3, conf = 0.95))
# 4단계: 지지도를 35%, 신뢰도 95%로 높인 경우 67개 규칙 발견
ar4 <- apriori(Adult, parameter = list(supp = 0.35, conf = 0.95))
# 5단계: 지지도를 40%, 신뢰도 95%로 높인 경우 36개 규칙 발견
ar5 <- apriori(Adult, parameter = list(supp = 0.4, conf = 0.95))

# 2단계: 연관규칙 시각화
library(arulesViz)
plot(ar3, method = "graph", control = list(type = "items"))

# 지지도, 신뢰도 조정 필요

# 실습 (Groceries 데이터 셋으로 연관분석)
# arules패키지에서 제공되는 Groceries 데이터 셋 사용
# 1단계: Groceries 데이터 셋 가져오기
data("Groceries")
str(Groceries)
Groceries
# 더 알아보기 (Groceries 데이터 셋)
# https://www.rdocumentation.org/packages/arules/versions/1.6-8/topics/Groceries
# 2단계: 데이터프레임으로 형 변환
Groceries.df <- as(Groceries, "data.frame")
head(Groceries.df)
# 3단계: 지지도 0.001, 신뢰도 0.8 적용 규칙 발견
rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.8))
# 4단계: 규칙을 구성하는 왼쪽(LHS)  오른쪽(RHS)의 item 빈도수 보기
# 규칙의 표현 A(LHS)  B(RHS)
plot(rules, method = "grouped")

