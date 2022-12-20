#임성구
#R ch16 연습문제 연관분석
# 1. tranExam.csv 파일을 대상으로 중복된 트랜잭션 없이 1-2컬럼만 single형식으로 트랜잭
# 션 객체를 생성하시오.
# 1단계: 트랜잭션 객체 생성 및 확인
# 2단계: 각 items별로 빈도수 확인
# 3단계: 파라미터(supp = 0.3, conf = 0.1)를 이용하여 규칙(rule)생성
# 4단계: 연관규칙 결과 보기

#1단계
library(arules)
stran <- read.transactions("Part4/tranExam.csv", format = "single", sep = ",", cols = c(1, 2), rm.duplicates=T)
inspect(stran)

#2단계
summary(stran)

#3단계
ruleExam <- apriori(stran)
ruleExam <- apriori(stran,parameter=list(supp=0.3, conf=0.1))

#4단계
ruleExam
inspect(ruleExam)




#################################################################################
# 2. Adult데이터 셋을 대상으로 다음 조건에 맞게 연관분석을 수행하시오.
# 1) 최소 support = 0.5, 최소 confidence = 0.9를 지정하여 연관규칙을 생성한다.
# 2) 수행한 결과를 lift기준으로 정렬하여 상위 10개 규칙을 기록한다.
# 3) 연관분석 결과를 LHS와 RHS의 빈도수로 시각화한다.
# 4) 연관분석 결과를 연관어의 네트워크 형태로 시각화한다.
# 5) 연관어 중심 단어를 해설한다



# (1) 최소 support = 0.5, 최소 confidence = 0.9를 지정하여 연관규칙을 생성한다.
data("Adult")
Adult
ruleAdult <- apriori(Adult,parameter=list(supp=0.5, conf=0.9))
ruleAdult

# Parameter specification:
#   confidence minval smax arem  aval originalSupport maxtime support minlen maxlen target  ext
# 0.9    0.1    1 none FALSE            TRUE       5     0.5      1     10  rules TRUE
# 
# Algorithmic control:
#   filter tree heap memopt load sort verbose
# 0.1 TRUE TRUE  FALSE TRUE    2    TRUE
# 
# Absolute minimum support count: 24421 
# 
# set item appearances ...[0 item(s)] done [0.00s].
# set transactions ...[115 item(s), 48842 transaction(s)] done [0.04s].
# sorting and recoding items ... [9 item(s)] done [0.01s].
# creating transaction tree ... done [0.02s].
# checking subsets of size 1 2 3 4 done [0.00s].
# writing ... [52 rule(s)] done [0.00s].
# creating S4 object  ... done [0.00s].



# (2)  수행한 결과를 lift기준으로 정렬하여 상위 10개 규칙을 기록한다.
inspect(head(sort(ruleAdult, by = "lift"),10))

# lhs                                                            rhs                            support   confidence
# [1]  {sex=Male, native-country=United-States}                    => {race=White}                   0.5415421 0.9051090 
# [2]  {sex=Male, capital-loss=None, native-country=United-States} => {race=White}                   0.5113632 0.9032585 
# [3]  {race=White}                                                => {native-country=United-States} 0.7881127 0.9217231 
# [4]  {race=White, capital-loss=None}                             => {native-country=United-States} 0.7490480 0.9205626 
# [5]  {race=White, sex=Male}                                      => {native-country=United-States} 0.5415421 0.9204803 
# [6]  {race=White, capital-gain=None}                             => {native-country=United-States} 0.7194628 0.9202807 
# [7]  {race=White, sex=Male, capital-loss=None}                   => {native-country=United-States} 0.5113632 0.9190124 
# [8]  {race=White, capital-gain=None, capital-loss=None}          => {native-country=United-States} 0.6803980 0.9189249 
# [9]  {workclass=Private, race=White}                             => {native-country=United-States} 0.5433848 0.9144157 
# [10] {workclass=Private, race=White, capital-loss=None}          => {native-country=United-States} 0.5181401 0.9130498 
# coverage  lift     count
# [1]  0.5983170 1.058554 26450
# [2]  0.5661316 1.056390 24976
# [3]  0.8550428 1.027076 38493
# [4]  0.8136849 1.025783 36585
# [5]  0.5883256 1.025691 26450
# [6]  0.7817862 1.025469 35140
# [7]  0.5564268 1.024056 24976
# [8]  0.7404283 1.023958 33232
# [9]  0.5942427 1.018933 26540
# [10] 0.5674829 1.017411 25307



#(3) 연관분석 결과를 LHS와 RHS의 빈도수로 시각화한다.
#install.packages("ggraph",type='binary')
#install.packages("seriation",type='binary')
install.packages("igraph",type='binary')
library(arulesViz)
library(igraph)
plot(ruleAdult)
plot(ruleAdult, method="grouped")

#(4) 연관분석 결과를 연관어의 네트워크 형태로 시각화한다.
plot(ruleAdult, method="graph", control=list(type="items")) 

#(5) 연관어 중심 단어를 해설한다

rules <- labels(ruleAdult, ruleSep=' ')
rules <- sapply(rules,strsplit," ",USE.NAMES = F)
rulematrix <- do.call('rbind',rules)

#rulematrix[c(3:52),]
igraphAdult <- graph.edgelist(rulematrix[c(3:52),],directed=F)
plot.igraph(igraphAdult)
plot.igraph(igraphAdult,vertex.label=(igraphAdult)$name,vertex.label.cex=0.8,vertex.label.color='black',vertex.size=20,vertex.color='green',vertex.frame.color='blue')
#capital-gain,capital-loss 중심으로 연관어가 형성되어 있다.
