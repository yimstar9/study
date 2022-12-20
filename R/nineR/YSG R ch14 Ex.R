#임성구

#1. 다음은 drinking,water_example.sav 파일의 데이터셋이 구성된 테이블이다. 전체 2 개의 요인에
#의해서 7 개의 변수로 구성되어 있다. 아래에서 제시된 각 단계에 맞게 요인 분석을 수행하시오

#1) 데이터파일 가져오기
library(memisc)
data.spss <- as.data.set(spss.system.file('part3/drinking_water_example.sav')) 
data.spss 
drinkig_water_exam <- data.spss[1:7] 
drinkig_water_exam_df <- as.data.frame(drinkig_water_exam) 
drinkig_water_exam_df
str(drinkig_water_exam_df)

#2) 베리맥스 회전법, 요인수 2, 요인점수 회귀분석 방법을 적용하여 요인 분석
result <- factanal(drinkig_water_exam_df, factors = 2, rotation = "varimax", scores="regression")
result

# 3) 요인적재량 행렬의 컬럼명 변경
colnames(result$loadings) <- c("제품친밀도","제품만족도")
result

# 4) 요인점수를 이용한 요인적재량 시각화
plot(result$scores[,c(1,2)], main="제품친밀도와 제품만족도 요인점수 행렬")
text(result$scores[ , 1], result$scores[ , 2],
     labels = name, cex = 0.6, pos = 1, col = "red")
points(result$loadings[ , c(1,2)], pch = 19, col = "blue")
text(result$loadings[ , 1], result$loadings[ , 2],
     labels = rownames(result$loadings),
     cex = 0.9, pos = 3, col = "blue")

# 5) 요인별 변수 묶기
l1 <- data.frame(drinkig_water_exam_df$Q1, drinkig_water_exam_df$Q2, drinkig_water_exam_df$Q3)
l2 <- data.frame(drinkig_water_exam_df$Q4, drinkig_water_exam_df$Q5, 
                drinkig_water_exam_df$Q6, drinkig_water_exam_df$Q7)
colnames(l1)<-c('q1','q2','q3')
colnames(l2)<-c('q4','q5','q6','q7')

l1_avg<-round(((l1$q1+l1$q2+l1$q3)/ncol(l1)),2);l1_avg
l2_avg<-round(((l2$q4+l2$q5+l2$q6+l2$q7)/ncol(l2)),2);l2_avg

# 2. 1 번에서 생성된 두 개의 요인을 데이터프레임으로 생성한 후 이를 이용하여 두 요인 간의
# 상관관계 계수를 제시하시오
drinking_factor_df <- data.frame(l1_avg, l2_avg)
cor(drinking_factor_df)
