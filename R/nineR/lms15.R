# 임성구
# 1. iris 데이터를 이용하여 CART 기법 적용(rpart()함수 이용)하여 분류분석 하시오.
# (1) 데이터 가져오기 & 샘플링
# (2) 분류모델 생성
# (3) 테스트 데이터를 이용하여 분류
# (4) 예측정확도

#(1)데이터 가져오기 & 샘플링
library(rpart)
library(rpart.plot)
data(iris)
set.seed(1000) 
idx <- sample(1:nrow(iris),0.7*nrow(iris))
train <- iris[idx,]
test <-  iris[-idx,]

#(2)분류모델 생성
rpart_model <- rpart(Species ~ ., data = train)
rpart_model
rpart.plot(rpart_model)
# 종(Species) 판단
# Petal.Length <= 2.5 : setosa로 판단
# Petal.Width < 1.8 : versicolor로 판단
#나머지: virginica 로 판단

#(3)테스트 데이터를 이용하여 분류
pred <- predict(rpart_model, test)
pred

# 5-2단계: y의 범주로 코딩 변환
pred2 <- ifelse(pred[ , 1] >= 0.5, 'setosa',
                ifelse(pred[ , 2] >= 0.5, 'versicolor',
                       ifelse(pred[,3]>0.5,'virginica',0)))
# 6단계: 모델 평가
table(pred2, test$Species)
(12 + 15 + 16) / nrow(test)


##############################################################################
# (의사결정나무 – 조건부 추론나무)
# 2. iris 데이터를 이용하여 조건부 추론나무 적용(ctree()함수 이용)하여 분류분석 하시오.
# (1) 데이터 가져오기 & 샘플링
# (2) 분류모델 생성
# (3) 테스트 데이터를 이용하여 분류
# (4) 시각화

# 조건부추론나무
#(1) 데이터 가져오기&샘플링
library(party) 
set.seed(1000) 
idx2 <- sample((1:nrow(iris)),0.7*nrow(iris))
train2 <- iris[idx2,] 
test2 <- iris[-idx2, ] 

#(2) 분류모델 생성
ctree_model <- ctree(Species~., data=train2)
#모델예측값과 실제값 비교
table(predict(ctree_model), train2$Species)

#(3) 테스트 데이터를 이용하여 분류
pred_ctree <- predict(ctree_model, newdata=test2)

# 예측결과와 실제값 비교
table(pred_ctree, test2$Species) 

#(4) 시각화
#시각화
plot(ctree_model) 

#---------------------------------------------------
# 결과 해석
# 종(Species) 판단
# Petal.Length <= 1.9 : setosa로 판단
# Petal.Length > 1.9 & Petal.Width <= 1.7 : versicolor로 판단
#나머지: virginica 로 판단
  

  
  
  
  
##############################################################################
# (계층적 군집분석)
# 3. iris데이터 셋의 1~4번 변수를 대상으로 유클리드 거리 매트릭스를 구하여 
# idist에 저장한 후 계층적 클러스터링을 적용하여 결과를 시각화하시오.
# (1) 유클리드 거리 계산
# (2) 계층형 군집 분석(클러스터링)
# (3) 분류결과를 대상으로 음수값을 제거하여 덴드로그램 시각화
# (4) 그룹 수를 3개로 지정하여 그룹별로 테두리 표시

# (1) 유클리드 거리 계산
library(cluster)
data(iris)
x <- iris[,c(1:4)]
idist <- dist(x, method = "euclidean")
idist

#(2) 계층형 군집 분석(클러스터링)
hc <- hclust(dist)

#(3) 분류결과를 대상으로 음수값 제거하여 덴드로그램 시각화
plot(hc, hang=-1)

#(4) 그룹수를 3개로 지정하여 그룹별로 테두리 표시
rect.hclust(hc, k = 3, border ="red")



##############################################################################
# (K-Means 군집분석)
# 4. iris데이터에서 species 컬럼 데이터를 제거한 후 k-means clustering를 다음 
# 단계별로 실행하시오
# (1) iris 데이터셋 로딩
# (2) species 데이터 제거
# (3) k-means clustering 실행
# (4) 군집 결과 시각화


#(1) iris 데이터셋 로딩
data(iris)

#(2) species 데이터 제거
iris$Species <- NULL 

#(3) k-means clustering 실행
kmeans_result <- kmeans(iris, 5) 
kmeans_result
for(i in 1:20){
  cluster_test <-  kmeans(iris, center=i, iter.max=1000)
  test_tot[i]=cluster_test$tot.withinss
}
plot(c(1:20), test_tot,type="b")
#군집수 2개일때 Total within-cluster sum of squares 가 팔꿈치 모양으로 꺽인다. 3개이상부터는 tot.withinss 의 변화가 매우 작음
kmeans_result <- kmeans(iris, 2) #군집수 2개로 다시 모델링

#(4) 군집 결과 시각화
plot(iris[c("Sepal.Length", "Sepal.Width")], col=kmeans_result$cluster)
points(kmeans_result$centers[, c("Sepal.Length", "Sepal.Width")], col=4:1, pch=8, cex=5)
plot(iris[c("Petal.Length", "Petal.Width")], col=kmeans_result$cluster)
points(kmeans_result$centers[, c("Petal.Length", "Petal.Width")], col=4:1, pch=9, cex=5)


