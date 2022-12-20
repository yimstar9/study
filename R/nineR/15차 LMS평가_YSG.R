#임성구

#1.(의사결정트리-CART) iris데이터를 이용하여 CART기법 적용(rpart()함수 이용)하여 분류분석 하시오
# (1) 데이터 가져오기 &샘플링
data(iris)
set.seed(1000)
idx <- sample(1:nrow(iris), 0.7*nrow(iris))
train <- iris[idx,]
test <- iris[-idx,]


# (2) 분류모델 생성
library(rpart)
library(rpart.plot)
model <- rpart(Species~.,train)
rpart.plot(model)
# 모델 해석
# 종(Species) 판단
# Petal.Length < 2.5 : setosa로 판단
# Petal.Width >  : versicolor로 판단
# 나머지 versicolor로 판단


# (3) 테스트 데이터를 이용하여 분류
pred <- predict(model,test)


# (4) 예측정확도
library(caret)
library(ModelMetrics)

pred2 <- ifelse(pred[ , 1] >= 0.5, 'setosa',
         ifelse(pred[ , 2] >= 0.5, 'versicolor',
         ifelse(pred[,3]>0.5,'virginica',0)))
pred2<-as.factor(pred2)
caret::confusionMatrix(pred2,test$Species)$overall[1]
# Accuracy 
# 0.9555556 

#또는 직접 계산하는 방법
table(pred2, test$Species)
(12 + 15 + 16) / nrow(test)
#0.95555

#2. (의사결정나무-조건부 추론나무) iris 데이터를 이용하여 조건부 
# 추론나무를 적용(ctree()함수 이용)하여 분류분석 하시오

# (1) 데이터 가져오기&샘플링
data(iris)
set.seed(1000)
idx <- sample(1:nrow(iris), 0.7*nrow(iris))
train2 <- iris[idx,]
test2 <- iris[-idx,]


# (2) 분류모델 생성
library(party)
model2 <- ctree(Species~.,train2)



# (3) 테스트 데이터를 이용하여 분류
pred2 <- predict(model2, test2)



# (4) 시각화

# 예측결과와 실제값 비교
table(pred2, test2$Species) 
#시각화
plot(model2) 

# 결과 해석
# 종(Species) 판단
# Petal.Length <= 1.9 : setosa로 판단
# Petal.Length > 1.9 & Petal.Width <= 1.7 : versicolor로 판단
#나머지: virginica 로 판단


#3. (계층적 군집분석) iris 데이터 셋의 1~4번 변수를 대상으로 유클리드 거리 매트릭스를
# 구하여 idist에 저장한 후 계층적 클러스터링을 적용하여 결과를 시각화 하시오
# (1) 유클리드 거리 계산
library(cluster)
data(iris)
x <- iris[,c(1:4)]
idist <- dist(x, method = "euclidean")


# (2) 계층적 군집 분석(클러스터링)
hc <- hclust(idist)


# (3) 분류결과를 대상으로 음수값을 제거하여 덴드로그램 시각화
plot(hc, hang=-1)


# (4) 그룹 수를 3개로 지정하여 그룹별로 테두리 표시
rect.hclust(hc, k = 3, border ="red")



#4.(K-means 군집분석) iris 데이터에서 specices 컬럼 데이터를 제거한 후 k-means
# clustering을 다음 단계별로 실행하시오

#(1) iris 데이터셋 로딩
data(iris)

#(2) species 데이터 제거
iris$Species <- NULL 

#(3) k-means clustering 실행

# 군집 수 결정
#실루엣 계수로 최적의 군집수 구하기
dist <- dist(iris, method="euclidean")
sil = silhouette(kmeans_result$cluster,dist)
plot(sil)

avg_sil <- function(k, data) {
     result3 <- kmeans(data, centers = k)
     ss <- silhouette(result3$cluster, dist(data))
     avgSil <- mean(ss[, 3])
     return(avgSil)
   }

kClusters <- 2:10
resultForEachK <- data.frame(k = kClusters, silAvg = rep(NA, length(kClusters))) 
for(i in 1:length(kClusters)){
       resultForEachK$silAvg[i] <- avg_sil(kClusters[i], iris)
  }
plot(resultForEachK$k, resultForEachK$silAvg,
           type = "b", pch = 19, frame = FALSE, 
           xlab = "Number of clusters K",
           ylab = "Average Silhouettes")

# 군집 수 2개에서 실루엣 계수 평균이 제일 높으므로 군집수 2개로 결정


kmeans_result <- kmeans(iris, 2) 
kmeans_result



#(4) 군집 결과 시각화
plot(iris[c("Sepal.Length", "Sepal.Width")], col=kmeans_result$cluster,cex=2)
points(kmeans_result$centers[, c("Sepal.Length", "Sepal.Width")], col=2:1, pch=9, cex=7)
plot(iris[c("Petal.Length", "Petal.Width")], col=kmeans_result$cluster,cex=3)
points(kmeans_result$centers[, c("Petal.Length", "Petal.Width")], col=2:1, pch=9, cex=7)


