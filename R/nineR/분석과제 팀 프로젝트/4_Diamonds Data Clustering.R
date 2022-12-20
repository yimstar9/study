#################################################################################
#4. 아래의 조건을 고려하여 군집분석을 실행하시오.
# (1) 데이터: ggplot2 패키지 내 diamonds 데이터
# (2) philentropy::distance() 함수 내 다양한 거리 계산 방법 중 Euclidian거리를 제외한
# 3개를 이용하여 거리 계산 및 사용된 거리에 대한 설명
# (3) 탐색적 목적의 계층적 군집분석 실행
# (4) 군집수 결정 및 결정 사유 설명
# (5) k-means clustering 실행
# (6) 시각화
# (7) 거리 계산 방법에 따른 결과 차이 비교
library(ggplot2)
library(cluster)
library(philentropy)
data(diamonds)
diamonds <- na.omit(diamonds)
set.seed(1000)
t <- sample(1:nrow(diamonds),100)
df <- diamonds[t,]

#(2)philentropy::distance() 함수 내 다양한 거리 계산
x = c(0, 0)
y = c(6,6)

distance(rbind(x, y), method = "manhattan")
distance(rbind(x, y), method = "canberra")
distance(rbind(x, y), method = "chebyshev")
distance(rbind(x, y), method = "jaccard")

#(3) 탐색적 목적의 계층적 군집분석 실행
dist <- distance(df[,-c(2,3,4)], method="euclidean")
hc <- hclust(daisy(dist, metric = "euclidean"))
par(mfrow=c(4,1))
plot(hc,hang = -1)
rect.hclust(hc, k = 3, border ="red")

#(7) 거리 계산 방법에 따른 결과 차이 비교

dist <- distance(df[,-c(2,3,4)], method="manhattan")
dist2 <- distance(df[,-c(2,3,4)], method = "canberra")
dist5 <- distance(df[,-c(2,3,4)], method = "jaccard")

hc <- hclust(daisy(dist),method="single")
hc2 <- hclust(daisy(dist2),method="single")
hc5 <- hclust(daisy(dist5),method="single")

par(mfrow=c(3,1))

plot(hc,hang=-1)
rect.hclust(hc, k = 3, border ="red")

plot(hc2,hang=-1)
rect.hclust(hc2, k = 3, border ="red")

plot(hc5,hang=-1)
rect.hclust(hc5, k = 3, border ="red")


par(mfrow=c(1,2))
agn1 <- agnes(df, metric="manhattan", stand=TRUE)
plot(agn1)
rect.hclust(agn1, k = 5, border ="red")

#
agn2 <- agnes(df, metric="canberra", stand=TRUE)
plot(agn2)
rect.hclust(agn2, k = 5, border ="red")

#
agn3 <- agnes(df, metric="minkowski", stand=TRUE)
plot(agn3)
rect.hclust(agn3, k = 5, border ="red")



# (4) 군집수 결정 및 결정 사유 설명
# 군집 수에 따른 집단 내 제곱합(within-groups sum of squares)의 그래프
# 
# 군집간의 개체간 거리의 제곱합 : 데이터가 얼마나 뭉쳐져있는지
# 뭉쳐져있는 값이 커서도 안되고 너무 작아서도 안됨, 각 객체마다 적절한 withiness를 가져야하며 tot.withiness의 산
# 점도를 그려 거기서 적절한 중간값을 찾는다.

#########엘보우 기법으로 최적 군집수 찾기
test_tot <- as.numeric()
for (i in 1:10){
  result <- kmeans(t,i)
  test_tot[i] <- result$tot.withinss
}
plot(c(1:10),test_tot,type='b')

#########실루엣계수로 최적 군집수 찾기
kmeans_result <- kmeans(t,3)
dist <- dist(t, method="euclidean")
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
  resultForEachK$silAvg[i] <- avg_sil(kClusters[i],t)
}
barplot(resultForEachK$k, resultForEachK$silAvg,
        type = "b", pch = 19, frame = FALSE, 
        xlab = "Number of clusters K",
        ylab = "Average Silhouettes")


# (5) k-means clustering 실행
t <- sample(1:nrow(diamonds), 1000)
test <- diamonds[t, ]
mydia <- test[c("price", "carat", "depth", "table")]
head(mydia)

result2 <- kmeans(mydia,2)
names(result2)
result2$cluster
mydia$cluster <- result2$cluster
head(mydia)
cor(mydia[ , -5], method = "pearson")

# (6) 시각화
plot(mydia[,-5], col=mydia$cluster)
plot(mydia$carat, mydia$price, col = mydia$cluster)
points(result2$centers[ , c("carat", "price")],col = c( 1, 2,3), pch = 9, cex = 10)



