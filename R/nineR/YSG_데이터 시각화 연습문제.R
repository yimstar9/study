# R ch5 연습문제
# 1. iris데이터 셋을 대상으로 다음 조건에 맞게 시각화 하시오
# 1) 1번 컬럼을 x축으로 하고 3번 컬럼을 y축으로 한다.
# 2) 5번 컬럼으로 색상지정한다.
# 3) 차트 제목을 “iris 데이터 산포도”로 추가한다.
# 4) 다음 조건에 맞추어 작성한 차트를 파일에 저장한다.
# - 작업 디렉토리: “C:/Rwork/output”
# - 파일명: “iris.jpg”
# - 크기: 폭(720픽셀), 높이(480픽셀)
data(iris)
df <- as.data.frame(iris[,c(1,3)])
setwd("C:/Rwork/output")
jpeg(filename="iris.jpg",width=720,height=480,unit="px",bg="transparent")
plot(df, col = iris$Species,main="iris 데이터 산포도",cex=2)
dev.off()



# 2. iris3 데이터 셋을 대상으로 다음 조건에 맞게 산점도를 그리시오
# 1) iris3 데이터 셋의 컬럼명을 확인한다.
# 2) iris3 데이터 셋의 구조를 확인한다.
# 3) 꽃의 종별로 산점도 그래프를 그린다.
library(scatterplot3d)
data(iris3)

# 1) iris3 데이터 셋의 컬럼명을 확인한다.
dimnames(iris3)
colnames(iris3)
# 2) iris3 데이터 셋의 구조를 확인한다.
str(iris3)
dim(iris3)

# 3) 꽃의 종별로 산점도 그래프를 그린다.
a<-as.data.frame(iris3[,,1])
b<-as.data.frame(iris3[,,2])
c<-as.data.frame(iris3[,,3])

max <- max(iris3)
min <- min(iris3)
cor(iris3[,,2])
d3 <- scatterplot3d(a$'Petal W.',
                    a$'Sepal L.',
                    a$'Sepal W.',
                    type = 'n',
                    xlab="Peral L", ylab="Sepal L", zlab="Sepal W",
                    highlight.3d=TRUE,
                    xlim=c(min,max),
                    ylim=c(min,max),
                    zlim=c(min,max),
                    
                    angle =70)
    
  d3$points3d(a$'Petal W.',
              a$'Sepal L.',
              a$'Sepal W.',
              bg = 1, pch = 22,cex=1.3)
  d3$points3d(b$'Petal W.',
              b$'Sepal L.',
              b$'Sepal W.',
              bg = 2, pch = 21,cex=1.3)
  
  d3$points3d(c$'Petal W.',
              c$'Sepal L.',
              c$'Sepal W.',
              bg = 3, pch = 24,cex=1.3)


  
# install.packages("plotly")
# library(plotly)
# p <- plot_ly(iris3, x = iris$Sepal.Length, y = iris$Sepal.Width, z = iris$Petal.Length,
#                    marker = list(color = iris$Petal.Width, colorscale = c('#FFE1A1', '#683531'), showscale = TRUE), alpha = 0.5) %>%
#         add_markers() %>%
#         layout(scene = list(xaxis = list(title = 'Sepal.Length'),
#                             yaxis = list(title = 'Sepal.Width'),
#                             zaxis = list(title = 'Petal.Length')),
#                                         annotations = list(
#                                                              x =1.17, #text x위치
#                                                              y = 1.05, #text y위치
#                                                              text = 'Petal.Width',
#                                                              xref = 'paper',
#                                                              yref = 'paper',
#                                                              showarrow = FALSE))
# p
#       
