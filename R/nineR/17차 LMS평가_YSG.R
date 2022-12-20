#임성구
#(분류분석)
#1. Iris 데이터를 이용하여 인공신경망 기법 또는 xgboost
#기법을 이용하여 분류분석 하시오
#(1)데이터 가져오기 및 샘플링하시오(샘플링시 species별 데이터가 같게)
library(caret)
library(dplyr)
library(nnet)
library(ModelMetrics)
library(ggplot2)
set.seed(1000)

data(iris)
df<-iris
df$Species<-as.factor(df$Species)

#species별 데이터수 같게 7:3으로 train,test셋 나눔
idx<-createDataPartition(df$Species,p=0.7)
train<-df[idx$Resample1,]
test<-df[-idx$Resample1,]

table(train$Species)
# setosa versicolor  virginica 
# 35         35         35 
table(test$Species)
# setosa versicolor  virginica 
# 15         15         15 

#(2)분류모델을 생성하시오
m_nnet<-nnet(Species~.,train,size=3) #은닉층 3개

#(3)테스트 데이터를 이용하여 분류하시오
p_nnet<-predict(m_nnet,subset(test,select=-Species),type="class")


#(4)예측정확도를 산출하시오
table(p_nnet, test$Species)
caret::confusionMatrix(as.factor(p_nnet),test$Species)$overall[1]
#Accuracy : 1 (100%정확도)


#------------------------------------------------------------------
#(분석결과 시각화)
#2. iris데이터를 대상으로 다음 조건에 맞게 시각화 하시오
#(1)1번 컬럼을 x축으로 하고 3번 컬럼을 y축으로 하고 5번 컬럼을 색상 지정하시오.
str(iris)
p<-plot(iris[,1],iris[,3], col =iris[,5])

#(2)차트 제목을 "Scatter plot for iris data"로 설정하시오 
p<-plot(iris[,1],iris[,3], col =iris[,5],main="Scatter plot for iris data")

#(3)작성한 차트를 파일명이 "iris_(본인이니셜).jpg"인 파일에 저장하고 제출
jpeg(filename="iris_(YSG).jpg",width=720,height=480,unit="px",bg="transparent")
p<-plot(iris[,1],iris[,3], col =iris[,5],main="Scatter plot for iris data")
dev.off()


#-------------------------------------------------------------------
#(분석결과 시각화)
#3. diamonds데이터 셋을 대상으로
#(1)x축에 carat변수, y축에 price변수를 지정하고, clarity변수를 
#선 색으로 지정하여 미적 요소 맵핑 객체를 생성하시오

data(diamonds)
str(diamonds)
caret<-diamonds$carat
price<-diamonds$price
colorc <- factor(diamonds$clarity)
q1<- ggplot(data=diamonds, aes(caret, price, color=colorc))+ geom_point()
q1

#(2)산점도 그래프 주변에 부드러운 곡선이 추가되도록
#레이아웃을 추가하시오
q2<-q1  + geom_smooth()
q2

#(3)작성한 차트를 파일명이 "diamonds_(본인영문이니셜).jpg"인
#파일에 저장하고 제출하시오
ggsave("diamonds_(YSG).jpg", scale = 0.5)
