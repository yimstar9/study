#############################################################################
# 3. mlbench패키지 내 BostonHousing 데이터셋을 대상으로 예측기법 2개를 적용하여
# 기법별 결과를 비교하고 시각화하시오. (R과 python 버전으로 모두 실행)
# -종속변수는MEDV 또는CMEDV를사용
library(caret)
library(randomForest)
library(mlbench)
library(ModelMetrics)
library(e1071)

data("BostonHousing")
df <- BostonHousing
write.csv(df,'BostonHousing.csv',row.names = F)

head(df)
str(df)
summary(df)

#스케일링
pre_df<-df[,-c(4,14)]
head(pre_df)
summary(pre_df)
pre <- preProcess(pre_df,'range')
pre_df<-predict(pre,pre_df)
pre_df<-cbind(pre_df,df[,c(4,14)])
head(pre_df)

#train, test 셋 나누기
set.seed(1)
idx <- sample(1:nrow(pre_df),nrow(pre_df)*0.7)
train <- pre_df[idx,]
test <- pre_df[-idx,]

########모델 생성(선형회귀, svr, 랜덤포레스트)
m_lm=lm(medv~.,data=train)
m_svr <- svm(medv~., data=train)
m_rf=randomForest(medv~.,data=train,ntree=100,proximity=T)

p_lm=predict(m_lm,test)
p_svr=predict(m_svr,test)
p_rf=predict(m_rf,test,type="response")

R2(p_lm,test$medv)
rmse(p_lm,test$medv)

R2(p_svr,test$medv)
rmse(p_svr,test$medv)

R2(p_rf,test$medv)
rmse(p_rf,test$medv)

x<-test$medv
y<-p_svr
y2<-p_rf

library(pracma)
plot(x, y,xlab = "",ylab ="", main="SVR Model",col="red")
m= polyfit(x, y)[1]
b= polyfit(x, y)[2]
par(new=TRUE)
plot(x, m * x + b, 'l',axes=F,xlab='실제값',ylab='예측값',col='blue')

plot(x, y2,xlab = "",ylab ="", main="RandomForestRegressor Model",col="red")
m= polyfit(x, y2)[1]
b= polyfit(x, y2)[2]
par(new=TRUE)
plot(x, m * x + b, 'l',axes=F,xlab='실제값',ylab='예측값',col='blue')
