##############################################################################
##1번

#데이터 불러오고 EDA
library(ggplot2)
library(caret)
#install.packages("SyncRNG")


df<-read.csv("dataset2/product.csv")
df
head(df)
summary(df)
str(df)
ggplot(df)+geom_histogram(aes(제품_만족도))
ggplot(df)+geom_histogram(aes(제품_적절성))
pairs(df)


#모델 생성
library(SyncRNG)
v <- 1:nrow(df)
s <- SyncRNG(seed=42)
idx <- s$shuffle(v)[1:round(nrow(df)*0.7)]

idx[1:length(idx)]
train <- df[idx,]
test <- df[-idx,]
str(train)
# 
# idx<-createDataPartition(df$제품_만족도,p=0.7,list=T)
# train<-df[idx$Resample1,]
# test<-df[-idx$Resample1,]
# dim(train)
# dim(test)

m_lm<-lm(제품_만족도~제품_적절성, train[,-1])
m_lm
# Call:
#   lm(formula = 제품_만족도 ~ 제품_적절성, data = train[, -1])
# 
# Coefficients:
#   (Intercept)  제품_적절성  
# 0.7171       0.7607 

summary(m_lm)
#Call:
#   lm(formula = 제품_만족도 ~ 제품_적절성, data = train)
# 
# Residuals:
#   Min       1Q   Median       3Q      Max 
# -1.99346 -0.25635  0.00654  0.26943  1.26943 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  0.78213    0.15151   5.162 6.29e-07 ***
#   제품_적절성  0.73711    0.04679  15.753  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.5564 on 184 degrees of freedom
# Multiple R-squared:  0.5742,	Adjusted R-squared:  0.5719 
# F-statistic: 248.2 on 1 and 184 DF,  p-value: < 2.2e-16

#p value<2.2e-16 이고 설명계수 0.5564
#회귀식 : 제품_만족도=0.74*제품_적절성 + 0.78

#모델 평가
p_lm<-predict(m_lm,test[,-1])
RMSE(p_lm,test[,3])
R2(p_lm,test[,3])


#시각화
ggplot(train,aes(x=제품_적절성,y=제품_만족도))+geom_count()+
      geom_point(color='blue')+
      stat_smooth(method='lm',color='red')+
      geom_text(x=4.3, y=3.5, label="제품_만족도=0.76x제품_적절성+0.72")+
      geom_text(x=4, y=3.3, label="R²=0.50")


################################################################################
#2번
# 비 유무 예측 주제로 R을 이용한 로지스틱 회귀분석을 실시하고, 인공신경망을 
# 이용한 로지스틱 회귀분석을 python으로 실행하여 결과를 비교한다.
library(car)
library(caret)
library(ModelMetrics)
library(corrplot)
df<-read.csv("E:/GoogleDrive/DATASET/dataset4/weather.csv")


####EDA
head(df)
str(df)
summary(df)

df<-na.omit(df) #결측치 제거
cordf<-df[sapply(df,is.numeric)]  #numeric컬럼만 cordf에 넣기
M <- cor(cordf)
corrplot(M, method="circle")
#Temp와minTemp, maxTemp 상관관계가 매우 크다. 다중공선성 가능성 큼

######결측치 제거, 전처리
df<-df[,-1] #날짜 제거
df<-na.omit(df) #결측치 제거
p<-preProcess(df,"range") #모든 숫자형변수 범위0~1 최소-최대 정규화
df<-predict(p,df)

##############################
#모델 생성후 다중 공선성 문제로 다시 전처리 부분
df<-df[,-c(1,2,5,7)]
#####전처리 후 다중공선성 
# Rainfall      Sunshine WindGustSpeed     WindSpeed      Humidity      Pressure         Cloud 
# FALSE         FALSE         FALSE         FALSE         FALSE         FALSE         FALSE 
# Temp     RainToday 
# FALSE         FALSE 
##############################

####모델생성
v <- 1:nrow(df)
s <- SyncRNG(seed=42)
idx <- s$shuffle(v)[1:round(nrow(df)*0.7)]

# idx[1:length(idx)]
# train <- df[idx,]
# test <- df[-idx,]
# 
# idx<-createDataPartition(df$RainTomorrow,p=0.7,list=T)

train<-df[idx,]
test<-df[-idx,]
m_glm<-glm(RainTomorrow~.,train,family = 'binomial')
####Warning message:
####glm.fit: 적합된 확률값들이 0 또는 1 입니다 
### 원인 다중공선성
### 다중공선성 변수 제거해야함
vif(m_glm)
# GVIF    Df GVIF^(1/(2*Df))
# MinTemp       FALSE FALSE           FALSE
# MaxTemp        TRUE FALSE           FALSE
# Rainfall      FALSE FALSE           FALSE
# Sunshine      FALSE FALSE           FALSE
# WindGustDir    TRUE  TRUE           FALSE
# WindGustSpeed FALSE FALSE           FALSE
# WindDir        TRUE  TRUE           FALSE
# WindSpeed     FALSE FALSE           FALSE
# Humidity       TRUE FALSE           FALSE
# Pressure      FALSE FALSE           FALSE
# Cloud         FALSE FALSE           FALSE
# Temp           TRUE FALSE           FALSE
# RainToday     FALSE FALSE           FALSE


summary(m_glm)
# Call:
#   glm(formula = RainTomorrow ~ ., family = "binomial", data = train)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.1641  -0.4185  -0.2097  -0.0936   2.9673  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)   
# (Intercept)    -2.2881     2.5532  -0.896  0.37017   
# Rainfall       -0.8210     1.9925  -0.412  0.68030   
# Sunshine       -2.4072     1.4188  -1.697  0.08975 . 
# WindGustSpeed   7.7124     2.3636   3.263  0.00110 **
# WindSpeed      -4.3129     1.9332  -2.231  0.02569 * 
# Humidity        4.0970     1.9471   2.104  0.03537 * 
# Pressure       -5.5289     1.7796  -3.107  0.00189 **
# Cloud           1.4177     1.0200   1.390  0.16453   
# Temp            1.9017     1.4710   1.293  0.19610   
# RainTodayYes   -0.3432     0.6819  -0.503  0.61477   
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 236.49  on 251  degrees of freedom
# Residual deviance: 135.65  on 242  degrees of freedom
# AIC: 155.65
# 
# Number of Fisher Scoring iterations: 6

######유의한 변수 선택
#후진제거법으로 로지스틱회귀모형을 새롭게 정의
m_glm=step(m_glm, direction = "backward")
# Start:  AIC=155.65
# RainTomorrow ~ Rainfall + Sunshine + WindGustSpeed + WindSpeed + 
#   Humidity + Pressure + Cloud + Temp + RainToday
# 
# Df Deviance    AIC
# - Rainfall       1   135.83 153.83
# - RainToday      1   135.91 153.91
# - Temp           1   137.35 155.35
# - Cloud          1   137.64 155.64
# <none>               135.65 155.65
# - Sunshine       1   138.61 156.61
# - Humidity       1   140.46 158.46
# - WindSpeed      1   141.02 159.02
# - Pressure       1   146.65 164.65
# - WindGustSpeed  1   147.88 165.88
# 
# Step:  AIC=153.83
# RainTomorrow ~ Sunshine + WindGustSpeed + WindSpeed + Humidity + 
#   Pressure + Cloud + Temp + RainToday
# 
# Df Deviance    AIC
# - RainToday      1   136.63 152.63
# - Cloud          1   137.67 153.67
# - Temp           1   137.70 153.70
# <none>               135.83 153.83
# - Sunshine       1   139.07 155.07
# - Humidity       1   140.58 156.58
# - WindSpeed      1   141.20 157.20
# - Pressure       1   146.66 162.66
# - WindGustSpeed  1   148.54 164.54
# 
# Step:  AIC=152.63
# RainTomorrow ~ Sunshine + WindGustSpeed + WindSpeed + Humidity + 
#   Pressure + Cloud + Temp
# 
# Df Deviance    AIC
# - Temp           1   138.43 152.43
# <none>               136.63 152.63
# - Cloud          1   138.76 152.76
# - Sunshine       1   140.16 154.16
# - Humidity       1   140.67 154.67
# - WindSpeed      1   141.70 155.70
# - Pressure       1   146.66 160.66
# - WindGustSpeed  1   149.39 163.39
# 
# Step:  AIC=152.43
# RainTomorrow ~ Sunshine + WindGustSpeed + WindSpeed + Humidity + 
#   Pressure + Cloud
# 
# Df Deviance    AIC
# <none>               138.43 152.43
# - Cloud          1   140.63 152.63
# - Humidity       1   140.72 152.72
# - Sunshine       1   142.11 154.11
# - WindSpeed      1   148.27 160.27
# - WindGustSpeed  1   152.39 164.39
# - Pressure       1   156.69 168.69
summary(m_glm)
# Call:
#   glm(formula = RainTomorrow ~ Sunshine + WindGustSpeed + WindSpeed + 
#         Humidity + Pressure + Cloud, family = "binomial", data = train)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.2317  -0.4318  -0.2244  -0.1064   2.9461  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)    -0.1487     1.8355  -0.081 0.935433    
# Sunshine       -2.6014     1.3771  -1.889 0.058886 .  
# WindGustSpeed   8.1376     2.3437   3.472 0.000516 ***
# WindSpeed      -5.2871     1.7763  -2.976 0.002916 ** 
# Humidity        2.1781     1.4704   1.481 0.138538    
# Pressure       -5.9720     1.5375  -3.884 0.000103 ***
# Cloud           1.4371     0.9835   1.461 0.143933    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 236.49  on 251  degrees of freedom
# Residual deviance: 138.43  on 245  degrees of freedom
# AIC: 152.43
# 
# Number of Fisher Scoring iterations: 6



p_glm<-predict(m_glm,test,type='response')
p.glm<-ifelse(p_glm>=0.5,'Yes','No')
#################################모델평가###########################
table(test$RainTomorrow)
table(p.glm)

# > table(test$RainTomorrow)
# No Yes 
# 88  19 

# > table(p.glm)
# No Yes 
# 94  13 


caret::confusionMatrix(as.factor(p.glm),test$RainTomorrow)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction No Yes
# No  84  10
# Yes  4   9
# 
# Accuracy : 0.8692          
# 95% CI : (0.7902, 0.9266)
# No Information Rate : 0.8224          
# P-Value [Acc > NIR] : 0.1253          
# 
# Kappa : 0.4887          
# 
# Mcnemar's Test P-Value : 0.1814          
#                                           
#             Sensitivity : 0.9545          
#             Specificity : 0.4737          
#          Pos Pred Value : 0.8936          
#          Neg Pred Value : 0.6923          
#              Prevalence : 0.8224          
#          Detection Rate : 0.7850          
#    Detection Prevalence : 0.8785          
#       Balanced Accuracy : 0.7141          
#                                           
#        'Positive' Class : No  

caret::confusionMatrix(as.factor(p.glm),test$RainTomorrow)$byClass
# Sensitivity          Specificity       Pos Pred Value       Neg Pred Value            Precision 
# 0.9545455            0.4736842            0.8936170            0.6923077            0.8936170 
# Recall                   F1           Prevalence       Detection Rate Detection Prevalence 
# 0.9545455            0.9230769            0.8224299            0.7850467            0.8785047 
# Balanced Accuracy 
# 0.7141148 

caret::confusionMatrix(as.factor(p.glm),test$RainTomorrow)$overall
# > caret::confusionMatrix(as.factor(p.glm),test$RainTomorrow)$overall
# Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull AccuracyPValue  McnemarPValue 
# 0.8691589      0.4887372      0.7902236      0.9265834      0.8224299      0.1253417      0.1814492
auc(as.factor(p.glm),test$RainTomorrow)
# 0.7929624


####################시각화##################################
#ROC 곡선 그리기
library(ROCR)
ROCR_p_glm <- prediction(p_glm,list(test$RainTomorrow))
ROCR_pf_glm <- performance(ROCR_p_glm,'tpr','fpr')
plot(ROCR_pf_glm,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
# TPR을 높이고 FPR을 줄이는 임계값을 찾아 선택할 수 있다.

#독립변수들의 Odds ratio 구하기
ORtable=function(x,digits=2){
  suppressMessages(a<-confint(x))
  result=data.frame(exp(coef(x)),exp(a))
  result=round(result,digits)
  result=cbind(result,round(summary(x)$coefficient[,4],3))   #Pr(>|z|)
  colnames(result)=c("OR","2.5%","97.5%","p")
  result
}
ORtable(m_glm)

#                    OR  2.5%     97.5% Pr(>|z|)
# (Intercept)      0.86  0.02     33.42 0.935
# Sunshine         0.07  0.00      1.06 0.059
# WindGustSpeed 3420.80 42.52 448429.37 0.001
# WindSpeed        0.01  0.00      0.14 0.003
# Humidity         8.83  0.53    178.17 0.139
# Pressure         0.00  0.00      0.04 0.000
# Cloud            4.21  0.63     31.16 0.144
####95%신뢰구간, p-value, OR 구하기
confint(m_glm)
coef(m_glm)
str(summary(m_glm)$coefficient)




#################################################
#3번
library(randomForest)
library(SyncRNG)
library(ggplot2)
library(caret)
library(ModelMetrics)
data(iris)
head(iris)
str(iris)
summary(iris)
df<-iris
table(df$Species)


v <- 1:nrow(df)
s <- SyncRNG(seed=38)
idx <- s$shuffle(v)[1:round(nrow(df)*0.7)]
train<-df[idx,]
test<-df[-idx,]
m_rf<-randomForest(Species~.,train,ntree=100,random_state=0)
plot(m_rf)

randomForest::varImpPlot(m_rf)
p_rf<-predict(m_rf, test[,-5])
caret::confusionMatrix(p_rf,test$Species)
caret::confusionMatrix(p_rf,test$Species)$byClass

